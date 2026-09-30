"""Generate and QC the user-style variants (user_variants.jsonl).

For each (scenario, trait, pole) cell, the rewriter writes 3 paraphrases in one call. Each paraphrase goes through
code checks, the checklist judge and the forced-choice judge (both orders). A failed paraphrase is regenerated on
its own with the feedback from everything that failed, up to --max-attempts in total, and then dropped.

Usage:
    python gen_user_variants.py --dry-run                       # build prompts, no API calls
    python gen_user_variants.py --scenarios code_01,explain_01,emo_01,brainstorm_01,code_02 --traits E
    python gen_user_variants.py                                 # everything: 20 scenarios x E, A x +, -
"""

import argparse
import csv
import difflib
import json
import random
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import prompts as P
from llm import DiskCache, Usage

HERE = Path(__file__).resolve().parent

CONFIG = {
    "rewriter": {"model": "deepseek-flash", "thinking": True, "reasoning_effort": "high",
                 "temperature": 1.0, "max_tokens": 8000},
    "judge": {"model": "gpt-6-luna", "checklist_effort": "low", "forced_choice_effort": "none"},
    "n_paraphrases": 3,
    "max_attempts": 3,
    "length_ratio": [0.5, 2.0],     # in words, relative to u0
    "length_extra_words": 25,       # short originals may grow by up to this many words even if above 2x
    "max_sim_to_original": 0.95,    # difflib ratio; higher means the rewrite barely changed u0
    "max_sim_between": 0.90,        # difflib ratio between paraphrases of the same cell
    "manual_check_frac": 0.10,
    "seed": 0,
    "qwen_tokenizer": "Qwen/Qwen2.5-7B-Instruct",
}

POLES = ["+", "-"]
TRAITS = ["E", "A"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def norm(text):
    return re.sub(r"\s+", " ", text.strip().lower())


def n_words(text):
    return len(text.split())


def similarity(a, b):
    return difflib.SequenceMatcher(None, norm(a), norm(b)).ratio()


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def row_id(scenario, trait, pole, k):
    return f"{scenario}|U|{trait}|{pole}|k{k}"


def git_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=HERE,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


class TokenCounter:
    """Counts Qwen tokens in the user text. Falls back to None if the tokenizer can't be loaded."""

    def __init__(self, name, enabled=True):
        self.tok = None
        if not enabled:
            return
        try:
            from tokenizers import Tokenizer
            self.tok = Tokenizer.from_pretrained(name)
        except Exception as e:
            print(f"[warn] Qwen tokenizer unavailable ({e.__class__.__name__}); user_len_tokens will be null.")

    def __call__(self, text):
        return None if self.tok is None else len(self.tok.encode(text, add_special_tokens=False).ids)


# ---------------------------------------------------------------------------
# QC
# ---------------------------------------------------------------------------

def code_checks(text, u0, others, cfg):
    """Cheap checks that need no LLM. Returns (feedback lines, fatal) where fatal means skip the judge."""
    if not text.strip():
        return ["- Empty: the rewrite is empty."], True
    fails, fatal = [], False
    w0, w = n_words(u0), n_words(text)
    lo, hi = cfg["length_ratio"]
    max_words = max(hi * w0, w0 + cfg["length_extra_words"])
    if w < lo * w0 or w > max_words:
        fails.append(f"- Length: the rewrite has {w} words and the original has {w0}; "
                     f"keep it between {int(lo * w0 + 0.999)} and {int(max_words)} words.")
    if similarity(text, u0) > cfg["max_sim_to_original"]:
        fails.append("- Unchanged: the rewrite is almost identical to the original; make the style visible.")
        fatal = True
    for o in others:
        if similarity(text, o) > cfg["max_sim_between"]:
            fails.append(f'- Duplicate: the rewrite is too similar to another accepted rewrite ("{o}").')
            fatal = True
            break
    return fails, fatal


def evaluate(text, scn, trait, pole, accepted_others, judge, cfg):
    u0 = scn["user"]
    code_fails, fatal = code_checks(text, u0, accepted_others, cfg)
    result = {"code_fails": code_fails, "checklist": None, "forced_choice": None}
    feedback = list(code_fails)
    if fatal:
        result.update(passed=False, feedback="\n".join(feedback))
        return result

    checklist = judge.checklist(P.checklist_messages(trait, scn["intent"], u0, text), P.CHECKLIST_SCHEMA)
    result["checklist"] = checklist
    for cid in P.CHECK_IDS:
        if checklist[cid]["answer"] != "yes":
            feedback.append(f"- {cid} ({P.CHECK_LABELS[cid]}): {checklist[cid]['reason']}")

    # Rewrite as B, then as A. It passes only if chosen in both orders.
    c1 = judge.forced_choice(P.forced_choice_messages(trait, pole, u0, text), P.FORCED_SCHEMA)
    c2 = judge.forced_choice(P.forced_choice_messages(trait, pole, text, u0), P.FORCED_SCHEMA)
    trait_pass = (c1 == "B") and (c2 == "A")
    result["forced_choice"] = {"rewrite_as_B": c1, "rewrite_as_A": c2, "passed": trait_pass}
    if not trait_pass:
        adj = P.POLES[(trait, pole)]["judge_adjectives"]
        feedback.append(f"- Trait: in a side-by-side comparison, the rewrite did not come across as more {adj} "
                        f"than the original. Make the style clearer.")

    result.update(passed=not feedback, feedback="\n".join(feedback))
    return result


# ---------------------------------------------------------------------------
# One cell: (scenario, trait, pole)
# ---------------------------------------------------------------------------

def run_cell(scn, trait, pole, rewriter, judge, cfg, log):
    n, max_attempts = cfg["n_paraphrases"], cfg["max_attempts"]
    accepted = [None] * n     # slot -> {"text", "attempts", "eval"}
    last_fail = {}            # slot -> (text, feedback)

    texts = rewriter.rewrite(P.rewrite_messages(trait, pole, scn["intent"], scn["user"], n=n), n)
    pending = list(enumerate(texts))
    for attempt in range(1, max_attempts + 1):
        for slot, text in pending:
            others = [a["text"] for a in accepted if a is not None]
            ev = evaluate(text, scn, trait, pole, others, judge, cfg)
            log({"scenario": scn["scenario"], "trait": trait, "pole": pole, "slot": slot,
                 "attempt": attempt, "text": text, **ev})
            if ev["passed"]:
                accepted[slot] = {"text": text, "attempts": attempt, "eval": ev}
            else:
                last_fail[slot] = (text, ev["feedback"])
        failed = [s for s in range(n) if accepted[s] is None]
        if not failed or attempt == max_attempts:
            break
        pending = []
        for slot in failed:
            rejected, feedback = last_fail[slot]
            others = [a["text"] for a in accepted if a is not None]
            msgs = P.rewrite_messages(trait, pole, scn["intent"], scn["user"], n=1, rejected_rewrite=rejected,
                                      feedback=feedback, accepted_rewrites=others)
            pending.append((slot, rewriter.rewrite(msgs, 1)[0]))

    dropped = [{"scenario": scn["scenario"], "trait": trait, "pole": pole, "slot": s,
                "last_text": last_fail[s][0], "last_feedback": last_fail[s][1]}
               for s in range(n) if accepted[s] is None]
    return accepted, dropped


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------

def make_row(scn, trait, pole, k, acc, count_tokens):
    fc = acc["eval"]["forced_choice"]
    return {
        "id": row_id(scn["scenario"], trait, pole, k),
        "scenario": scn["scenario"], "domain": scn["domain"], "set": "user_variant",
        "trait": trait, "user_pole": pole, "sys_pole": "0",
        "user_paraphrase": k, "sys_paraphrase": 0,
        "system": P.NEUTRAL_SYSTEM,
        "user": acc["text"],
        "forced_response": scn.get("forced_response"),
        "qc": {"passed": True, "attempts": acc["attempts"],
               "forced_choice": {"rewrite_as_B": fc["rewrite_as_B"], "rewrite_as_A": fc["rewrite_as_A"]}},
        "user_len_tokens": count_tokens(acc["text"]),
    }


def export_manual_check(rows, scen_by_id, qc_dir, frac, seed):
    """Blind sheet for humans (same questions as the judge) plus a separate answer key."""
    rng = random.Random(seed)
    k = max(1, round(frac * len(rows))) if rows else 0
    sample = rng.sample(rows, k)
    sheet, key = [], []
    for r in sample:
        scn = scen_by_id[r["scenario"]]
        rewrite_first = rng.random() < 0.5
        a, b = (r["user"], scn["user"]) if rewrite_first else (scn["user"], r["user"])
        sheet.append({"id": r["id"], "intent": scn["intent"], "original": scn["user"], "rewrite": r["user"],
                      "L1_question": P.LEAKAGE_QUESTION[r["trait"]],
                      **{cid: "" for cid in P.CHECK_IDS},
                      "message_A": a, "message_B": b,
                      "forced_choice_question": f"Which writer comes across as more "
                                                f"{P.POLES[(r['trait'], r['user_pole'])]['judge_adjectives']}?",
                      "choice_A_or_B": "", "notes": ""})
        key.append({"id": r["id"], "rewrite_is": "A" if rewrite_first else "B"})
    for name, data in (("manual_check.csv", sheet), ("manual_check_key.csv", key)):
        if not data:
            continue
        with open(qc_dir / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0].keys()))
            w.writeheader()
            w.writerows(data)
    return len(sample)


def summarize(rows, dropped, attempts_log, cells, cfg):
    n = cfg["n_paraphrases"]
    per_cell = {}
    for r in rows:
        key = f"{r['trait']}{r['user_pole']}"
        per_cell.setdefault(key, 0)
        per_cell[key] += 1
    check_fail = {cid: 0 for cid in P.CHECK_IDS}
    judged = trait_fail = code_fail = 0
    for a in attempts_log:
        if a["code_fails"]:
            code_fail += 1
        if a["checklist"] is not None:
            judged += 1
            for cid in P.CHECK_IDS:
                if a["checklist"][cid]["answer"] != "yes":
                    check_fail[cid] += 1
        if a["forced_choice"] is not None and not a["forced_choice"]["passed"]:
            trait_fail += 1
    len_by_pole = {}
    for r in rows:
        key = f"{r['trait']}{r['user_pole']}"
        len_by_pole.setdefault(key, []).append(n_words(r["user"]))
    incomplete = sorted({f"{d['scenario']}|{d['trait']}|{d['pole']}" for d in dropped})
    return {
        "cells": len(cells), "expected_rows": len(cells) * n, "accepted_rows": len(rows),
        "dropped_rows": len(dropped), "incomplete_cells": incomplete,
        "rows_per_trait_pole": per_cell,
        "attempts_distribution": {str(k): sum(1 for r in rows if r["qc"]["attempts"] == k)
                                  for k in range(1, cfg["max_attempts"] + 1)},
        "evaluations": len(attempts_log), "judged": judged,
        "fail_rate_per_check": {cid: round(v / judged, 3) if judged else None for cid, v in check_fail.items()},
        "trait_forced_choice_fail_rate": round(trait_fail / judged, 3) if judged else None,
        "code_check_fail_rate": round(code_fail / len(attempts_log), 3) if attempts_log else None,
        "mean_words_by_pole": {k: round(sum(v) / len(v), 1) for k, v in len_by_pole.items()},
        "duplicate_user_texts": len(rows) - len({norm(r["user"]) for r in rows}),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def dry_run(cells, cfg, qc_dir):
    """Build the prompts without calling any API. Writes one sample of each prompt and a size estimate."""
    scn0, trait0, pole0 = cells[0]
    samples = {
        "rewrite": P.rewrite_messages(trait0, pole0, scn0["intent"], scn0["user"], n=cfg["n_paraphrases"]),
        "rewrite_retry": P.rewrite_messages(trait0, pole0, scn0["intent"], scn0["user"], n=1,
                                            rejected_rewrite="<rejected>", feedback="- C3 (added a new fact): ...",
                                            accepted_rewrites=["<accepted 1>", "<accepted 2>"]),
        "checklist": P.checklist_messages(trait0, scn0["intent"], scn0["user"], "<variant>"),
        "forced_choice": P.forced_choice_messages(trait0, pole0, scn0["user"], "<variant>"),
    }
    chars = {k: sum(len(m["content"]) for m in v) for k, v in samples.items()}
    qc_dir.mkdir(parents=True, exist_ok=True)
    with open(qc_dir / "dry_run_prompts.txt", "w") as f:
        for name, msgs in samples.items():
            f.write(f"{'=' * 30} {name} ({chars[name]} chars, ~{chars[name] // 4} tokens) {'=' * 30}\n")
            for m in msgs:
                f.write(f"--- {m['role']} ---\n{m['content']}\n")
            f.write("\n")
    n = cfg["n_paraphrases"]
    print(f"Dry run: {len(cells)} cells -> {len(cells) * n} paraphrases on first attempt.")
    print(f"First-attempt calls: {len(cells)} rewriter, {len(cells) * n} checklist, {len(cells) * n * 2} forced-choice.")
    for k, v in chars.items():
        print(f"  {k:14s} prompt ~{v // 4} tokens ({v} chars)")
    print(f"Sample prompts written to {qc_dir / 'dry_run_prompts.txt'}")


def main(argv=None, rewriter=None, judge=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenarios", help="comma-separated scenario ids (default: all)")
    ap.add_argument("--traits", default="E,A")
    ap.add_argument("--poles", default="+,-")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-attempts", type=int, default=CONFIG["max_attempts"])
    ap.add_argument("--scenarios-file", default=str(HERE / "data" / "scenarios.jsonl"))
    ap.add_argument("--out", default=str(HERE / "data" / "user_variants.jsonl"))
    ap.add_argument("--qc-dir", default=str(HERE / "data" / "qc"))
    ap.add_argument("--cache-dir", default=str(HERE / ".cache" / "llm"))
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--no-tokenizer", action="store_true", help="skip Qwen token counts (user_len_tokens = null)")
    ap.add_argument("--dry-run", action="store_true", help="build prompts only, no API calls")
    args = ap.parse_args(argv)

    cfg = json.loads(json.dumps(CONFIG))
    cfg["max_attempts"] = args.max_attempts
    scenarios = load_jsonl(args.scenarios_file)
    if args.scenarios:
        wanted = args.scenarios.split(",")
        unknown = set(wanted) - {s["scenario"] for s in scenarios}
        if unknown:
            sys.exit(f"Unknown scenarios: {sorted(unknown)}")
        scenarios = [s for s in scenarios if s["scenario"] in wanted]
    traits = args.traits.split(",")
    poles = args.poles.split(",")
    cells = [(s, t, p) for s in scenarios for t in traits for p in poles]
    qc_dir = Path(args.qc_dir)

    if args.dry_run:
        dry_run(cells, cfg, qc_dir)
        return

    usage = Usage()
    cache = DiskCache(args.cache_dir, enabled=not args.no_cache)
    if rewriter is None:
        from llm import DeepSeekRewriter
        rewriter = DeepSeekRewriter(cfg["rewriter"], usage, cache)
    if judge is None:
        from llm import OpenAIJudge
        judge = OpenAIJudge(cfg["judge"], usage, cache)
    count_tokens = TokenCounter(cfg["qwen_tokenizer"], enabled=not args.no_tokenizer)

    qc_dir.mkdir(parents=True, exist_ok=True)
    attempts_log, lock = [], threading.Lock()
    attempts_file = open(qc_dir / "attempts.jsonl", "w")

    def log(rec):
        with lock:
            attempts_log.append(rec)
            attempts_file.write(json.dumps(rec, ensure_ascii=False) + "\n")
            attempts_file.flush()

    t0 = time.time()
    results, errors = {}, []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_cell, s, t, p, rewriter, judge, cfg, log): (s["scenario"], t, p)
                   for s, t, p in cells}
        for i, fut in enumerate(as_completed(futures), 1):
            key = futures[fut]
            try:
                results[key] = fut.result()
                ok = sum(a is not None for a in results[key][0])
                print(f"[{i}/{len(cells)}] {'|'.join(key)}: {ok}/{cfg['n_paraphrases']} accepted")
            except Exception as e:
                errors.append({"cell": "|".join(key), "error": f"{e.__class__.__name__}: {e}"})
                print(f"[{i}/{len(cells)}] {'|'.join(key)}: ERROR {e}")
    attempts_file.close()

    rows, dropped = [], []
    for s, t, p in cells:
        key = (s["scenario"], t, p)
        if key not in results:
            continue
        accepted, drop = results[key]
        dropped.extend(drop)
        for k, acc in enumerate(accepted):
            if acc is not None:
                rows.append(make_row(s, t, p, k, acc, count_tokens))

    write_jsonl(Path(args.out), rows)
    write_jsonl(qc_dir / "dropped.jsonl", dropped)
    scen_by_id = {s["scenario"]: s for s in scenarios}
    n_manual = export_manual_check(rows, scen_by_id, qc_dir, cfg["manual_check_frac"], cfg["seed"])
    summary = summarize(rows, dropped, attempts_log, cells, cfg)
    summary["cell_errors"] = errors
    summary["manual_check_rows"] = n_manual
    summary["seconds"] = round(time.time() - t0, 1)
    usage_summary = usage.summary()
    (qc_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (qc_dir / "usage.json").write_text(json.dumps(usage_summary, indent=2))
    (qc_dir / "run_config.json").write_text(json.dumps(
        {"config": cfg, "args": vars(args), "git": git_hash(), "time": time.strftime("%Y-%m-%d %H:%M:%S")},
        indent=2))

    print(json.dumps(summary, indent=2))
    print("Usage:", json.dumps(usage_summary, indent=2))
    print(f"Wrote {len(rows)} rows to {args.out}; QC logs in {qc_dir}")
    return summary


if __name__ == "__main__":
    main()
