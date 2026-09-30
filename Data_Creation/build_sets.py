"""Assemble the sys-twin, neutral and factorial sets from files we already have. No API calls.

    python build_sets.py sys_twins neutral      # needs scenarios.jsonl (with r_b) + prompts/sys_prompts.json
    python build_sets.py factorial              # also needs data/user_variants.jsonl (the full run)

Row schema is the shared team schema (same fields as user_variants.jsonl). Paraphrase indices follow the plan's
convention: user_paraphrase / sys_paraphrase is 0 when that side is neutral, so always filter on `set` before
joining across files.

sys_twins  user = u0, system = one of 5 persona prompts per trait-pole         20 x 2 x 2 x 5 = 400
neutral    user = u0, system = neutral                                         20
factorial  system {+, -, none} x user {+, -}, both traits; system prompt k is
           paired with user paraphrase k (k = 0, 1, 2)                          20 x 2 x 3 x 2 x 3 = 720
forced_response is r_b on every row (WS3 ignores it when free-generating).
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import prompts as P
from gen_user_variants import TokenCounter, load_jsonl, write_jsonl, CONFIG

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
TRAITS, POLES, N_SYS, N_USER = ["E", "A"], ["+", "-"], 5, 3


def load_sys_prompts():
    spec = json.loads((HERE / "prompts" / "sys_prompts.json").read_text())
    assert spec["neutral"] == P.NEUTRAL_SYSTEM
    table = {(p["trait"], p["pole"], p["sys_paraphrase"]): p["system"] for p in spec["prompts"]}
    missing = [(t, p, k) for t in TRAITS for p in POLES for k in range(N_SYS) if (t, p, k) not in table]
    assert not missing, f"missing system prompts: {missing}"
    return table


def row(scn, set_name, rid, trait, user_pole, sys_pole, user_k, sys_k, system, user, qc, count_tokens):
    return {"id": rid, "scenario": scn["scenario"], "domain": scn["domain"], "set": set_name,
            "trait": trait, "user_pole": user_pole, "sys_pole": sys_pole,
            "user_paraphrase": user_k, "sys_paraphrase": sys_k,
            "system": system, "user": user, "forced_response": scn["forced_response"],
            "qc": qc, "user_len_tokens": count_tokens(user)}


def build_sys_twins(scenarios, sys_prompts, count_tokens):
    return [row(s, "sys_twin", f"{s['scenario']}|S|{t}|{p}|k{k}", t, "0", p, 0, k, sys_prompts[(t, p, k)],
                s["user"], {"passed": True, "source": "hand-written system prompt"}, count_tokens)
            for s in scenarios for t in TRAITS for p in POLES for k in range(N_SYS)]


def build_neutral(scenarios, count_tokens):
    return [row(s, "neutral", f"{s['scenario']}|N|_|0|k0", None, "0", "0", 0, 0, P.NEUTRAL_SYSTEM,
                s["user"], {"passed": True, "source": "neutral baseline"}, count_tokens)
            for s in scenarios]


def build_factorial(scenarios, sys_prompts, count_tokens):
    variants = {(r["scenario"], r["trait"], r["user_pole"], r["user_paraphrase"]): r
                for r in load_jsonl(DATA / "user_variants.jsonl")}
    rows, missing = [], []
    for s in scenarios:
        for t in TRAITS:
            for sp in ["+", "-", "0"]:
                for up in POLES:
                    for k in range(N_USER):
                        uv = variants.get((s["scenario"], t, up, k))
                        if uv is None:
                            missing.append(f"{s['scenario']}|{t}|user{up}|k{k}")
                            continue
                        system = P.NEUTRAL_SYSTEM if sp == "0" else sys_prompts[(t, sp, k)]
                        rows.append(row(s, "factorial", f"{s['scenario']}|F|{t}|S{sp}U{up}|k{k}", t, up, sp, k,
                                        k if sp != "0" else 0, system, uv["user"],
                                        {"passed": True, "user_variant_id": uv["id"]}, count_tokens))
    # a dropped user variant removes its row in all 3 system conditions
    return rows, sorted(set(missing))


def check(rows, expected, name):
    ids = [r["id"] for r in rows]
    problems = []
    if len(ids) != len(set(ids)):
        problems.append("duplicate ids")
    if any(not r["forced_response"] for r in rows):
        problems.append("rows without forced_response")
    if any(not r["system"] or not r["user"] for r in rows):
        problems.append("empty system or user text")
    pairs = Counter((r["system"], r["user"]) for r in rows)
    if any(c > 1 for c in pairs.values()):
        problems.append(f"{sum(c > 1 for c in pairs.values())} duplicate (system, user) pairs")
    cells = Counter((r["trait"], r["sys_pole"], r["user_pole"]) for r in rows)
    print(f"{name}: {len(rows)}/{expected} rows | per (trait, sys_pole, user_pole): "
          f"{dict(sorted(cells.items(), key=str))}")
    for p in problems:
        print(f"  PROBLEM: {p}")
    return not problems


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sets", nargs="+", choices=["sys_twins", "neutral", "factorial"])
    ap.add_argument("--no-tokenizer", action="store_true")
    args = ap.parse_args()

    scenarios = load_jsonl(DATA / "scenarios.jsonl")
    if any(s["forced_response"] is None for s in scenarios):
        sys.exit("scenarios.jsonl has null forced_response; generate r_b first.")
    sys_prompts = load_sys_prompts()
    count_tokens = TokenCounter(CONFIG["qwen_tokenizer"], enabled=not args.no_tokenizer)
    ok = True
    if "sys_twins" in args.sets:
        rows = build_sys_twins(scenarios, sys_prompts, count_tokens)
        ok &= check(rows, len(scenarios) * 2 * 2 * N_SYS, "sys_twins")
        write_jsonl(DATA / "sys_twins.jsonl", rows)
    if "neutral" in args.sets:
        rows = build_neutral(scenarios, count_tokens)
        ok &= check(rows, len(scenarios), "neutral")
        write_jsonl(DATA / "neutral.jsonl", rows)
    if "factorial" in args.sets:
        rows, missing = build_factorial(scenarios, sys_prompts, count_tokens)
        ok &= check(rows, len(scenarios) * 2 * 3 * 2 * N_USER, "factorial")
        if missing:
            print(f"  {len(missing)} user variants missing (dropped in QC), each removes 3 rows: {missing}")
        write_jsonl(DATA / "factorial.jsonl", rows)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
