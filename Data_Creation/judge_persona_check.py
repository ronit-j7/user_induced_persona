"""System-prompt check, step 2 (local, needs OPENAI_API_KEY): judge the replies from gen_persona_check.py.

For each persona reply:
  - trait: A/B against the neutral reply to the same scenario, asked from the high end in both orders.
           + prompts should win both times, - prompts should lose both times.
  - helps: does it still genuinely help (no refusal or token answer)?  abusive: flagged, not expected.
Reports per prompt (over the scenarios) and per trait-pole.

    python judge_persona_check.py --gens data/persona_check/persona_check_gens.jsonl
"""

import argparse
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import prompts as P
from gen_user_variants import CONFIG, write_jsonl
from llm import DiskCache, OpenAIJudge, Usage

HERE = Path(__file__).resolve().parent


def judge_one(g, neutral, judge):
    trait, pole = g["trait"], g["pole"]
    c1 = judge.forced_choice(P.persona_forced_messages(trait, neutral["reply"], g["reply"]), P.FORCED_SCHEMA)
    c2 = judge.forced_choice(P.persona_forced_messages(trait, g["reply"], neutral["reply"]), P.FORCED_SCHEMA)
    want = ("B", "A") if pole == "+" else ("A", "B")
    helps = judge.complete_schema(P.helps_messages(g["user"], g["reply"]), P.HELPS_SCHEMA, "helps_check",
                                  judge.cfg["checklist_effort"])
    return dict(g, trait_choices=[c1, c2], trait_pass=(c1, c2) == want, trait_consistent=(c1 != c2),
                helps=helps["helps"]["answer"] == "yes", helps_reason=helps["helps"]["reason"],
                abusive=helps["abusive"]["answer"] == "yes", abusive_reason=helps["abusive"]["reason"],
                len_ratio=round(g["new_tokens"] / max(1, neutral["new_tokens"]), 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gens", default=str(HERE / "data" / "persona_check" / "persona_check_gens.jsonl"))
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    lines = [json.loads(l) for l in open(args.gens)]
    gens = [g for g in lines if "meta" not in g]
    neutral = {g["scenario"]: g for g in gens if g["pole"] == "0"}
    persona = [g for g in gens if g["pole"] != "0"]
    usage, cache = Usage(), DiskCache(HERE / ".cache" / "llm")
    judge = OpenAIJudge(CONFIG["judge"], usage, cache)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(lambda g: judge_one(g, neutral[g["scenario"]], judge), persona))

    out_dir = Path(args.gens).parent
    write_jsonl(out_dir / "persona_check_results.jsonl", results)

    by_prompt, by_pole = defaultdict(list), defaultdict(list)
    for r in results:
        by_prompt[(r["trait"], r["pole"], r["sys_paraphrase"])].append(r)
        by_pole[(r["trait"], r["pole"])].append(r)

    def stats(rs):
        n = len(rs)
        return {"n": n, "trait_pass": sum(r["trait_pass"] for r in rs), "helps": sum(r["helps"] for r in rs),
                "abusive": sum(r["abusive"] for r in rs),
                "mean_len_ratio": round(sum(r["len_ratio"] for r in rs) / n, 2)}

    print(f"\n{'prompt':10s} {'trait ok':>8s} {'helps':>6s} {'abusive':>7s} {'len/neutral':>11s}  system prompt")
    for key in sorted(by_prompt, key=lambda k: (k[0], k[1] == "-", k[2])):
        s = stats(by_prompt[key])
        print(f"{key[0]}{key[1]} k{key[2]:<5d} {s['trait_pass']:>5d}/{s['n']} {s['helps']:>4d}/{s['n']} "
              f"{s['abusive']:>5d}/{s['n']} {s['mean_len_ratio']:>11.2f}  {by_prompt[key][0]['system'][:70]}")
    summary = {"per_trait_pole": {f"{t}{p}": stats(rs) for (t, p), rs in sorted(by_pole.items())},
               "per_prompt": {f"{t}{p}|k{k}": stats(rs) for (t, p, k), rs in sorted(by_prompt.items())},
               "usage": usage.summary()}
    print("\nPer trait-pole:", json.dumps(summary["per_trait_pole"], indent=1))
    print("Usage:", json.dumps(summary["usage"]))
    (out_dir / "persona_check_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Results in {out_dir}")


if __name__ == "__main__":
    main()
