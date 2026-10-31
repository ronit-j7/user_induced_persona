"""Pairwise trait judging: which of two replies is more <trait>? Both orders, logprob-weighted.

Absolute A scores sit near 91 in every condition, so they cannot show mirroring.
Here the replies to the high- and low-pole versions of the same request (same
scenario, paraphrase and sample) are compared directly. The judge sees only the
neutral intent and the two replies, never the user messages.
p_high is P(the high-pole reply is chosen), averaged over the two orders; 0.5 = no mirroring.
"""
import json
from pathlib import Path

import numpy as np

from .judge import _cache_path, call_logprobs, load_api_key
from .mirroring import bootstrap_ci, effect

DESCRIPTIONS = {
    "E": "extraverted (outgoing, energetic, and assertive)",
    "A": "agreeable (warm, polite, and considerate)",
}

PAIR_TEMPLATE = """I am evaluating a different language model. Below are two answers to the same task. Focus only on which answer is more {description}. Ignore length, formatting, and factual correctness.

The task, stated neutrally, was:
[INTENT START]
{intent}
[INTENT END]

[ANSWER A START]
{a}
[ANSWER A END]

[ANSWER B START]
{b}
[ANSWER B END]

Which answer is more {description}? Respond with a single letter, A or B. Don't say anything else."""


def render_pair(trait, intent, a, b):
    if not a.strip() or not b.strip():
        raise ValueError("Pairwise prompt needs two nonempty answers")
    return PAIR_TEMPLATE.format(description=DESCRIPTIONS[trait], intent=intent, a=a, b=b)


def choice_probability(token_probs):
    """P(A) among the A/B mass; None when the judge put almost no mass on either."""
    mass = {"A": 0.0, "B": 0.0}
    for token, prob in token_probs.items():
        key = str(token).strip().upper()
        if key in mass:
            mass[key] += float(prob)
    total = mass["A"] + mass["B"]
    return None if total < 0.25 else mass["A"] / total


def build_pairs(records, trait):
    groups = {}
    for row in records:
        if row["kind"] == "user_variant" and row["trait"] == trait:
            groups.setdefault((row["scenario"], row["k"], row["sample"]), {})[row["pole"]] = row
    pairs = []
    for key, poles in sorted(groups.items()):
        if set(poles) == {"+", "-"}:
            pairs.append({"scenario": key[0], "k": key[1], "sample": key[2], "domain": poles["+"]["domain"],
                          "intent": poles["+"]["intent"], "high": poles["+"]["text"], "low": poles["-"]["text"]})
    return pairs


def summarize(results, *, resamples=5000, seed=0):
    by_scenario = {}
    for row in results:
        if row["p_high"] is not None:
            by_scenario.setdefault(row["scenario"], []).append(row["p_high"])
    centered = np.array([np.mean(v) - 0.5 for _, v in sorted(by_scenario.items())])
    agree = [r for r in results if r["p_high_order1"] is not None and r["p_high_order2"] is not None]
    return {
        "n_pairs": len(results),
        "mean_p_high": None if not len(centered) else float(centered.mean() + 0.5),
        "vs_chance": effect(centered),
        "ci95_minus_half": bootstrap_ci(centered, resamples=resamples, seed=seed),
        "order_consistency": None if not agree else float(np.mean(
            [(r["p_high_order1"] > 0.5) == (r["p_high_order2"] > 0.5) for r in agree])),
        "per_facet_k": {str(k): float(np.mean([r["p_high"] for r in results if r["k"] == k and r["p_high"] is not None]))
                        for k in sorted({r["k"] for r in results})},
    }


def main(argv=None):
    import argparse
    from concurrent.futures import ThreadPoolExecutor
    from openai import OpenAI
    parser = argparse.ArgumentParser(description="Pairwise high-vs-low trait judging")
    parser.add_argument("--generations", required=True, help="gen.jsonl of an unhooked arm")
    parser.add_argument("--trait", choices=sorted(DESCRIPTIONS), required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--ws3-config", default="Workstream_3/configs/ws3.json")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)
    settings = json.loads(Path(args.ws3_config).read_text())
    key, base_url = load_api_key()
    if not key:
        raise SystemExit("OPENAI_API_KEY is not set")
    client = OpenAI(api_key=key, base_url=settings.get("judge_base_url") or base_url)
    model = settings["judge_model"]
    records = [json.loads(line) for line in Path(args.generations).read_text().splitlines() if line.strip()]
    pairs = build_pairs(records, args.trait)

    def ask(prompt):
        payload = {"model": model, "metric": f"pair_{args.trait}", "prompt": prompt, "top_logprobs": 20}
        path = _cache_path(args.cache, payload)
        if path.is_file():
            return json.loads(path.read_text())["p_a"]
        probs, _tokens = call_logprobs(client, model, prompt, top_logprobs=20)
        p_a = choice_probability(probs)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"request": payload, "p_a": p_a, "probs": probs}))
        return p_a

    def judge(pair):
        first = ask(render_pair(args.trait, pair["intent"], pair["high"], pair["low"]))
        second = ask(render_pair(args.trait, pair["intent"], pair["low"], pair["high"]))
        o1 = first
        o2 = None if second is None else 1 - second
        both = [v for v in (o1, o2) if v is not None]
        return {k: pair[k] for k in ("scenario", "k", "sample", "domain")} | {
            "p_high_order1": o1, "p_high_order2": o2, "p_high": float(np.mean(both)) if both else None}

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(judge, pairs))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "pairs.jsonl", "w", encoding="utf-8") as handle:
        for row in results:
            handle.write(json.dumps(row) + "\n")
    summary = {"trait": args.trait, "judge_model": model, **summarize(results)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
