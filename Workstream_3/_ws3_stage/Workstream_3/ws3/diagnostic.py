"""Same-pole paraphrase contrast. A wording-only null for the head-wise significance test."""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def same_pole_differences(acts, rows, trait, pole, *, k_a=0, k_b=1):
    """Pair two system paraphrases of the same assigned pole. Returns [B,L,H,D] and scenario names."""
    groups = defaultdict(dict)
    for index, row in enumerate(rows):
        if row.get("set") != "sys_twin" or row.get("trait") != trait or row.get("sys_pole") != pole:
            continue
        k = row.get("sys_paraphrase")
        if k in (k_a, k_b):
            groups[row["scenario"]][k] = index
    names, diffs = [], []
    for scenario in sorted(groups):
        chosen = groups[scenario]
        if k_a in chosen and k_b in chosen:
            names.append(scenario)
            diffs.append(np.asarray(acts[chosen[k_a]], dtype=np.float32)
                         - np.asarray(acts[chosen[k_b]], dtype=np.float32))
    if len(diffs) < 2:
        raise ValueError(f"Need at least two scenarios for {trait} {pole}")
    return np.stack(diffs), names


def null_localization(acts, rows, weights, trait, pole, *, permutations=19999, seed=0, k_a=0, k_b=1):
    from attentionseekers.heads import bh_qvalues, head_contributions, sign_flip_pvalues
    diffs, names = same_pole_differences(acts, rows, trait, pole, k_a=k_a, k_b=k_b)
    delta = diffs.mean(axis=0)
    raw = np.stack([head_contributions(delta[layer], weights[layer])[0] for layer in range(len(weights))])
    pvalues = sign_flip_pvalues(diffs, weights, raw, permutations=permutations, seed=seed)
    qvalues = bh_qvalues(pvalues)
    return {
        "trait": trait,
        "pole": pole,
        "k_pair": [k_a, k_b],
        "n_scenarios": len(names),
        "scenarios": names,
        "permutations": permutations,
        "fdr_significant": int(np.sum(qvalues < 0.05)),
        "n_heads": int(qvalues.size),
        "max_abs_score": float(np.max(np.abs(raw))),
    }


def main(argv=None):
    import argparse
    from attentionseekers.io import load_array, read_jsonl
    parser = argparse.ArgumentParser(description="Same-pole null contrast for head significance")
    parser.add_argument("--acts", required=True)
    parser.add_argument("--readout", default="resp", choices=("first", "resp", "user"))
    parser.add_argument("--traits", nargs="+", default=["E", "A"])
    parser.add_argument("--poles", nargs="+", default=["+", "-"])
    parser.add_argument("--permutations", type=int, default=19999)
    parser.add_argument("--reference-metrics", nargs="*", default=[],
                        help="Assigned localization metrics.json files to compare against")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    root = Path(args.acts)
    rows = read_jsonl(root / "index.jsonl")
    acts = load_array(root / f"{args.readout}.npy")
    config = json.loads((root / "config.json").read_text())["model"]
    weights = [load_array(root / "weights" / f"layer_{i:02d}.npy") for i in range(config["num_layers"])]
    contrasts = []
    for trait in args.traits:
        for pole in args.poles:
            print(f"[progress] {trait} pole={pole}", flush=True)
            contrasts.append(null_localization(
                acts, rows, weights, trait, pole, permutations=args.permutations))
            print(json.dumps(contrasts[-1]), flush=True)
    references = []
    for path in args.reference_metrics:
        metrics = json.loads(Path(path).read_text())
        references.append({
            "path": str(path),
            "trait": metrics.get("trait"),
            "source": metrics.get("source"),
            "readout": metrics.get("readout"),
            "fdr_significant": metrics.get("fdr_significant"),
        })
    real = [item["fdr_significant"] for item in references if item["fdr_significant"] is not None]
    nulls = [item["fdr_significant"] for item in contrasts]
    if not real:
        conclusion = "Reference trait-contrast counts were not supplied. Null counts stand on their own."
    elif np.median(nulls) > 0.5 * np.median(real):
        conclusion = (
            "Same-pole paraphrases also mark a large share of heads significant, so the test "
            "is sensitive to wording changes and not only to the trait contrast. Prefer ranks "
            "and effect sizes over the significant-head count."
        )
    else:
        conclusion = "Same-pole paraphrases mark substantially fewer heads than the trait contrast."
    report = {"readout": args.readout, "contrasts": contrasts, "references": references,
              "conclusion": conclusion}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    (out / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(conclusion, flush=True)


if __name__ == "__main__":
    main()
