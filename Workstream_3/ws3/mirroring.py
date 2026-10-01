"""Behavioral mirroring: M_t, Wilcoxon, d_z, scenario bootstrap, domain and facet splits."""
import json
from pathlib import Path

import numpy as np


def _mean(values):
    values = [value for value in values if value is not None]
    if not values:
        return None
    return float(np.mean(values))


def scenario_means(rows, score_key):
    buckets = {}
    for row in rows:
        buckets.setdefault((row["scenario"], row["pole"]), []).append(row[score_key])
    return {key: _mean(values) for key, values in buckets.items()}


def paired_diffs(means, scenarios):
    diffs = []
    used = []
    for scenario in scenarios:
        high = means.get((scenario, "+"))
        low = means.get((scenario, "-"))
        if high is None or low is None:
            continue
        diffs.append(high - low)
        used.append(scenario)
    return np.asarray(diffs, dtype=float), used


def effect(diffs):
    if len(diffs) == 0:
        return {"n": 0, "mean": None, "dz": None, "wilcoxon_p": None}
    mean = float(diffs.mean())
    sd = float(diffs.std(ddof=1)) if len(diffs) > 1 else None
    pvalue = None
    if len(diffs) > 1 and not np.allclose(diffs, 0):
        from scipy.stats import wilcoxon
        pvalue = float(wilcoxon(diffs, zero_method="wilcox", alternative="two-sided").pvalue)
    elif len(diffs) > 1:
        pvalue = 1.0
    return {"n": int(len(diffs)), "mean": mean,
            "dz": None if not sd else mean / sd, "wilcoxon_p": pvalue}


def bootstrap_ci(diffs, *, resamples=5000, seed=0, alpha=0.05):
    if len(diffs) < 2:
        return {"low": None, "high": None, "resamples": 0}
    rng = np.random.default_rng(seed)
    draws = rng.choice(diffs, size=(resamples, len(diffs)), replace=True).mean(axis=1)
    low, high = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
    return {"low": float(low), "high": float(high), "resamples": resamples}


def pole_summary(rows, key):
    out = {}
    for pole in ("+", "-", "0"):
        chosen = [row for row in rows if row["pole"] == pole]
        out[pole] = {
            "n": len(chosen),
            "mean_score": _mean([row.get(key) for row in chosen]) if key else None,
            "mean_tokens": _mean([row.get("n_tokens") for row in chosen]),
            "refusal_rate": _mean([None if row.get("refused") is None else float(row["refused"])
                                   for row in chosen]),
        }
    return out


def mirroring_report(rows, *, resamples=5000, seed=0):
    """rows are judged generations. trait is the manipulated trait; scores are E and A."""
    report = {"status": "COMPLETE", "traits": {}, "length_by_pole": {}, "note": (
        "A- on emotional_advice is a known asymmetry (DECISIONS D9): low agreeableness "
        "toward someone asking for support is close to being unhelpful."
    )}
    manipulated = [row for row in rows if row["trait"] in {"E", "A"}]
    for trait in ("E", "A"):
        subset = [row for row in manipulated if row["trait"] == trait]
        scenarios = sorted({row["scenario"] for row in subset})
        primary = scenario_means(subset, trait)
        diffs, used = paired_diffs(primary, scenarios)
        stats = effect(diffs)
        stats["ci95"] = bootstrap_ci(diffs, resamples=resamples, seed=seed)
        stats["scenarios"] = used
        domains = {}
        for domain in sorted({row["domain"] for row in subset}):
            domain_rows = [row for row in subset if row["domain"] == domain]
            domain_diffs, domain_used = paired_diffs(scenario_means(domain_rows, trait), 
                                                     sorted({row["scenario"] for row in domain_rows}))
            domains[domain] = {**effect(domain_diffs), "scenarios": domain_used,
                               "flag": trait == "A" and domain == "emotional_advice"}
        facets = {}
        for k in sorted({row["k"] for row in subset}):
            facet_rows = [row for row in subset if row["k"] == k]
            facet_diffs, _used = paired_diffs(
                scenario_means(facet_rows, trait), sorted({row["scenario"] for row in facet_rows}))
            facets[str(k)] = effect(facet_diffs)
        cross_key = "A" if trait == "E" else "E"
        cross_diffs, _used = paired_diffs(scenario_means(subset, cross_key), scenarios)
        neutral = [row for row in rows if row["trait"] == "none"]
        report["traits"][trait] = {
            "mirroring": stats,
            "per_domain": domains,
            "per_facet_k": facets,
            "cross_trait_mean_diff": effect(cross_diffs),
            "poles": pole_summary(subset, trait),
            "neutral_mean": _mean([row.get(trait) for row in neutral]),
            "missing_scores": sum(row.get(trait) is None for row in subset),
        }
    for trait in ("E", "A", "none"):
        chosen = [row for row in rows if row["trait"] == trait]
        report["length_by_pole"][trait] = pole_summary(chosen, None)
    if rows and all(row.get("E") is None and row.get("A") is None for row in rows):
        report["status"] = "BLOCKED_NO_JUDGE_SCORES"
    return report


def save_plot(report, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels, means, lows, highs = [], [], [], []
    for trait, block in report["traits"].items():
        stats = block["mirroring"]
        if stats["mean"] is None:
            continue
        labels.append(trait)
        means.append(stats["mean"])
        ci = stats["ci95"]
        lows.append(0 if ci["low"] is None else stats["mean"] - ci["low"])
        highs.append(0 if ci["high"] is None else ci["high"] - stats["mean"])
    fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)
    if labels:
        ax.bar(labels, means, yerr=np.vstack([lows, highs]), capsize=4, color=["#3b6ea5", "#c47b2b"])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set(ylabel="M (high pole minus low pole)", title="Single-turn mirroring")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description="Mirroring statistics from judged generations")
    parser.add_argument("--scores", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--resamples", type=int, default=5000)
    args = parser.parse_args(argv)
    rows = [json.loads(line) for line in Path(args.scores).read_text().splitlines() if line.strip()]
    report = mirroring_report(rows, resamples=args.resamples)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    save_plot(report, out / "mirroring.png")
    print(json.dumps({trait: block["mirroring"] for trait, block in report["traits"].items()}, indent=2))


if __name__ == "__main__":
    main()
