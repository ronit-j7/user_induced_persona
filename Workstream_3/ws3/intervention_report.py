"""Projection arms vs. their own job's baseline.

Mirroring M: per-scenario mean E(user +) - E(user -) with the neutral system.
Factorial cells (system +/- x user +/-), per scenario:
  assigned  = mean over user poles of E(sys +) - E(sys -)   (persona kept? RQ3)
  user      = mean over sys poles of E(user +) - E(user -)  (mirroring under a persona; B4)
  conflict  = E(sys +, user -) - E(sys -, user +)           (persona held against the user)
Arms are compared with the baseline of the same job only (same precision, GPUs, batches).
"""
import json
from pathlib import Path

import numpy as np

from .mirroring import bootstrap_ci, effect, paired_diffs, scenario_means

FACTORIAL_CONTRASTS = ("assigned", "user", "conflict")


def mirroring_by_scenario(rows, key="E"):
    chosen = [row for row in rows if row["kind"] == "user_variant"]
    diffs, used = paired_diffs(scenario_means(chosen, key), sorted({row["scenario"] for row in chosen}))
    return dict(zip(used, diffs))


def factorial_by_scenario(rows, key="E"):
    cells = {}
    for row in rows:
        if row["kind"] == "factorial" and row.get(key) is not None:
            cells.setdefault(row["scenario"], {}).setdefault((row["sys_pole"], row["pole"]), []).append(row[key])
    out = {name: {} for name in FACTORIAL_CONTRASTS}
    for scenario, by_cell in cells.items():
        if set(by_cell) != {("+", "+"), ("+", "-"), ("-", "+"), ("-", "-")}:
            continue
        e = {cell: float(np.mean(values)) for cell, values in by_cell.items()}
        out["assigned"][scenario] = (e["+", "+"] + e["+", "-"] - e["-", "+"] - e["-", "-"]) / 2
        out["user"][scenario] = (e["+", "+"] + e["-", "+"] - e["+", "-"] - e["-", "-"]) / 2
        out["conflict"][scenario] = e["+", "-"] - e["-", "+"]
    return out


def _mean(values):
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def reduction_ci(arm, base, *, resamples=5000, seed=0):
    names = sorted(set(arm) & set(base))
    a = np.array([arm[n] for n in names])
    b = np.array([base[n] for n in names])
    if len(names) < 2 or b.mean() == 0:
        return {"value": None, "low": None, "high": None}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(names), size=(resamples, len(names)))
    draws = 1 - a[idx].mean(axis=1) / b[idx].mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return {"value": float(1 - a.mean() / b.mean()), "low": float(low), "high": float(high)}


def compare(arm, base, *, resamples, seed):
    names = sorted(set(arm) & set(base))
    diffs = np.array([arm[n] - base[n] for n in names])
    return {**effect(diffs), "ci95": bootstrap_ci(diffs, resamples=resamples, seed=seed)}


def summarize(value_by_scenario, *, resamples, seed):
    values = np.array(list(value_by_scenario.values()))
    return {**effect(values), "ci95": bootstrap_ci(values, resamples=resamples, seed=seed)}


def arm_report(rows, base_rows, *, resamples=5000, seed=0):
    kw = {"resamples": resamples, "seed": seed}
    mir, base_mir = mirroring_by_scenario(rows), mirroring_by_scenario(base_rows)
    fact, base_fact = factorial_by_scenario(rows), factorial_by_scenario(base_rows)
    report = {
        "n": len(rows),
        "mirroring": summarize(mir, **kw),
        "mirroring_vs_baseline": compare(mir, base_mir, **kw),
        "mirroring_reduction": reduction_ci(mir, base_mir, **kw),
        "mean_E_user_pole": {p: _mean([r["E"] for r in rows if r["kind"] == "user_variant" and r["pole"] == p])
                             for p in ("+", "-")},
        "coherence": _mean([r["coherence"] for r in rows]),
        "refusal_rate": _mean([None if r.get("refused") is None else float(r["refused"]) for r in rows]),
        "mean_tokens": _mean([r["n_tokens"] for r in rows]),
        "missing_E": sum(r["E"] is None for r in rows),
    }
    for name in FACTORIAL_CONTRASTS:
        if fact[name]:
            report[f"factorial_{name}"] = summarize(fact[name], **kw)
            report[f"factorial_{name}_vs_baseline"] = compare(fact[name], base_fact[name], **kw)
    return report


def load_scores(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def build_report(jobs, *, resamples=5000, seed=0):
    """jobs: {job: {arm: rows}}; each job needs a 'baseline' arm. A-only arms are skipped."""
    report = {"jobs": {}}
    for job, arms in jobs.items():
        if "baseline" not in arms:
            raise ValueError(f"Job {job} has no baseline arm")
        report["jobs"][job] = {arm: arm_report(rows, arms["baseline"], resamples=resamples, seed=seed)
                               for arm, rows in arms.items() if any(r["trait"] == "E" for r in rows)}
    return report


def save_plot(report, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    panels = {"mirroring": "User-style mirroring (neutral system)",
              "factorial_assigned": "Assigned persona effect (factorial)",
              "factorial_user": "User effect under a persona (factorial)"}
    labels, values = [], {key: [] for key in panels}
    for job, arms in report["jobs"].items():
        for arm, block in arms.items():
            labels.append(f"{job}: {arm}")
            for key in panels:
                stats = block.get(key)
                ci = stats["ci95"] if stats else {"low": None, "high": None}
                values[key].append((stats["mean"] if stats else None, ci["low"], ci["high"]))
    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 0.45 * len(labels) + 1.5),
                             sharey=True, constrained_layout=True)
    y = np.arange(len(labels))
    colors = ["#888888" if label.endswith("baseline") else "#3b6ea5" for label in labels]
    for ax, (key, title) in zip(axes, panels.items()):
        means = np.array([v[0] if v[0] is not None else np.nan for v in values[key]])
        low = np.array([v[0] - v[1] if v[1] is not None else 0 for v in values[key]])
        high = np.array([v[2] - v[0] if v[2] is not None else 0 for v in values[key]])
        ax.barh(y, means, xerr=np.vstack([low, high]), color=colors, capsize=3)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("E(high) - E(low), judge points")
        ax.set_title(title)
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _fmt(value):
    return "-" if value is None else f"{value:.2f}"


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description="Projection intervention report")
    parser.add_argument("--job", action="append", required=True,
                        help="NAME=DIR where DIR/<arm>/scores.jsonl are judged arms")
    parser.add_argument("--out", required=True)
    parser.add_argument("--resamples", type=int, default=5000)
    args = parser.parse_args(argv)
    jobs = {}
    for item in args.job:
        name, root = item.split("=", 1)
        jobs[name] = {path.parent.name: load_scores(path) for path in sorted(Path(root).glob("*/scores.jsonl"))}
    report = build_report(jobs, resamples=args.resamples)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    save_plot(report, out / "intervention.png")
    for job, arms in report["jobs"].items():
        for arm, block in arms.items():
            print(f"{job}:{arm:14s} M={_fmt(block['mirroring']['mean'])} "
                  f"reduction={_fmt(block['mirroring_reduction']['value'])} "
                  f"assigned={_fmt(block.get('factorial_assigned', {}).get('mean'))} "
                  f"user|persona={_fmt(block.get('factorial_user', {}).get('mean'))} "
                  f"coherence={_fmt(block['coherence'])}")


if __name__ == "__main__":
    main()
