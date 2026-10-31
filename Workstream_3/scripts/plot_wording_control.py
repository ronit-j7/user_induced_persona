"""Trait contrast vs wording-only change at the style heads (layer 19, heads 2, 4, 27).

Wording-only values are the largest head score anywhere in the model for that same-pole
contrast, so they are an upper bound on what rewording does to any one style head.

    python Workstream_3/scripts/plot_wording_control.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("Workstream_3/ws3_results/ada/jobA/analysis")
OUT = Path("Workstream_3/ws3_results/ada/figures/wording_control.png")
LAYER, HEADS = 19, (2, 4, 27)

fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True, constrained_layout=True)
same_pole = json.loads((ROOT / "same-pole/metrics.json").read_text())["contrasts"]
for ax, trait in zip(axes, ("E", "A")):
    raw = np.load(ROOT / f"{trait}-assigned/raw.npy")
    labels = [f"head {h}\n(trait)" for h in HEADS]
    values = [float(raw[LAYER, h]) for h in HEADS]
    for item in same_pole:
        if item["trait"] == trait:
            labels.append(f"rewording\n(pole {item['pole']})")
            values.append(item["max_abs_score"])
    colors = ["#3b6ea5"] * len(HEADS) + ["#999999"] * (len(values) - len(HEADS))
    ax.bar(labels, values, color=colors)
    ax.set_title({"E": "Extraversion", "A": "Agreeableness"}[trait] + " system prompt")
axes[0].set_ylabel("Head score")
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=160)
print(f"wrote {OUT}")
