"""Extra assigned-vs-user geometry that ws2 compare does not report, plus side-by-side heatmaps."""
import json
from pathlib import Path

import numpy as np


def group_vector(delta, layer, heads):
    return np.concatenate([np.asarray(delta[layer, head], dtype=np.float64) for head in heads])


def cosine(a, b):
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return None
    return float(np.dot(a, b) / denom)


def random_percentile(delta_a, delta_u, layer, n_heads, group_size, observed, *, draws=1000, seed=0):
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(draws):
        heads = rng.choice(n_heads, size=group_size, replace=False)
        value = cosine(group_vector(delta_a, layer, heads), group_vector(delta_u, layer, heads))
        if value is not None:
            values.append(value)
    values = np.asarray(values)
    return {
        "draws": int(len(values)),
        "observed": observed,
        "percentile_signed": float(np.mean(values <= observed) * 100) if len(values) else None,
        "percentile_abs": float(np.mean(np.abs(values) <= abs(observed)) * 100) if len(values) else None,
        "random_mean_abs": float(np.mean(np.abs(values))) if len(values) else None,
    }


def side_by_side(assigned_z, user_z, path, *, title, smh_layer, smh_heads):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    vmax = max(float(np.nanmax(np.abs(assigned_z))), float(np.nanmax(np.abs(user_z))), 1e-6)
    for ax, matrix, label in ((axes[0], assigned_z, "assigned"), (axes[1], user_z, "user")):
        image = ax.imshow(matrix, cmap="coolwarm", vmin=-vmax, vmax=vmax, aspect="auto")
        for head in smh_heads:
            ax.plot(head, smh_layer, marker="o", markerfacecolor="none",
                    markeredgecolor="black", markersize=9, markeredgewidth=1.4)
        ax.set(xlabel="Query head (0 indexed)", ylabel="Layer (0 indexed)", title=label)
    fig.colorbar(image, ax=axes, label="Within-layer raw z", shrink=0.85)
    fig.suptitle(title)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def geometry(assigned_dir, user_dir, *, draws=1000, seed=0):
    from attentionseekers.io import load_array
    assigned = Path(assigned_dir)
    user = Path(user_dir)
    a_metrics = json.loads((assigned / "metrics.json").read_text())
    u_metrics = json.loads((user / "metrics.json").read_text())
    if a_metrics["trait"] != u_metrics["trait"] or a_metrics["readout"] != u_metrics["readout"]:
        raise ValueError("Assigned and user bundles disagree")
    delta_a = load_array(assigned / "delta.npy")
    delta_u = load_array(user / "delta.npy")
    controls = a_metrics["controls"]
    layer = controls["layer"]
    smh = controls["smh"]
    observed = cosine(group_vector(delta_a, layer, smh), group_vector(delta_u, layer, smh))
    top = cosine(group_vector(delta_a, layer, controls["top_norm"]),
                 group_vector(delta_u, layer, controls["top_norm"]))
    return {
        "trait": a_metrics["trait"],
        "readout": a_metrics["readout"],
        "smh_cosine": observed,
        "top_norm_heads": controls["top_norm"],
        "top_norm_cosine": top,
        "random_percentile": random_percentile(
            delta_a, delta_u, layer, delta_a.shape[1], len(smh), observed, draws=draws, seed=seed),
        "smh_ranks_user": u_metrics["smh_ranks_in_layer"],
        "smh_ranks_assigned": a_metrics["smh_ranks_in_layer"],
        "fdr_assigned": a_metrics["fdr_significant"],
        "fdr_user": u_metrics["fdr_significant"],
    }


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description="Side-by-side heatmaps and SMH-group cosine")
    parser.add_argument("--assigned", required=True)
    parser.add_argument("--user", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--draws", type=int, default=1000)
    args = parser.parse_args(argv)
    from attentionseekers.io import load_array
    report = geometry(args.assigned, args.user, draws=args.draws)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    (out / "geometry.json").write_text(json.dumps(report, indent=2) + "\n")
    side_by_side(
        load_array(Path(args.assigned) / "raw_z.npy"),
        load_array(Path(args.user) / "raw_z.npy"),
        out / "side_by_side.png",
        title=f"{report['trait']} {report['readout']}: assigned vs user",
        smh_layer=json.loads((Path(args.assigned) / "metrics.json").read_text())["smh_layer"],
        smh_heads=json.loads((Path(args.assigned) / "metrics.json").read_text())["smh_heads"],
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
