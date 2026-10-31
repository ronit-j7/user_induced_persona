"""Cross-fit user-style directions for the projection intervention.

Folds hold out whole scenarios, so the direction used while generating a scenario
never saw that scenario. Each group maps layer -> heads; its direction per layer is
the unit vector of the mean (high - low) user-variant contrast on the concatenated
head slices of the resp readout, and mu0 is the neutral-row mean of the same slices.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

SMH_LAYER = 19
SMH_HEADS = (2, 4, 27)


def assign_folds(rows, n_folds=4):
    """scenario -> fold. Rotating start per domain spreads every domain over all folds."""
    by_domain = {}
    for row in rows:
        by_domain.setdefault(row["domain"], set()).add(row["scenario"])
    folds = {}
    for offset, domain in enumerate(sorted(by_domain)):
        for position, scenario in enumerate(sorted(by_domain[domain])):
            folds[scenario] = (position + offset) % n_folds
    return folds


def random_triple(num_heads=28, exclude=SMH_HEADS, seed=0):
    rng = np.random.default_rng(seed)
    pool = [h for h in range(num_heads) if h not in set(exclude)]
    return sorted(int(h) for h in rng.choice(pool, size=len(exclude), replace=False))


def top_heads(delta, weights, k=10):
    """Top-k heads network-wide by raw head contribution score. Returns {layer: [heads]}."""
    from attentionseekers.heads import head_contributions
    raw = np.stack([head_contributions(delta[layer], weights[layer])[0] for layer in range(len(weights))])
    order = np.argsort(-raw.ravel(), kind="stable")[:k]
    chosen = {}
    for flat in order:
        layer, head = divmod(int(flat), raw.shape[1])
        chosen.setdefault(layer, []).append(head)
    return {layer: sorted(heads) for layer, heads in sorted(chosen.items())}, raw


def group_vectors(delta, mu0, layer_heads):
    """{layer: heads} -> {layer: (heads, unit u, mu0)} on concatenated head slices."""
    out = {}
    for layer, heads in sorted(layer_heads.items()):
        d = np.concatenate([delta[layer, h] for h in heads]).astype(np.float64)
        norm = np.linalg.norm(d)
        if norm == 0:
            raise ValueError(f"Zero contrast at layer {layer} heads {heads}")
        m = np.concatenate([mu0[layer, h] for h in heads]).astype(np.float64)
        out[layer] = (list(heads), (d / norm).astype(np.float32), m.astype(np.float32))
    return out


def fold_inputs(acts, rows, folds, fold, trait="E"):
    """Training-scenario user contrast delta [L,H,D] and neutral mean mu0 [L,H,D]."""
    from attentionseekers.heads import paired_scenario_differences
    diffs, names, _pairs = paired_scenario_differences(acts, rows, trait, "user")
    train = [i for i, name in enumerate(names) if folds[name] != fold]
    if len(train) < 2:
        raise ValueError(f"Fold {fold} leaves fewer than two training scenarios")
    delta = diffs[train].mean(axis=0)
    neutral = [i for i, row in enumerate(rows)
               if row["set"] == "neutral" and folds.get(row["scenario"]) not in (None, fold)]
    if not neutral:
        raise ValueError(f"Fold {fold}: no neutral training rows for mu0")
    mu0 = np.mean(np.asarray(acts[neutral], dtype=np.float32), axis=0)
    return delta, mu0, [names[i] for i in train]


def build_directions(acts, rows, weights, *, n_folds=4, seed=0, top_k=10, trait="E"):
    num_heads = acts.shape[2]
    user_rows = [row for row in rows if row["set"] == "user_variant" and row["trait"] == trait]
    folds = assign_folds(user_rows, n_folds)
    rand3 = random_triple(num_heads, SMH_HEADS, seed)
    groups, stats = {}, {"folds": folds, "rand3_heads": rand3, "per_fold": {}}
    for fold in range(n_folds):
        delta, mu0, train = fold_inputs(acts, rows, folds, fold, trait)
        top, raw = top_heads(delta, weights, top_k)
        layer_order = np.argsort(-raw[SMH_LAYER], kind="stable")
        fold_groups = {
            "smh": group_vectors(delta, mu0, {SMH_LAYER: list(SMH_HEADS)}),
            "l19": group_vectors(delta, mu0, {SMH_LAYER: list(range(num_heads))}),
            "rand3": group_vectors(delta, mu0, {SMH_LAYER: rand3}),
            "top10": group_vectors(delta, mu0, top),
        }
        rng = np.random.default_rng(seed + 1000 + fold)
        random_u = rng.standard_normal(num_heads * acts.shape[3])
        heads, _u, m = fold_groups["l19"][SMH_LAYER]
        fold_groups["randdir19"] = {SMH_LAYER: (heads, (random_u / np.linalg.norm(random_u)).astype(np.float32), m)}
        for name, layers in fold_groups.items():
            groups.setdefault(name, {})[fold] = layers
        stats["per_fold"][str(fold)] = {
            "train_scenarios": train,
            "top10": {str(layer): heads for layer, heads in top.items()},
            "smh_ranks_in_layer": {str(h): int(np.where(layer_order == h)[0][0]) + 1 for h in SMH_HEADS},
            "smh_delta_norm": float(np.linalg.norm(np.concatenate([delta[SMH_LAYER, h] for h in SMH_HEADS]))),
        }
    smh_u = [groups["smh"][f][SMH_LAYER][1] for f in range(n_folds)]
    stats["smh_direction_cross_fold_cosine_min"] = float(min(
        np.dot(a, b) for i, a in enumerate(smh_u) for b in smh_u[i + 1:]))
    return groups, stats


def add_residual_group(groups, stats, resid, rows, layer, *, trait="E", name="resid19"):
    """Cross-fit residual-stream directions on the same folds; heads=[] marks a layer-output hook."""
    folds = stats["folds"]
    for fold in sorted(set(folds.values())):
        delta, mu0, _train = fold_inputs(resid, rows, folds, fold, trait)
        u = delta / np.linalg.norm(delta)
        groups.setdefault(name, {})[fold] = {layer: ([], u.astype(np.float32), mu0.astype(np.float32))}
    stats[f"{name}_cross_fold_cosine_min"] = float(min(
        np.dot(groups[name][a][layer][1], groups[name][b][layer][1])
        for a in groups[name] for b in groups[name] if a < b))
    return groups, stats


def save_directions(groups, stats, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    arrays, spec = {}, {}
    for name, by_fold in groups.items():
        for fold, layers in by_fold.items():
            for layer, (heads, u, mu0) in layers.items():
                arrays[f"{name}|{fold}|{layer}|u"] = u
                arrays[f"{name}|{fold}|{layer}|mu0"] = mu0
                spec.setdefault(name, {}).setdefault(str(fold), {})[str(layer)] = heads
    np.savez(out / "directions.npz", **arrays)
    digest = hashlib.sha256((out / "directions.npz").read_bytes()).hexdigest()
    (out / "directions.json").write_text(json.dumps(
        {"groups": spec, "stats": stats, "npz_sha256": digest}, indent=2) + "\n")
    return out


def load_directions(path):
    """-> (groups[name][fold][layer] = (heads, u, mu0), folds{scenario: fold})."""
    path = Path(path)
    meta = json.loads((path / "directions.json").read_text())
    arrays = np.load(path / "directions.npz")
    groups = {}
    for name, by_fold in meta["groups"].items():
        for fold, layers in by_fold.items():
            for layer, heads in layers.items():
                groups.setdefault(name, {}).setdefault(int(fold), {})[int(layer)] = (
                    heads, arrays[f"{name}|{fold}|{layer}|u"], arrays[f"{name}|{fold}|{layer}|mu0"])
    return groups, meta["stats"]["folds"]


def main(argv=None):
    import argparse
    from attentionseekers.io import load_array, read_jsonl
    parser = argparse.ArgumentParser(description="Cross-fit user-style directions for projection")
    parser.add_argument("--acts", required=True, help="ws2 extract bundle with resp.npy")
    parser.add_argument("--out", required=True)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trait", default="E")
    parser.add_argument("--resid", help="Optional ws3.residual output dir; adds the resid19 group")
    args = parser.parse_args(argv)
    root = Path(args.acts)
    rows = read_jsonl(root / "index.jsonl")
    acts = load_array(root / "resp.npy")
    config = json.loads((root / "config.json").read_text())["model"]
    weights = [load_array(root / "weights" / f"layer_{i:02d}.npy") for i in range(config["num_layers"])]
    groups, stats = build_directions(acts, rows, weights, n_folds=args.folds, seed=args.seed, trait=args.trait)
    if args.resid:
        resid_root = Path(args.resid)
        layer = json.loads((resid_root / "meta.json").read_text())["layer"]
        add_residual_group(groups, stats, np.load(resid_root / "resid.npy"),
                           read_jsonl(resid_root / "index.jsonl"), layer, trait=args.trait)
    save_directions(groups, stats, args.out)
    print(json.dumps({k: v for k, v in stats.items() if k != "folds"}, indent=2))


if __name__ == "__main__":
    main()
