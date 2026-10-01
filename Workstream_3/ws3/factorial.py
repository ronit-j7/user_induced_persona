"""Orthogonal-coded factorial regression on a head-group activation matrix.

Label shuffling is used only for coefficient norms, never as a null for the cosine.
"""
import json
from pathlib import Path

import numpy as np

CODES = ("A_pole", "A_present", "U", "A_pole_x_U", "A_present_x_U")
BLOCKS = {"A": [0, 1], "U": [2], "AxU": [3, 4]}


def design_row(sys_pole, user_pole):
    c1 = {"+": 1.0, "-": -1.0, "0": 0.0}[sys_pole]
    c2 = {"+": 1.0, "-": 1.0, "0": -2.0}[sys_pole]
    cu = {"+": 1.0, "-": -1.0}[user_pole]
    return np.array([c1, c2, cu, c1 * cu, c2 * cu], dtype=np.float64)


def design_matrix(rows):
    return np.vstack([design_row(row["sys_pole"], row["user_pole"]) for row in rows])


def center_within(values, scenarios):
    out = np.array(values, dtype=np.float64, copy=True)
    scenarios = np.asarray(scenarios)
    for name in np.unique(scenarios):
        idx = np.flatnonzero(scenarios == name)
        out[idx] -= out[idx].mean(axis=0)
    return out


def ols(x, z):
    beta, *_ = np.linalg.lstsq(x, z, rcond=None)
    return beta


def multivariate_r2(x, z, columns):
    ss_tot = float(np.sum(z * z))
    if ss_tot <= 0 or not columns:
        return 0.0
    beta = ols(x[:, columns], z)
    residual = z - x[:, columns] @ beta
    return 1.0 - float(np.sum(residual * residual)) / ss_tot


def cosine(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return None
    return float(np.dot(a, b) / denom)


def fit_centered(x, z, scenarios):
    xc = center_within(x, scenarios)
    zc = center_within(z, scenarios)
    beta = ols(xc, zc)
    r2 = {name: multivariate_r2(xc, zc, cols) for name, cols in BLOCKS.items()}
    r2["full"] = multivariate_r2(xc, zc, list(range(x.shape[1])))
    r2["residual"] = 1.0 - r2["full"]
    r2["block_sum"] = r2["A"] + r2["U"] + r2["AxU"]
    return beta, r2


def scenario_bootstrap(x, z, scenarios, *, resamples=1000, seed=0):
    unique = np.unique(scenarios)
    rng = np.random.default_rng(seed)
    cosines = []
    for _ in range(resamples):
        draw = rng.choice(unique, size=len(unique), replace=True)
        idx = []
        labels = []
        for copy_id, name in enumerate(draw):
            chosen = np.flatnonzero(scenarios == name)
            idx.extend(chosen.tolist())
            labels.extend([f"{name}#{copy_id}"] * len(chosen))
        beta, _r2 = fit_centered(x[idx], z[idx], np.array(labels))
        value = cosine(beta[0], beta[2])
        if value is not None:
            cosines.append(value)
    if not cosines:
        return {"low": None, "high": None, "n": 0}
    low, high = np.quantile(cosines, [0.025, 0.975])
    return {"low": float(low), "high": float(high), "n": len(cosines)}


def split_half(x, z, scenarios, *, seed=0):
    unique = np.array(sorted(set(scenarios.tolist())))
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    half = max(len(unique) // 2, 1)
    groups = unique[:half], unique[half:]
    betas = []
    for group in groups:
        mask = np.isin(scenarios, group)
        beta, _r2 = fit_centered(x[mask], z[mask], scenarios[mask])
        betas.append(beta)
    rel = {}
    for name, row in (("A_pole", 0), ("U", 2)):
        corr = cosine(betas[0][row], betas[1][row])
        rel[name] = None if corr is None else float(2 * corr / (1 + corr))
    return rel


def shuffle_norms(x, z, scenarios, sys_poles, *, shuffles=1000, seed=0):
    """Shuffle c_U inside each scenario x assigned-pole cell. Null for norms, not for the cosine."""
    rng = np.random.default_rng(seed)
    observed, _r2 = fit_centered(x, z, scenarios)
    obs_u = float(np.linalg.norm(observed[2]))
    obs_i = float(np.linalg.norm(observed[3:5]))
    ge_u = 0
    ge_i = 0
    cells = {}
    for i, (scenario, pole) in enumerate(zip(scenarios.tolist(), sys_poles.tolist())):
        cells.setdefault((scenario, pole), []).append(i)
    for _ in range(shuffles):
        shuffled = x.copy()
        for idx in cells.values():
            order = rng.permutation(len(idx))
            taken = shuffled[idx]
            taken[:, 2] = taken[order, 2]
            taken[:, 3] = taken[:, 0] * taken[:, 2]
            taken[:, 4] = taken[:, 1] * taken[:, 2]
            shuffled[idx] = taken
        beta, _r2 = fit_centered(shuffled, z, scenarios)
        ge_u += float(np.linalg.norm(beta[2]) >= obs_u - 1e-8)
        ge_i += float(np.linalg.norm(beta[3:5]) >= obs_i - 1e-8)
    return {
        "norm_U": obs_u,
        "norm_interaction": obs_i,
        "p_norm_U": (ge_u + 1) / (shuffles + 1),
        "p_norm_interaction": (ge_i + 1) / (shuffles + 1),
        "shuffles": shuffles,
    }


def group_matrix(acts, layer, heads):
    parts = [np.asarray(acts[:, layer, head, :], dtype=np.float32) for head in heads]
    return np.concatenate(parts, axis=1)


def analyze_group(rows, z, *, resamples=1000, shuffles=1000, seed=0):
    x = design_matrix(rows)
    scenarios = np.array([row["scenario"] for row in rows])
    poles = np.array([row["sys_pole"] for row in rows])
    beta, r2 = fit_centered(x, z, scenarios)
    signed = cosine(beta[0], beta[2])
    rel = split_half(x, z, scenarios, seed=seed)
    disattenuated = None
    if signed is not None and rel["A_pole"] and rel["U"] and rel["A_pole"] > 0 and rel["U"] > 0:
        disattenuated = float(signed / np.sqrt(rel["A_pole"] * rel["U"]))
    width = z.shape[1]
    return {
        "n": int(len(rows)),
        "width": int(width),
        "r2": r2,
        "cosine_A_pole_U": signed,
        "cosine_ci95": scenario_bootstrap(x, z, scenarios, resamples=resamples, seed=seed),
        "split_half_spearman_brown": rel,
        "disattenuated_cosine": disattenuated,
        "random_baseline_abs_cos": float(np.sqrt(2 / (np.pi * width))),
        "shuffle_norms": shuffle_norms(x, z, scenarios, poles, shuffles=shuffles, seed=seed),
    }


def main(argv=None):
    import argparse
    from attentionseekers.io import load_array, read_jsonl
    parser = argparse.ArgumentParser(description="Factorial regression on one activation bundle")
    parser.add_argument("--acts", required=True)
    parser.add_argument("--trait", required=True, choices=("E", "A"))
    parser.add_argument("--readout", required=True, choices=("first", "resp"))
    parser.add_argument("--controls", required=True, help="metrics.json with smh, top_norm, and random groups")
    parser.add_argument("--out", required=True)
    parser.add_argument("--resamples", type=int, default=1000)
    parser.add_argument("--shuffles", type=int, default=1000)
    args = parser.parse_args(argv)
    rows = [row for row in read_jsonl(Path(args.acts) / "index.jsonl") if row["trait"] == args.trait]
    if not rows:
        raise SystemExit(f"No {args.trait} rows in {args.acts}")
    order = [row["row"] for row in rows]
    acts = load_array(Path(args.acts) / f"{args.readout}.npy")[order]
    controls = json.loads(Path(args.controls).read_text())["controls"]
    groups = {"smh": controls["smh"], "top_norm": controls["top_norm"]}
    for index, heads in enumerate(controls["random"]):
        groups[f"random_{index}"] = heads
    layer = controls["layer"]
    report = {"trait": args.trait, "readout": args.readout, "layer": layer, "groups": {}}
    for name, heads in groups.items():
        print(f"[progress] {name}", flush=True)
        report["groups"][name] = {
            "heads": list(heads),
            **analyze_group(rows, group_matrix(acts, layer, heads),
                            resamples=args.resamples, shuffles=args.shuffles),
        }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    (out / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({name: block["cosine_A_pole_U"] for name, block in report["groups"].items()}, indent=2))


if __name__ == "__main__":
    main()
