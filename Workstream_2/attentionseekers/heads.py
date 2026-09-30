"""Head contribution, matched contrasts, scenario-level tests, and controls."""
from collections import defaultdict

import numpy as np

from .data import matched_pairs


def head_contributions(delta, weight, target=None):
    """delta [H,D], weight [M,H*D] -> raw scores [H]."""
    delta = np.asarray(delta, dtype=np.float32)
    weight = np.asarray(weight, dtype=np.float32)
    if delta.ndim != 2 or weight.shape[1] != delta.size:
        raise ValueError("Head delta and output projection shape mismatch")
    h, d = delta.shape
    projected = np.einsum("mhd,hd->hm", weight.reshape(weight.shape[0], h, d), delta,
                          optimize=True)
    total = projected.sum(axis=0)
    target = total if target is None else np.asarray(target, dtype=np.float32)
    if target.shape != total.shape:
        raise ValueError("Target attention write shape mismatch")
    scores = projected @ target
    if target is total:
        error = abs(float(scores.sum()) - float(total @ total)) / max(float(total @ total), 1e-12)
        if error > 1e-4:
            raise ValueError(f"Head-score identity failed: {error}")
    return scores.astype(np.float64), projected.astype(np.float32)


def normalize_scores(raw, *, signed_log=False):
    arr = np.asarray(raw, dtype=np.float64)
    if signed_log:
        arr = np.sign(arr) * np.log1p(np.abs(arr))
    mean = arr.mean(axis=-1, keepdims=True)
    std = arr.std(axis=-1, keepdims=True)
    return np.divide(arr - mean, std, out=np.zeros_like(arr), where=std > 0)


def bh_qvalues(p):
    values = np.asarray(p, dtype=np.float64)
    shape = values.shape
    flat = values.ravel()
    if np.any((flat < 0) | (flat > 1) | ~np.isfinite(flat)):
        raise ValueError("p values must be finite and within [0,1]")
    order = np.argsort(flat)
    ranked = flat[order]
    q = np.minimum.accumulate((ranked * len(flat) / np.arange(1, len(flat) + 1))[::-1])[::-1]
    out = np.empty_like(flat)
    out[order] = np.minimum(q, 1)
    return out.reshape(shape)


def ranks_descending(values):
    order = np.argsort(-np.asarray(values), kind="stable")
    ranks = np.empty(len(order), dtype=int)
    ranks[order] = np.arange(1, len(order) + 1)
    return ranks


def spearman(a, b):
    x, y = np.asarray(a).ravel(), np.asarray(b).ravel()
    if len(x) != len(y) or len(x) < 2:
        raise ValueError("Spearman arrays must have same length >= 2")
    def average_ranks(z):
        order = np.argsort(z, kind="stable")
        out = np.empty(len(z), dtype=float)
        sorted_z = z[order]
        boundaries = np.r_[0, np.flatnonzero(sorted_z[1:] != sorted_z[:-1]) + 1, len(z)]
        for lo, hi in zip(boundaries[:-1], boundaries[1:]):
            out[order[lo:hi]] = (lo + hi - 1) / 2
        return out
    rx, ry = average_ranks(x), average_ranks(y)
    if rx.std() == 0 or ry.std() == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def paired_scenario_differences(acts, rows, trait, source):
    """Average paraphrase contrasts per scenario, then return [B,L,H,D]."""
    pairs = matched_pairs(rows, trait, source)
    if len(acts) != len(rows):
        raise ValueError("Activation/index row-count mismatch")
    clusters = defaultdict(list)
    for plus, minus, scenario in pairs:
        clusters[scenario].append(np.asarray(acts[plus], dtype=np.float32) -
                                  np.asarray(acts[minus], dtype=np.float32))
    counts = {len(v) for v in clusters.values()}
    if len(counts) != 1:
        raise ValueError("Unequal paraphrase counts across scenarios")
    names = sorted(clusters)
    return np.stack([np.mean(clusters[name], axis=0, dtype=np.float32) for name in names]), names, pairs


def sign_flip_pvalues(differences, weights, observed, *, permutations=19999, seed=0):
    """One-sided p for positive head scores, with whole scenarios sign flipped."""
    if permutations < 1:
        raise ValueError("At least one permutation required")
    b, layers, heads, dim = differences.shape
    if b < 2:
        raise ValueError("At least two independent scenarios required for inference")
    counts = np.zeros((layers, heads), dtype=np.int32)
    rng = np.random.default_rng(seed)
    # Precompute the scenario-pair bilinear form. For a sign vector s and head h,
    # score_h = s.T @ K_h @ s / B**2. This avoids an O-projection per permutation.
    signs = rng.choice((-1.0, 1.0), size=(permutations, b)).astype(np.float32)
    for layer in range(layers):
        w = np.asarray(weights[layer], dtype=np.float32)
        per_scenario = np.einsum("mhd,bhd->bhm", w.reshape(w.shape[0], heads, dim),
                                 differences[:, layer], optimize=True)
        total = per_scenario.sum(axis=1)
        gram = np.einsum("bhm,cm->hbc", per_scenario, total, optimize=True)
        for start in range(0, permutations, 1024):
            batch = signs[start:start + 1024]
            null_scores = np.einsum("pb,hbc,pc->ph", batch, gram, batch, optimize=True) / (b * b)
            counts[layer] += np.count_nonzero(null_scores >= observed[layer] - 1e-7, axis=0)
    return (counts + 1) / (permutations + 1)


def layer_snr(acts, weights, delta):
    """Write norm of contrast / RMS write norm of centered observations."""
    n, layers, heads, dim = acts.shape
    out = np.zeros(layers, dtype=float)
    for layer in range(layers):
        w = np.asarray(weights[layer], dtype=np.float32)
        signal = w @ np.asarray(delta[layer], dtype=np.float32).reshape(-1)
        samples = np.asarray(acts[:, layer], dtype=np.float32).reshape(n, -1)
        centered = samples - samples.mean(axis=0)
        noise = centered @ w.T
        denom = np.sqrt(np.mean(np.sum(noise * noise, axis=1)))
        out[layer] = np.linalg.norm(signal) / denom if denom > 0 else 0
    return out


def localize(acts, rows, weights, trait, source, *, permutations=19999, seed=0):
    diffs, scenarios, pairs = paired_scenario_differences(acts, rows, trait, source)
    delta = diffs.mean(axis=0)
    layers, heads, dim = delta.shape
    if len(weights) != layers:
        raise ValueError("Need one output projection matrix per layer")
    raw = np.stack([head_contributions(delta[l], weights[l])[0] for l in range(layers)])
    p = sign_flip_pvalues(diffs, weights, raw, permutations=permutations, seed=seed)
    q = bh_qvalues(p)
    selected = sorted({i for a, b, _ in pairs for i in (a, b)})
    snr = layer_snr(acts[selected], weights, delta)
    return {"raw": raw, "raw_z": normalize_scores(raw),
            "signed_log_z": normalize_scores(raw, signed_log=True), "p": p, "q": q,
            "snr": snr, "delta": delta, "n_pairs": len(pairs),
            "n_scenarios": len(scenarios), "scenarios": scenarios}


def control_groups(acts_first, rows, config, *, seed=0, random_groups=5):
    indices = [i for i, row in enumerate(rows) if row["set"] == "neutral"]
    if not indices:
        raise ValueError("Neutral rows required to select highest-norm controls")
    layer = config.smh_layer
    norms = np.linalg.norm(np.asarray(acts_first[indices, layer], dtype=np.float32), axis=-1).mean(axis=0)
    top = np.argsort(-norms, kind="stable")[:len(config.smh_heads)].tolist()
    rng = np.random.default_rng(seed)
    random = [sorted(rng.choice(config.num_heads, len(config.smh_heads), replace=False).tolist())
              for _ in range(random_groups)]
    return {"layer": layer, "smh": list(config.smh_heads), "top_norm": top,
            "random": random, "mean_neutral_head_norms": norms.tolist(),
            "neutral_count": len(indices), "seed": seed}
