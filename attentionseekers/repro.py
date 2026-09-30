"""Upstream Izawa CSV/vector adapter. Project-mode tokenization is separate."""
import csv
from pathlib import Path

import numpy as np

from .chat import EncodedInput
from .heads import head_contributions, spearman


UPSTREAM_COMMIT = "532da0151b319efa99145cdad88035c889d72ef3"


def encode_upstream(tokenizer, row, max_length=2048):
    """Replay upstream's joint prompt+answer IDs and separate prompt length."""
    prompt, answer = row["prompt"], row["answer"]
    ids = tokenizer(prompt + answer, add_special_tokens=False)["input_ids"]
    prefix_length = len(tokenizer.encode(prompt, add_special_tokens=False))
    if not 0 < prefix_length < len(ids) or len(ids) > max_length:
        raise ValueError(f"{row['id']}: invalid upstream prompt/answer token boundary")
    return EncodedInput(tuple(ids), prefix_length, None, (prefix_length, len(ids)))


def load_upstream_csvs(pos_csv, neg_csv, trait, *, threshold=50):
    """Use the exact paired threshold in upstream get_persona_effective."""
    def read(path):
        with open(path, newline="", encoding="utf-8") as file:
            return list(csv.DictReader(file))
    pos, neg = read(pos_csv), read(neg_csv)
    if len(pos) != len(neg):
        raise ValueError("Upstream positive/negative CSVs must have equal row count")
    rows, selected = [], []
    for i, (p, n) in enumerate(zip(pos, neg)):
        for r in (p, n):
            if any(k not in r for k in (trait, "coherence", "prompt", "answer")):
                raise ValueError("Upstream CSV missing required columns")
        if (float(p[trait]) >= threshold and float(n[trait]) < 100 - threshold
                and float(p["coherence"]) >= 50 and float(n["coherence"]) >= 50):
            for pole, value in (("+", p), ("-", n)):
                if not value["prompt"] or not value["answer"]:
                    raise ValueError(f"Empty upstream prompt/answer at original row {i}")
                rows.append({"id": f"upstream|{trait}|{i}|{pole}", "scenario": str(i),
                             "trait": trait, "pole": pole, "prompt": value["prompt"],
                             "answer": value["answer"], "original_csv_row": i})
            selected.append(i)
    if not selected:
        raise ValueError("No upstream CSV pairs passed filtering")
    return rows, {"trait": trait, "threshold": threshold, "selected_csv_rows": selected,
                  "selected_pairs": len(selected), "upstream_commit": UPSTREAM_COMMIT}


def replay_scores(activation_dir, trait, upstream_vectors, output_dir):
    """Compare own independently captured vectors/scores with upstream saved vectors.

    This requires the upstream vector files from its actual Qwen run. Missing
    files are an error; never infer reproduction from our own scores alone.
    """
    import torch
    activation_dir = Path(activation_dir)
    vector_dir = Path(upstream_vectors)
    ours_pre = np.load(activation_dir / "resp.npy", mmap_mode="r").astype(np.float32)
    ours_out = np.load(activation_dir / "output_resp.npy", mmap_mode="r").astype(np.float32)
    if len(ours_pre) % 2:
        raise ValueError("Upstream replay needs paired positive/negative rows")
    own_delta = ours_pre[0::2].mean(axis=0) - ours_pre[1::2].mean(axis=0)
    own_target = ours_out[0::2].mean(axis=0) - ours_out[1::2].mean(axis=0)
    pre_path = vector_dir / f"{trait}_response_avg_diff_attn_pre_o_proj.pt"
    target_path = vector_dir / f"{trait}_response_avg_diff_attn_output.pt"
    if not pre_path.is_file() or not target_path.is_file():
        raise FileNotFoundError("Need both upstream response_avg vector files")
    their_pre = torch.load(pre_path, map_location="cpu", weights_only=True).float().numpy()
    their_out = torch.load(target_path, map_location="cpu", weights_only=True).float().numpy()
    layers, heads, dim = own_delta.shape
    if their_pre.shape != (layers, heads * dim) or their_out.shape != own_target.shape:
        raise ValueError("Upstream vector shapes do not match this model")
    our_scores, upstream_scores = [], []
    for layer in range(layers):
        weight = np.load(activation_dir / "weights" / f"layer_{layer:02d}.npy", allow_pickle=False)
        a, _ = head_contributions(own_delta[layer], weight, own_target[layer])
        b, _ = head_contributions(their_pre[layer].reshape(heads, dim), weight, their_out[layer])
        our_scores.append(a)
        upstream_scores.append(b)
    our_scores, upstream_scores = np.stack(our_scores), np.stack(upstream_scores)
    output_dir = Path(output_dir)
    np.save(output_dir / "replay_own_raw.npy", our_scores)
    np.save(output_dir / "replay_upstream_raw.npy", upstream_scores)
    layer = 19
    if layer >= layers:
        raise ValueError("Qwen SMH layer 19 absent")
    own_top = np.argsort(-our_scores[layer], kind="stable")[:3].tolist()
    upstream_top = np.argsort(-upstream_scores[layer], kind="stable")[:3].tolist()
    rho = spearman(our_scores[layer], upstream_scores[layer])
    return {"upstream_commit": UPSTREAM_COMMIT, "trait": trait, "smh_layer": layer,
            "paper_heads_zero_indexed": [2, 4, 27], "own_top3": own_top,
            "upstream_top3": upstream_top, "paper_heads_recovered": set(upstream_top) == {2, 4, 27},
            "own_matches_upstream": set(own_top) == set(upstream_top),
            "own_vs_upstream_spearman_layer19": rho,
            "extraction_agreement_pass": set(own_top) == set(upstream_top) and rho is not None and rho > 0.95,
            "own_vs_upstream_pre_rmse": float(np.sqrt(np.mean((own_delta.reshape(their_pre.shape) - their_pre) ** 2))),
            "own_vs_upstream_target_rmse": float(np.sqrt(np.mean((own_target - their_out) ** 2))),
            "selected_pairs": len(ours_pre) // 2}
