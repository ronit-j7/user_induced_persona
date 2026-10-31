"""Residual-stream (decoder-layer output) readout and projection hook.

Positive control for the head-level arms: if removing the user-style direction from
the whole residual stream at one layer does not reduce mirroring, direction-based
suppression at that layer fails regardless of which heads write it.
"""
from contextlib import contextmanager
import json
from pathlib import Path

import numpy as np


def _hidden(output):
    return output[0] if isinstance(output, tuple) else output


@contextmanager
def layer_output_hook(model, layer, u, mu0, lam):
    """At every position of layer's output: h <- h - lam * u u^T (h - mu0)."""
    import torch
    u = torch.as_tensor(u, dtype=torch.float32).reshape(-1)
    mu0 = torch.as_tensor(mu0, dtype=torch.float32).reshape(-1)
    if abs(float(u.norm()) - 1) > 1e-3:
        raise ValueError("Residual direction must be unit norm")
    placed = {}

    def hook(_module, _args, output):
        hidden = _hidden(output)
        if hidden.shape[-1] != u.numel():
            raise ValueError("Residual direction does not match hidden size")
        if hidden.device not in placed:
            placed[hidden.device] = (u.to(hidden.device), mu0.to(hidden.device))
        uu, mm = placed[hidden.device]
        h = hidden.float()
        edited = (h - lam * ((h - mm) @ uu).unsqueeze(-1) * uu).to(hidden.dtype)
        return (edited,) + tuple(output[1:]) if isinstance(output, tuple) else edited

    handle = model.model.layers[layer].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def extract_residual(model, tokenizer, config, rows, layer):
    """Mean layer output over the (first response_tokens) fixed-reply span -> [N, hidden]."""
    import torch
    from attentionseekers.chat import encode
    device = model.get_input_embeddings().weight.device
    out = np.zeros((len(rows), config.hidden_size), dtype=np.float32)
    for i, row in enumerate(rows):
        encoded = encode(tokenizer, row, config.max_length, response_tokens=config.response_tokens)
        start, end = encoded.response_span
        seen = {}

        def grab(_module, _args, output):
            seen["h"] = _hidden(output)[0, start:end].float().mean(dim=0).cpu().numpy()

        handle = model.model.layers[layer].register_forward_hook(grab)
        try:
            inputs = torch.tensor([encoded.ids], dtype=torch.long, device=device)
            with torch.inference_mode():
                model.model(input_ids=inputs, attention_mask=torch.ones_like(inputs), use_cache=False)
        finally:
            handle.remove()
        if not np.isfinite(seen["h"]).all():
            raise ValueError(f"{row['id']}: nonfinite residual readout")
        out[i] = seen["h"]
        if (i + 1) % 20 == 0 or i + 1 == len(rows):
            print(f"residual {i + 1}/{len(rows)}", flush=True)
    return out


def main(argv=None):
    import argparse
    from attentionseekers.config import load_config
    from attentionseekers.extract import load_model
    from attentionseekers.io import read_jsonl
    parser = argparse.ArgumentParser(description="Residual-stream readout for user variants and neutral rows")
    parser.add_argument("--rows", required=True, help="prepared rows.jsonl")
    parser.add_argument("--out", required=True)
    parser.add_argument("--config", default="Workstream_2/configs/qwen_ada.json")
    parser.add_argument("--layer", type=int, default=19)
    parser.add_argument("--trait", default="E")
    args = parser.parse_args(argv)
    rows = [r for r in read_jsonl(args.rows)
            if r["set"] == "neutral" or (r["set"] == "user_variant" and r["trait"] == args.trait)]
    config = load_config(args.config)
    model, tokenizer = load_model(config)
    acts = extract_residual(model, tokenizer, config, rows, args.layer)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "resid.npy", acts)
    with open(out / "index.jsonl", "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (out / "meta.json").write_text(json.dumps({"layer": args.layer, "n": len(rows), "trait": args.trait}) + "\n")


if __name__ == "__main__":
    main()
