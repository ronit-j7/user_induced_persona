"""Reusable Qwen/Llama-style pre-o_proj extraction, batch size one."""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path

import numpy as np

from .chat import describe_encoding, encode
from .data import validate_rows
from .io import finish_run, new_run, provenance, sha256, write_json, write_jsonl


def projection_modules(model, config):
    layers = model.model.layers
    actual = model.config
    expected = {"num_hidden_layers": config.num_layers, "num_attention_heads": config.num_heads,
                "num_key_value_heads": config.num_kv_heads, "hidden_size": config.hidden_size}
    for name, value in expected.items():
        if getattr(actual, name) != value:
            raise ValueError(f"Model {name} mismatch: expected {value}, got {getattr(actual, name)}")
    if len(layers) != config.num_layers:
        raise ValueError("Unexpected number of layers")
    modules = [layer.self_attn.o_proj for layer in layers]
    for module in modules:
        if tuple(module.weight.shape) != (config.hidden_size, config.num_heads * config.head_dim):
            raise ValueError("Unsupported o_proj dimensions")
        if module.bias is not None:
            raise ValueError("Biased output projections are not supported by this score contract")
    return modules


def load_model(config):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.manual_seed(config.seed)
    if config.device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; check driver/sandbox access before running Qwen")
        if config.dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("Configured GPU does not support bf16")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, revision=config.revision, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name, revision=config.revision, torch_dtype=getattr(torch, config.dtype),
        device_map={"": config.device}, low_cpu_mem_usage=True,
        attn_implementation=config.attention_implementation)
    model.eval()
    projection_modules(model, config)
    return model, tokenizer


@contextmanager
def capture_hooks(model, config, spans, *, capture_output=False):
    """Yield dicts of FP32 CPU readouts; remove every hook even on failure.

    spans maps readout name to [start, end) token indices for ONE forward.
    pre[name] is populated with layer -> [query_heads, head_dim].
    output[name] optionally contains independently measured [hidden_size] writes.
    """
    modules = projection_modules(model, config)
    pre, output = {name: {} for name in spans}, {name: {} for name in spans}
    handles = []

    def reduce_tensor(tensor, layer, destination, head_shape):
        if tensor.ndim != 3 or tensor.shape[0] != 1 or tensor.shape[-1] != config.hidden_size:
            raise ValueError("Hooks require [1, tokens, hidden_size]")
        for name, (start, end) in spans.items():
            if not 0 <= start < end <= tensor.shape[1]:
                raise ValueError(f"Invalid {name} token span: {(start, end)}")
            if layer in destination[name]:
                raise ValueError("Hook fired more than once; use a fresh capture context per forward")
            value = tensor[0, start:end].detach().float().mean(dim=0).cpu().numpy()
            if not np.isfinite(value).all():
                raise ValueError("Nonfinite captured activations")
            destination[name][layer] = value.reshape(config.num_heads, config.head_dim) if head_shape else value

    try:
        for layer, module in enumerate(modules):
            handles.append(module.register_forward_pre_hook(
                lambda mod, args, layer=layer: reduce_tensor(args[0], layer, pre, True)))
            if capture_output:
                handles.append(module.register_forward_hook(
                    lambda mod, args, result, layer=layer: reduce_tensor(result, layer, output, False)))
        yield pre, output
        if any(len(values) != config.num_layers for values in pre.values()):
            raise ValueError("Not all layer hooks fired")
    finally:
        for handle in handles:
            handle.remove()


def capture_encoded(model, config, encoded, readouts=("first", "resp", "user"), *, compatibility=False):
    import torch
    if model.training:
        raise ValueError("Call model.eval() before extraction")
    readouts = tuple(readouts)
    if not readouts or len(set(readouts)) != len(readouts) or set(readouts) - {"first", "resp", "user"}:
        raise ValueError("Readouts must be unique members of first,resp,user")
    spans = {"first": (encoded.first, encoded.first + 1), "resp": encoded.response_span,
             "user": encoded.user_span}
    if any(spans[name] is None or spans[name][0] == spans[name][1] for name in readouts):
        raise ValueError("Requested readout has an empty or unavailable span")
    # Prefix-only pass guarantees first/user do not vary with response length.
    groups = [readouts] if compatibility else [tuple(n for n in readouts if n != "resp"),
                                               tuple(n for n in readouts if n == "resp")]
    result, writes = {}, {}
    device = model.get_input_embeddings().weight.device
    for names in groups:
        if not names:
            continue
        ids = encoded.ids if compatibility or "resp" in names else encoded.ids[:encoded.prefix_length]
        inputs = torch.tensor([ids], dtype=torch.long, device=device)
        with capture_hooks(model, config, {name: spans[name] for name in names},
                           capture_output=compatibility) as (pre, output):
            with torch.inference_mode():
                # Base model avoids materializing vocabulary logits for every token.
                model.model(input_ids=inputs, attention_mask=torch.ones_like(inputs), use_cache=False)
        for name in names:
            result[name] = np.stack([pre[name][layer] for layer in range(config.num_layers)])
            if compatibility:
                writes[name] = np.stack([output[name][layer] for layer in range(config.num_layers)])
    return result, writes


def extract_dataset(model, tokenizer, config, rows, out, *, readouts=("first", "resp", "user"),
                    allow_sample=False, data_path=None, compatibility=False, metadata=None):
    """Write a self-contained, immutable activation bundle; return its directory."""
    if compatibility:
        from .repro import encode_upstream
        encoder = encode_upstream
        if "user" in readouts:
            raise ValueError("Upstream CSV replay does not define a user span")
        sample = False
    else:
        encoder = encode
        sample = validate_rows(rows, allow_sample=allow_sample, require_response="resp" in readouts)["is_sample"]
    # Check all token boundaries before creating arrays or invoking the model.
    encoded_rows = [encoder(tokenizer, row, config.max_length) if compatibility else
                    encoder(tokenizer, row, config.max_length, response_tokens=config.response_tokens)
                    for row in rows]
    path = new_run(out)
    config_snapshot = {"schema_version": 1, "model": config.as_dict(), "readouts": list(readouts),
                       "mode": "upstream_replay" if compatibility else "project",
                       "is_sample": sample, "provenance": provenance(), "metadata": metadata or {},
                       "is_synthetic": config.is_synthetic,
                       "response_tokens": None if compatibility else config.response_tokens,
                       "data_sha256": sha256(data_path) if data_path else None,
                       "resolved_model_revision": getattr(model.config, "_commit_hash", None),
                       "tokenizer_class": type(tokenizer).__name__,
                       "resolved_tokenizer_revision": getattr(tokenizer, "init_kwargs", {}).get("_commit_hash"),
                       "chat_template_sha256": hashlib.sha256(str(tokenizer.chat_template).encode()).hexdigest()}
    write_json(path / "config.json", config_snapshot)
    (path / "weights").mkdir()
    for layer, module in enumerate(projection_modules(model, config)):
        np.save(path / "weights" / f"layer_{layer:02d}.npy", module.weight.detach().float().cpu().numpy())
    shape = (len(rows), config.num_layers, config.num_heads, config.head_dim)
    arrays = {name: np.lib.format.open_memmap(path / f"{name}.npy", mode="w+", dtype=np.float16, shape=shape)
              for name in readouts}
    writes = {name: np.lib.format.open_memmap(path / f"output_{name}.npy", mode="w+", dtype=np.float32,
                                            shape=(len(rows), config.num_layers, config.hidden_size))
              for name in readouts} if compatibility else {}
    index = []
    for i, (row, encoded) in enumerate(zip(rows, encoded_rows)):
        values, output = capture_encoded(model, config, encoded, readouts, compatibility=compatibility)
        for name in readouts:
            if np.max(np.abs(values[name])) > np.finfo(np.float16).max:
                raise ValueError("Activations overflow FP16 storage")
            arrays[name][i] = values[name]
            if compatibility:
                writes[name][i] = output[name]
        detail = describe_encoding(tokenizer, encoded)
        index.append({**row, "row": i, **detail,
                      "user_len_tokens": (encoded.user_span[1] - encoded.user_span[0]) if encoded.user_span else None})
        if i == 0 or (i + 1) % 10 == 0 or i + 1 == len(rows):
            print(f"extracted {i + 1}/{len(rows)}", flush=True)
    for array in [*arrays.values(), *writes.values()]:
        array.flush()
    write_jsonl(path / "index.jsonl", index)
    finish_run(path, {"kind": "activations", "mode": config_snapshot["mode"], "is_sample": sample,
                      "is_synthetic": config.is_synthetic, "response_tokens": config_snapshot["response_tokens"],
                      "shape": list(shape), "readouts": list(readouts)})
    return path
