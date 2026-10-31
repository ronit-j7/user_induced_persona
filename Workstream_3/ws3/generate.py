"""Resumable batched free generation. Batch seeds are fixed by batch index, so a rerun matches."""
from contextlib import contextmanager
import json
from pathlib import Path
import time

from .jobs import build_jobs, public_record


def read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_done(path):
    path = Path(path)
    if not path.is_file():
        return {}
    return {row["id"]: row for row in read_jsonl(path)}


def batches(jobs, batch_size):
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    return [(index, jobs[start:start + batch_size])
            for index, start in enumerate(range(0, len(jobs), batch_size))]


@contextmanager
def zero_head_hook(model, layer, heads, head_dim=128):
    """Zero selected query-head slices of o_proj's input at every position."""
    module = model.model.layers[layer].self_attn.o_proj
    heads = [int(h) for h in heads]

    def pre_hook(_module, args):
        hidden = args[0]
        if hidden.shape[-1] < (max(heads) + 1) * head_dim:
            raise ValueError("Hidden width is smaller than the requested head slice")
        edited = hidden.clone()
        for head in heads:
            start = head * head_dim
            edited[:, :, start:start + head_dim] = 0
        return (edited,) + tuple(args[1:])

    handle = module.register_forward_pre_hook(pre_hook)
    try:
        yield
    finally:
        handle.remove()


@contextmanager
def projection_hook(model, groups, lam, head_dim=128):
    """Remove lam times the component along u from selected o_proj input head slices.

    groups maps layer -> (heads, u, mu0); u and mu0 have len(heads) * head_dim entries
    in head order. At every position: z <- z - lam * u u^T (z - mu0).
    """
    import torch
    handles = []
    try:
        for layer, (heads, u, mu0) in sorted(groups.items()):
            heads = [int(h) for h in heads]
            cols = torch.cat([torch.arange(h * head_dim, (h + 1) * head_dim) for h in heads])
            u = torch.as_tensor(u, dtype=torch.float32).reshape(-1)
            mu0 = torch.as_tensor(mu0, dtype=torch.float32).reshape(-1)
            if u.numel() != cols.numel() or mu0.numel() != cols.numel():
                raise ValueError(f"Layer {layer}: direction size does not match {len(heads)} heads")
            if abs(float(u.norm()) - 1) > 1e-3:
                raise ValueError(f"Layer {layer}: direction must be unit norm")
            # Sharded models keep layers on different GPUs; cache per device.
            placed = {}

            def pre_hook(_module, args, cols=cols, u=u, mu0=mu0, placed=placed):
                hidden = args[0]
                if hidden.device not in placed:
                    placed[hidden.device] = tuple(t.to(hidden.device) for t in (cols, u, mu0))
                c, uu, mm = placed[hidden.device]
                z = hidden[..., c].float()
                coef = (z - mm) @ uu
                edited = hidden.clone()
                edited[..., c] = (z - lam * coef.unsqueeze(-1) * uu).to(hidden.dtype)
                return (edited,) + tuple(args[1:])

            module = model.model.layers[layer].self_attn.o_proj
            handles.append(module.register_forward_pre_hook(pre_hook))
        yield
    finally:
        for handle in handles:
            handle.remove()


def generate_batch(model, tokenizer, jobs, *, max_new_tokens, temperature, top_p):
    import torch
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    prompts = []
    for job in jobs:
        messages = [
            {"role": "system", "content": job["system"]},
            {"role": "user", "content": job["user"]},
        ]
        prompts.append(tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True))
    encoded = tokenizer(prompts, return_tensors="pt", padding=True)
    device = model.get_input_embeddings().weight.device
    encoded = {key: value.to(device) for key, value in encoded.items()}
    prompt_len = encoded["input_ids"].shape[1]
    # temperature 0 means greedy decoding (deterministic, so arms differ only by the hook).
    sampling = {"do_sample": True, "temperature": temperature, "top_p": top_p} if temperature > 0 else \
        {"do_sample": False, "temperature": None, "top_p": None, "top_k": None}
    with torch.inference_mode():
        output = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            **sampling,
        )
    records = []
    for row, job in enumerate(jobs):
        new_ids = output[row, prompt_len:]
        if tokenizer.pad_token_id is not None:
            keep = new_ids != tokenizer.pad_token_id
            new_ids = new_ids[keep]
        text = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        records.append(public_record(job, text, int(new_ids.shape[0])))
    return records


def run_generation(jobs, out, config_path, *, batch_size=4, seed=0, max_new_tokens=300,
                   temperature=1.0, top_p=1.0, hook=None, model=None, tokenizer=None):
    import torch
    from attentionseekers.config import load_config
    from attentionseekers.extract import load_model

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "gen.jsonl"
    done = load_done(path)
    pending = []
    dropped = set()
    for index, batch in batches(jobs, batch_size):
        ids = {job["id"] for job in batch}
        if ids <= done.keys():
            continue
        dropped |= ids & done.keys()
        pending.append((index, batch))
    if dropped:
        done = {key: value for key, value in done.items() if key not in dropped}
        ordered = [done[job["id"]] for job in jobs if job["id"] in done]
        write_jsonl(path, ordered)
    if not pending:
        print(f"[progress] {len(jobs)}/{len(jobs)} already complete", flush=True)
        return path
    if model is None:
        model, tokenizer = load_model(load_config(config_path))
    started = time.time()
    finished = len(jobs) - sum(len(batch) for _, batch in pending)
    context = hook(model) if hook is not None else _null()
    with context:
        for index, batch in pending:
            torch.manual_seed(seed + index)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed + index)
            records = generate_batch(
                model, tokenizer, batch, max_new_tokens=max_new_tokens,
                temperature=temperature, top_p=top_p)
            with open(path, "a", encoding="utf-8") as handle:
                for record in records:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            finished += len(batch)
            elapsed = time.time() - started
            rate = (finished - (len(jobs) - sum(len(b) for _, b in pending))) / max(elapsed, 1e-6)
            remaining = (len(jobs) - finished) / max(rate, 1e-6)
            print(f"[progress] {finished}/{len(jobs)} eta_s={remaining:.0f}", flush=True)
    return path


class _null:
    def __enter__(self):
        return None

    def __exit__(self, *_args):
        return False


def main(argv=None):
    import argparse
    import json as _json
    parser = argparse.ArgumentParser(description="Free generation for the mirroring check")
    parser.add_argument("--variants", required=True)
    parser.add_argument("--scenarios", required=True)
    parser.add_argument("--config", default="Workstream_2/configs/qwen.json")
    parser.add_argument("--ws3-config", default="Workstream_3/configs/ws3.json")
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--limit", type=int, default=0, help="Generate only the first N jobs (smoke)")
    args = parser.parse_args(argv)
    settings = _json.loads(Path(args.ws3_config).read_text())["gen"]
    jobs = build_jobs(
        read_jsonl(args.variants), read_jsonl(args.scenarios),
        samples_per_variant=settings["samples_per_variant"],
        neutral_samples=settings["neutral_samples"])
    if args.limit:
        jobs = jobs[:args.limit]
    run_generation(
        jobs, args.out, args.config,
        batch_size=args.batch_size or settings["batch_size"],
        seed=settings["seed"],
        max_new_tokens=settings["max_new_tokens"],
        temperature=settings["temperature"],
        top_p=settings["top_p"])


if __name__ == "__main__":
    main()
