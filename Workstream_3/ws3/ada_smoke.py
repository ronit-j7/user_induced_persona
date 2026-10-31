"""Pre-flight on the target GPUs: sharded load, finite activations, lambda=0 identity, speed.

Prints a JSON verdict. Takes about ten minutes including the model load.
"""
import json
from pathlib import Path
import time

import numpy as np

from .generate import generate_batch, projection_hook, read_jsonl
from .project import build_projection_jobs
from .residual import layer_output_hook


def main(argv=None):
    import argparse
    import torch
    from attentionseekers.chat import encode
    from attentionseekers.config import load_config
    from attentionseekers.extract import capture_encoded, load_model
    parser = argparse.ArgumentParser(description="Ada pre-flight")
    parser.add_argument("--config", default="Workstream_2/configs/qwen_ada.json")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=300)
    parser.add_argument("--out", default=None, help="Optional path for the JSON verdict")
    args = parser.parse_args(argv)

    report = {"gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
              "torch": torch.__version__, "cuda": torch.version.cuda}
    print(json.dumps(report), flush=True)
    config = load_config(args.config)
    started = time.time()
    model, tokenizer = load_model(config)
    report["load_seconds"] = round(time.time() - started, 1)
    report["device_map_devices"] = sorted({str(v) for v in getattr(model, "hf_device_map", {}).values()})
    report["memory_gib"] = [round(torch.cuda.memory_allocated(i) / 2**30, 2) for i in range(torch.cuda.device_count())]

    rows = [r for r in read_jsonl("Data_Creation/data/user_variants.jsonl") if r["trait"] == "E"][:2]
    rows += [r for r in read_jsonl("Data_Creation/data/neutral.jsonl")][:2]
    finite = True
    for row in rows:
        values, _ = capture_encoded(model, config, encode(tokenizer, row, config.max_length,
                                                         response_tokens=config.response_tokens), ("resp",))
        finite &= bool(np.isfinite(values["resp"]).all())
    report["extraction_finite"] = finite

    jobs = build_projection_jobs(read_jsonl("Data_Creation/data/user_variants.jsonl"),
                                 read_jsonl("Data_Creation/data/factorial.jsonl"),
                                 read_jsonl("Data_Creation/data/scenarios.jsonl"))[:args.batch_size]
    gen = dict(max_new_tokens=args.max_new_tokens, temperature=0.0, top_p=1.0)

    def run(hook=None):
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        t0 = time.time()
        if hook is None:
            records = generate_batch(model, tokenizer, jobs, **gen)
        else:
            with hook:
                records = generate_batch(model, tokenizer, jobs, **gen)
        return records, time.time() - t0

    rng = np.random.default_rng(0)
    u = rng.standard_normal(3 * 128)
    u = (u / np.linalg.norm(u)).astype(np.float32)
    groups = {config.smh_layer: (list(config.smh_heads), u, np.zeros_like(u))}
    resid_u = rng.standard_normal(config.hidden_size)
    resid_u = (resid_u / np.linalg.norm(resid_u)).astype(np.float32)
    base, seconds = run()
    again, _ = run()
    same, _ = run(projection_hook(model, groups, 0.0))
    moved, _ = run(projection_hook(model, groups, 8.0))
    resid_same, _ = run(layer_output_hook(model, config.smh_layer, resid_u, np.zeros_like(resid_u), 0.0))
    resid_moved, _ = run(layer_output_hook(model, config.smh_layer, resid_u, np.zeros_like(resid_u), 1.0))
    report["seconds_per_generation"] = round(seconds / len(jobs), 2)
    report["mean_new_tokens"] = float(np.mean([r["n_tokens"] for r in base]))
    report["greedy_repeatable"] = sum(a["text"] == b["text"] for a, b in zip(base, again))
    report["lambda0_identical"] = sum(a["text"] == b["text"] for a, b in zip(base, same))
    report["lambda8_changed"] = sum(a["text"] != b["text"] for a, b in zip(base, moved))
    report["resid_lambda0_identical"] = sum(a["text"] == b["text"] for a, b in zip(base, resid_same))
    report["resid_lambda1_changed"] = sum(a["text"] != b["text"] for a, b in zip(base, resid_moved))
    report["n"] = len(jobs)
    report["empty_outputs"] = sum(not r["text"].strip() for r in base)
    report["peak_memory_gib"] = [round(torch.cuda.max_memory_allocated(i) / 2**30, 2)
                                 for i in range(torch.cuda.device_count())]
    report["sample"] = base[0]["text"][:300]
    # Largest job: 6 arms x 200 generations.
    report["estimated_hours_per_job_one_worker"] = round(report["seconds_per_generation"] * 1200 / 3600, 2)
    ok = (finite and report["empty_outputs"] == 0 and report["lambda8_changed"] > 0
          and report["greedy_repeatable"] == len(jobs) and report["lambda0_identical"] == len(jobs)
          and report["resid_lambda0_identical"] == len(jobs))
    report["verdict"] = "OK" if ok else "CHECK"
    text = json.dumps(report, indent=2)
    print(text, flush=True)
    if args.out:
        Path(args.out).write_text(text + "\n")


if __name__ == "__main__":
    main()
