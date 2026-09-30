"""Exercise the current real WS1 assigned dataset, including upstream hook agreement.

bash Workstream_2/scripts/uv_hdd.sh run python -m Workstream_2.scripts.verify_ws2_on_ws1 --model full --device cuda:0 --out results/ws1-full
Use --model tiny for software validation with random weights and the real tokenizer.
All outputs record whether the model is synthetic. This script does not create
or judge user-style variants, or assert humorous paper-head reproduction.
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path

from attentionseekers.chat import encode
from attentionseekers.config import load_config
from attentionseekers.data import load_rows, matched_pairs
from attentionseekers.extract import capture_encoded, load_model
from attentionseekers.io import finish_run, new_run, write_json
from attentionseekers.pipeline import run_pipeline
from attentionseekers.preflight import check_tokens
from attentionseekers.prepare import prepare_dataset
from attentionseekers.paths import DEFAULT_CONFIG, REPO_ROOT
from attentionseekers.reference_check import verify_reference_extraction


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--model", choices=("tiny", "full"), default="tiny")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--permutations", type=int, default=19999)
    args = parser.parse_args()
    import numpy as np
    import torch
    from transformers import AutoConfig, AutoTokenizer, Qwen2Config, Qwen2ForCausalLM
    torch.set_num_threads(2)
    root = REPO_ROOT
    out = new_run(args.out)
    prepared = prepare_dataset(root / "Data_Creation/data/scenarios.jsonl",
                               root / "Data_Creation/prompts/sys_prompts.json", out / "prepared")
    data_path = prepared / "rows.jsonl"
    rows = load_rows(data_path)
    config = replace(load_config(args.config), device=args.device)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, revision=config.revision, use_fast=True)
    token_bundle = check_tokens(tokenizer, config, rows, out / "token-preflight", data_path=data_path)
    if args.model == "full":
        model, tokenizer = load_model(config)
    else:
        original = AutoConfig.from_pretrained(config.model_name, revision=config.revision)
        config = replace(config, model_name="tiny-random-Qwen2-real-WS1-test", is_synthetic=True,
                         num_layers=2, num_heads=4, num_kv_heads=2, head_dim=8, hidden_size=32,
                         smh_layer=1, smh_heads=(0, 1, 2), dtype="float32")
        torch.manual_seed(config.seed)
        model = Qwen2ForCausalLM(Qwen2Config(
            vocab_size=original.vocab_size, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=config.max_length,
            pad_token_id=original.pad_token_id, eos_token_id=original.eos_token_id,
            attn_implementation="sdpa")).eval().to(config.device)
    neutral = next(r for r in rows if r["set"] == "neutral")
    first = encode(tokenizer, neutral, config.max_length, response_tokens=config.response_tokens)
    changed = encode(tokenizer, {**neutral, "forced_response": "An alternative response used to test response sensitivity."},
                     config.max_length, response_tokens=config.response_tokens)
    a, _ = capture_encoded(model, config, first)
    b, _ = capture_encoded(model, config, changed)
    assert np.array_equal(a["first"], b["first"]), "First readout depends on the response"
    assert np.array_equal(a["user"], b["user"]), "User readout depends on the response"
    assert not np.array_equal(a["resp"], b["resp"]), "Response readout is insensitive to the response"
    positive, negative, _ = matched_pairs(rows, "E", "assigned")[0]
    agreement = verify_reference_extraction(model, tokenizer, config, [rows[positive], rows[negative]])
    write_json(out / "reference-agreement.json", agreement)
    pipeline = run_pipeline(model, tokenizer, config, rows, out / "pipeline", data_path=data_path,
                            permutations=args.permutations)
    report = {"is_synthetic": config.is_synthetic, "model_name": config.model_name,
              "device": config.device, "rows": len(rows), "response_tokens": config.response_tokens,
              "token_preflight": json.loads((token_bundle / "summary.json").read_text()),
              "first_bit_identical_after_response_change": True,
              "user_bit_identical_after_response_change": True, "response_sensitive": True,
              "upstream_implementation_agreement": agreement,
              "pipeline": json.loads((pipeline / "summary.json").read_text()),
              "gpu": torch.cuda.get_device_name(model.device) if model.device.type == "cuda" else None,
              "torch_runtime": torch.__version__, "torch_cuda": torch.version.cuda,
              "gpu_compute_capability": torch.cuda.get_device_capability(model.device)
                  if model.device.type == "cuda" else None,
              "max_gpu_allocated_bytes": torch.cuda.max_memory_allocated(model.device)
                  if model.device.type == "cuda" else None,
              "paper_humorous_reproduction": "NOT_RUN"}
    write_json(out / "verification.json", report)
    finish_run(out, {"kind": "ws1_integration_verification", "is_sample": False,
                     "is_synthetic": config.is_synthetic})
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
