"""Check the real Qwen chat template and token spans without downloading weights."""
from pathlib import Path

from transformers import AutoConfig, AutoTokenizer

from attentionseekers.chat import describe_encoding, encode
from attentionseekers.config import load_config
from attentionseekers.data import load_rows

ROOT = Path(__file__).resolve().parents[1]
config = load_config(ROOT / "configs/qwen.json")
tokenizer = AutoTokenizer.from_pretrained(config.model_name, revision=config.revision,
                                           use_fast=True)
model_config = AutoConfig.from_pretrained(config.model_name, revision=config.revision)
for field, expected in (("num_hidden_layers", config.num_layers),
                        ("num_attention_heads", config.num_heads),
                        ("num_key_value_heads", config.num_kv_heads),
                        ("hidden_size", config.hidden_size)):
    observed = getattr(model_config, field)
    if observed != expected:
        raise ValueError(f"{field}: configured {expected}, upstream {observed}")
rows = load_rows(ROOT / "samples/ws2_sample.jsonl", allow_sample=True)
for row in (rows[0], next(r for r in rows if r["set"] == "sys_twin"),
            next(r for r in rows if r["set"] == "user_variant")):
    detail = describe_encoding(tokenizer, encode(tokenizer, row, config.max_length))
    print(row["id"], detail)
    if detail["first_token"] != "\n":
        raise ValueError("Qwen generation-prefix last token is not a newline")
print("Qwen2.5-7B tokenizer/config boundary check passed")
