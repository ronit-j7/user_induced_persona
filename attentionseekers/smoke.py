"""Tiny random Qwen CPU/GPU integration run. Results are always illustrative."""
import argparse
from argparse import Namespace
from pathlib import Path

from .cli import cmd_analyze, cmd_compare
from .config import ModelConfig
from .data import load_rows
from .extract import extract_dataset


class CharacterTokenizer:
    """ASCII fixture tokenizer with offsets; no checkpoint or network needed."""
    chat_template = "<|im_start|>{role}\n{content}<|im_end|>\n<|im_start|>assistant\n"

    def encode(self, text, add_special_tokens=False):
        if any(ord(ch) >= 256 for ch in text):
            raise ValueError("Smoke tokenizer supports Latin-1 fixtures only")
        return [ord(ch) for ch in text]

    def decode(self, ids, skip_special_tokens=False, clean_up_tokenization_spaces=False):
        return "".join(map(chr, ids))

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        rendered = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
        if add_generation_prompt:
            rendered += "<|im_start|>assistant\n"
        return self.encode(rendered) if tokenize else rendered

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        output = {"input_ids": self.encode(text)}
        if return_offsets_mapping:
            output["offset_mapping"] = [(i, i + 1) for i in range(len(text))]
        return output


def run_smoke(out, *, device="cpu"):
    import torch
    from transformers import Qwen2Config, Qwen2ForCausalLM
    root = Path(__file__).resolve().parents[1]
    dataset = root / "Data_Creation/samples/ws2_sample.jsonl"
    rows = [row for row in load_rows(dataset, allow_sample=True) if row["trait"] in ("E", "none")]
    config = ModelConfig(model_name="tiny-random-Qwen2-smoke", revision="untrained",
                         num_layers=2, num_heads=4, num_kv_heads=2, head_dim=8,
                         hidden_size=32, smh_layer=1, smh_heads=(0, 1, 2), dtype="float32",
                         device=device, attention_implementation="sdpa", max_length=512,
                         is_synthetic=True, seed=0)
    torch.manual_seed(0)
    model = Qwen2ForCausalLM(Qwen2Config(vocab_size=256, hidden_size=32, intermediate_size=64,
                                          num_hidden_layers=2, num_attention_heads=4,
                                          num_key_value_heads=2, max_position_embeddings=512,
                                          pad_token_id=0, eos_token_id=1)).eval().to(device)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    acts = extract_dataset(model, CharacterTokenizer(), config, rows, out / "activations",
                           allow_sample=True, data_path=dataset)
    for source in ("assigned", "user"):
        cmd_analyze(Namespace(acts=str(acts), out=str(out / source), trait="E", source=source,
                              readout="resp", permutations=31, seed=0))
    cmd_compare(Namespace(assigned=str(out / "assigned"), user=str(out / "user"),
                          out=str(out / "comparison")))
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cpu")
    arguments = parser.parse_args()
    print(run_smoke(arguments.out, device=arguments.device))
