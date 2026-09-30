"""Generate the forced responses r_b with Qwen2.5-7B-Instruct. Run this on the GPU machine.

One greedy generation per scenario from (neutral system prompt, u0), max 150 new tokens. If the reply was cut off
by the token limit, it is truncated at the last sentence or line boundary. A code block left open by the cut is
closed with ```. The same r_b is later reused for every user-variant and sys-twin row of that scenario.

Usage (on the cluster):
    python gen_forced_responses.py --in scenarios.jsonl --out scenarios_with_rb.jsonl
Writes:
    scenarios_with_rb.jsonl       scenarios with forced_response filled in
    forced_responses_raw.jsonl    raw generation, token counts, truncation flag, library versions
"""

import argparse
import json
import re
from pathlib import Path

MODEL = "Qwen/Qwen2.5-7B-Instruct"
NEUTRAL_SYSTEM = "You are a helpful assistant."
MAX_NEW_TOKENS = 150

FENCE = re.compile(r"^\s*```", re.M)
SENTENCE_END = re.compile(r"(?<!\d)[.!?][\"')\]*_]*(?=\s)")


def truncate(text, finished):
    """Cut an unfinished reply at its last sentence or line boundary. Returns (text, was_truncated)."""
    if finished:
        return text.strip(), False
    # Code-fence state at each position: inside a block, only line ends count as boundaries.
    fences = [m.start() for m in FENCE.finditer(text)]

    def in_code(pos):
        return sum(1 for f in fences if f < pos) % 2 == 1

    cuts = [m.end() for m in SENTENCE_END.finditer(text) if not in_code(m.end())]
    cuts += [m.start() for m in re.finditer(r"\n", text)]  # end of every complete line
    cuts = [c for c in cuts if text[:c].strip()]
    if not cuts:
        return text.strip(), True
    cut = max(cuts)
    out = text[:cut].rstrip()
    if in_code(cut):
        last_fence = max(f for f in fences if f < cut)
        if not text[last_fence:cut].strip().split("\n")[1:]:   # block opened but empty: drop it
            out = text[:last_fence].rstrip()
        else:
            out += "\n```"
    return out, True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="scenarios.jsonl")
    ap.add_argument("--out", default="scenarios_with_rb.jsonl")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    args = ap.parse_args()

    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="cuda:0")
    model.eval()
    # Explicit config: Qwen's shipped generation_config turns on sampling and repetition_penalty=1.05.
    eos_ids = model.generation_config.eos_token_id
    eos_ids = eos_ids if isinstance(eos_ids, list) else [eos_ids]
    gen_cfg = GenerationConfig(max_new_tokens=args.max_new_tokens, do_sample=False, repetition_penalty=1.0,
                               eos_token_id=eos_ids, pad_token_id=tok.pad_token_id or eos_ids[0])

    scenarios = [json.loads(l) for l in open(args.inp) if l.strip()]
    rows, raw_rows = [], []
    for scn in scenarios:
        msgs = [{"role": "system", "content": NEUTRAL_SYSTEM}, {"role": "user", "content": scn["user"]}]
        enc = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                      return_dict=True, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, generation_config=gen_cfg)
        new_ids = out[0, enc["input_ids"].shape[1]:].tolist()
        finished = bool(new_ids) and new_ids[-1] in eos_ids
        raw = tok.decode(new_ids, skip_special_tokens=True)
        rb, was_truncated = truncate(raw, finished)
        rb_tokens = len(tok(rb, add_special_tokens=False)["input_ids"])
        rows.append(dict(scn, forced_response=rb))
        raw_rows.append({"scenario": scn["scenario"], "raw": raw, "new_tokens": len(new_ids),
                         "finished": finished, "truncated": was_truncated, "rb_tokens": rb_tokens})
        print(f"{scn['scenario']:14s} new={len(new_ids):3d} finished={finished!s:5s} rb_tokens={rb_tokens}")

    meta = {"model": args.model, "revision": getattr(model.config, "_commit_hash", None),
            "system": NEUTRAL_SYSTEM, "max_new_tokens": args.max_new_tokens, "decoding": "greedy",
            "repetition_penalty": 1.0, "dtype": "bfloat16", "torch": torch.__version__,
            "transformers": transformers.__version__, "gpu": torch.cuda.get_device_name(0)}
    out_path = Path(args.out)
    with open(out_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(out_path.with_name("forced_responses_raw.jsonl"), "w") as f:
        f.write(json.dumps({"meta": meta}, ensure_ascii=False) + "\n")
        for r in raw_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {out_path} and {out_path.with_name('forced_responses_raw.jsonl')}")


if __name__ == "__main__":
    main()
