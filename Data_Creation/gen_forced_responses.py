"""Generate the forced responses r_b with Qwen2.5-7B-Instruct. Run this on the GPU machine.

One greedy generation per scenario from (neutral system prompt, u0), max 150 new tokens. If the reply was cut off
by the token limit, it is truncated at the last sentence or line boundary. A code block left open by the cut is
closed with ```. The same r_b is later reused for every user-variant and sys-twin row of that scenario.

Device: --device auto uses the GPU with the most free memory if it has >= 18 GiB free, otherwise the CPU
(fp32, --threads cores; ~20-40 min for all 20 scenarios). bf16 is used on GPU.

Usage (on the cluster):
    python gen_forced_responses.py --in scenarios.jsonl --out scenarios_with_rb.jsonl
Writes:
    scenarios_with_rb.jsonl       scenarios with forced_response filled in
    forced_responses_raw.jsonl    raw generation, token counts, truncation flag, library versions
"""

import argparse
import json
import re
import time
from pathlib import Path

MODEL = "Qwen/Qwen2.5-7B-Instruct"
NEUTRAL_SYSTEM = "You are a helpful assistant."
MAX_NEW_TOKENS = 150
MIN_FREE_GIB = 18

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
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda:N")
    ap.add_argument("--threads", type=int, default=16, help="CPU threads (shared machine: stay polite)")
    args = ap.parse_args()

    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

    device = args.device
    if device == "auto":
        device = "cpu"
        if torch.cuda.is_available():
            free = {i: torch.cuda.mem_get_info(i)[0] / 2**30 for i in range(torch.cuda.device_count())}
            print("Free GPU memory (GiB):", {i: round(f, 1) for i, f in free.items()})
            best = max(free, key=free.get)
            if free[best] >= MIN_FREE_GIB:
                device = f"cuda:{best}"
    dtype = torch.float32 if device == "cpu" else torch.bfloat16
    if device == "cpu":
        torch.set_num_threads(args.threads)
    print(f"Using device={device} dtype={dtype}")

    tok = AutoTokenizer.from_pretrained(args.model)
    # transformers < 4.56 calls this argument torch_dtype; an unknown name would silently load fp32.
    tf_version = tuple(int(x) for x in re.findall(r"\d+", transformers.__version__)[:2])
    load_kw = {"dtype": dtype} if tf_version >= (4, 56) else {"torch_dtype": dtype}
    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kw).to(device)
    print(f"Loaded {args.model} with transformers {transformers.__version__}, torch {torch.__version__}, "
          f"param dtype {next(model.parameters()).dtype}")
    model.eval()
    # Explicit config: Qwen's shipped generation_config turns on sampling and repetition_penalty=1.05.
    eos_ids = model.generation_config.eos_token_id
    eos_ids = eos_ids if isinstance(eos_ids, list) else [eos_ids]
    gen_cfg = GenerationConfig(max_new_tokens=args.max_new_tokens, do_sample=False, repetition_penalty=1.0,
                               eos_token_id=eos_ids, pad_token_id=tok.pad_token_id or eos_ids[0])

    scenarios = [json.loads(l) for l in open(args.inp) if l.strip()]
    rows, raw_rows = [], []
    for scn in scenarios:
        t0 = time.time()
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
        print(f"{scn['scenario']:14s} new={len(new_ids):3d} finished={finished!s:5s} rb_tokens={rb_tokens:3d} "
              f"({time.time() - t0:.0f}s)", flush=True)

    meta = {"model": args.model, "revision": getattr(model.config, "_commit_hash", None),
            "system": NEUTRAL_SYSTEM, "max_new_tokens": args.max_new_tokens, "decoding": "greedy",
            "repetition_penalty": 1.0, "device": device, "dtype": str(dtype), "torch": torch.__version__,
            "transformers": transformers.__version__,
            "gpu": torch.cuda.get_device_name(device) if device != "cpu" else None}
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
