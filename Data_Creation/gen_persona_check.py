"""System-prompt check, step 1 (GPU machine): does Qwen actually adopt each persona system prompt?

For a few scenarios (one per domain), generate Qwen's greedy reply to u0 under each of the 20 persona system
prompts plus the neutral one, with the same token cap for all, so persona and neutral replies are comparable.
Step 2 (judge_persona_check.py, run locally) compares each persona reply with the neutral one.

Usage (on the cluster, in the folder with scenarios.jsonl and sys_prompts.json):
    python gen_persona_check.py --model <path to Qwen2.5-7B-Instruct>
Writes persona_check_gens.jsonl.
"""

import argparse
import json
import re
import time

NEUTRAL_SYSTEM = "You are a helpful assistant."
MIN_FREE_GIB = 18


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--scenarios", default="code_01,explain_03,emo_03,brainstorm_05")
    ap.add_argument("--scenarios-file", default="scenarios.jsonl")
    ap.add_argument("--sys-prompts", default="sys_prompts.json")
    ap.add_argument("--out", default="persona_check_gens.jsonl")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=12)
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda:N")
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
    print(f"Using device={device} dtype={dtype}")

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"  # batched generation: pad on the left so every prompt ends at the same position
    tf_version = tuple(int(x) for x in re.findall(r"\d+", transformers.__version__)[:2])
    load_kw = {"dtype": dtype} if tf_version >= (4, 56) else {"torch_dtype": dtype}
    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kw).to(device).eval()
    eos_ids = model.generation_config.eos_token_id
    eos_ids = eos_ids if isinstance(eos_ids, list) else [eos_ids]
    gen_cfg = GenerationConfig(max_new_tokens=args.max_new_tokens, do_sample=False, repetition_penalty=1.0,
                               eos_token_id=eos_ids, pad_token_id=tok.pad_token_id or eos_ids[0])

    wanted = args.scenarios.split(",")
    scenarios = [s for s in map(json.loads, open(args.scenarios_file)) if s["scenario"] in wanted]
    spec = json.load(open(args.sys_prompts))
    assert spec["neutral"] == NEUTRAL_SYSTEM
    systems = [{"trait": None, "pole": "0", "sys_paraphrase": 0, "system": NEUTRAL_SYSTEM}] + spec["prompts"]
    jobs = [(s, p) for s in scenarios for p in systems]
    print(f"{len(jobs)} generations ({len(scenarios)} scenarios x {len(systems)} system prompts)")

    rows = []
    for i in range(0, len(jobs), args.batch_size):
        t0 = time.time()
        batch = jobs[i:i + args.batch_size]
        texts = [tok.apply_chat_template([{"role": "system", "content": p["system"]},
                                          {"role": "user", "content": s["user"]}],
                                         tokenize=False, add_generation_prompt=True) for s, p in batch]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        with torch.no_grad():
            out = model.generate(**enc, generation_config=gen_cfg)
        for (s, p), ids in zip(batch, out[:, enc["input_ids"].shape[1]:].tolist()):
            end = next((j for j, t in enumerate(ids) if t in eos_ids), None)
            gen = ids if end is None else ids[:end]
            rows.append({"scenario": s["scenario"], "domain": s["domain"], "user": s["user"],
                         "trait": p["trait"], "pole": p["pole"], "sys_paraphrase": p["sys_paraphrase"],
                         "system": p["system"], "reply": tok.decode(gen, skip_special_tokens=True).strip(),
                         "new_tokens": len(gen), "finished": end is not None})
        print(f"batch {i // args.batch_size + 1}/{-(-len(jobs) // args.batch_size)} done ({time.time() - t0:.0f}s)",
              flush=True)

    meta = {"model": args.model, "max_new_tokens": args.max_new_tokens, "decoding": "greedy",
            "repetition_penalty": 1.0, "device": device, "dtype": str(dtype), "batch_size": args.batch_size,
            "torch": torch.__version__, "transformers": transformers.__version__}
    with open(args.out, "w") as f:
        f.write(json.dumps({"meta": meta}) + "\n")
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} generations to {args.out}")


if __name__ == "__main__":
    main()
