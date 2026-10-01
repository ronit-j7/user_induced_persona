# Workstream 3

User-style experiments on Qwen2.5-7B-Instruct. WS2 owns extraction and head scoring.
This package owns free generation, the response-only judge, mirroring statistics,
the same-pole significance check, and the stretch regression and ablation.

Run everything through `bash Workstream_2/scripts/uv_hdd.sh` so packages and weights
stay on the hard disk. Large outputs go under `results/ws3/`, which is on that disk.

The judge needs `OPENAI_API_KEY` in the environment or in
`/media/gaurav/Data21/eshaan/secrets/.env`. It never sees the user message.
Coherence uses Izawa et al.'s rubric with the scenario's neutral intent in the
question slot. Generation, localization, and regression do not need a key.

```bash
bash Workstream_2/scripts/uv_hdd.sh run python -m ws3.generate \
  --variants Data_Creation/data/user_variants.jsonl \
  --scenarios Data_Creation/data/scenarios.jsonl \
  --out results/ws3/generations
bash Workstream_2/scripts/uv_hdd.sh run python -m ws3.judge \
  --generations results/ws3/generations/gen.jsonl \
  --out results/ws3/judged --cache results/ws3/judge_cache
bash Workstream_2/scripts/uv_hdd.sh run python -m ws3.mirroring \
  --scores results/ws3/judged/scores.jsonl --out results/ws3/mirroring
```

Generation uses every user paraphrase (each one is a different facet; see
`Data_Creation/DECISIONS.md` D3), two samples each, plus three neutral samples
per scenario. That is 540 generations.
