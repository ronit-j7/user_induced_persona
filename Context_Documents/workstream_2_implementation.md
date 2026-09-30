# Workstream 2 implementation plan

Scope: Qwen2.5-7B-Instruct SMH reproduction, Big Five assigned-persona
localization, reusable extraction/scoring, and neutral/random controls. The root
`INTERFACES.md` is the versioned handoff contract for all workstreams.

## Implementation sequence

1. Define version 1 JSONL and artifact contracts. Validate IDs, matched poles,
   QC, explicit system prompts, fixed responses, and neutral baselines. Keep
   illustrative fixtures separate from Workstream 1's unfinished data.
2. Implement exact chat token spans and batch-size-one pre-o_proj hooks. Record
   first/user/response readouts, FP32 reductions, FP16 storage, and per-layer
   output projection weights. Save row indices, model/tokenizer revisions,
   dependency versions, input hashes, seed, and run status.
3. Implement paired contrasts, raw head contribution scores, both normalization
   conventions, score identity checks, scenario-clustered sign-flip tests,
   BH correction, layer SNR, heatmaps, and seeded control groups.
4. Add an upstream CSV adapter and an isolated compatibility tokenizer for
   exact replay. Compare independently extracted scores with upstream arrays;
   report paper-head recovery and extraction agreement as separate gates.
5. Exercise the whole pipeline on small sample twins and a randomly initialized
   tiny Qwen on CPU. Test GQA dimensions, hook removal, token boundaries,
   response sensitivity, prefix invariance, scoring and malformed data.
6. Document installation, CPU smoke run, GPU reproduction, real-data swap,
   Python APIs, file schemas, and remaining empirical checks.

## Method decisions

- Real runs target only the configured Qwen model. The CPU demo is explicitly
  synthetic and never establishes replication or Big Five findings.
- First = final generation-prefix token (the position predicting the first
  assistant token). Extract first/user from a prefix-only forward pass. This
  also prevents sequence-length-dependent floating-point kernels from making
  this readout vary with a teacher-forced response.
- Response tokens are tokenized separately and appended as IDs in project mode.
  Upstream replay instead preserves its joint prompt+answer tokenization and
  separately counted prompt length. Never use replay mode for the twin study.
- Average paraphrase differences within each scenario for inference; randomize
  signs at scenario level. Paraphrases are repeated observations of scenarios,
  not independent scenario replications. Save pairs and cluster counts. The
  default is 19,999 permutations because 1,000 has a minimum Monte Carlo p of
  0.001, too coarse for BH correction over Qwen's 784 heads.
- Use all neutral rows' first readout to select the highest-mean-norm triple.
  Random triples sample query heads without replacement within each triple;
  overlap with SMHs is allowed and reported.
- Upstream commit `532da0151b319efa99145cdad88035c889d72ef3` was inspected.
  Its actual `normalize_matrix` uses raw-score z-scores despite the signed-log
  description. Save raw, raw-z, and signed-log-z arrays; replay uses raw-z.
  Its contribution target is an independently captured attention-output vector;
  save that target for replay comparisons instead of silently replacing it.

## Empirical gates still requiring a GPU and real data

- Upstream humorous reproduction: layer 19 top-3 = heads {2, 4, 27}.
- Own extraction on the same filtered CSV rows: same top-3 as upstream and
  Spearman rho > 0.95. Never label a score-array self-comparison reproduction.
- Real E/A twin localization, heatmaps, SMH ranks, neutral controls, and QC stats.
- Record failures as measured failures; absent runs remain NOT_RUN.

## Implementation and verification status (2026-09-30)

| Item | Status | Evidence |
| --- | --- | --- |
| Version 1 WS1/WS2/WS3 handoff | Implemented | Root `INTERFACES.md`, validator, sample JSONL. |
| Current WS1 adapter | Verified on real inputs | `prepare-data` builds 420 rows from 20 scenarios and 20 system prompts; 100 matched assigned pairs per E/A trait. Canonical responses stay unchanged. |
| Current WS1 QC protocol | Verified | Accepts the exact `gen_user_variants.make_row` export, checks both forced-choice orders, and preserves authored-system provenance. Legacy numeric QC remains supported. |
| Qwen hooks and token positions | Verified on tiny Qwen CPU and RTX 3090 | 19 tests pass without skips, including response sensitivity, causal windows, exception cleanup, and all E/A/readout comparison branches. |
| Real Qwen tokenizer/config | Verified on all 420 WS1 rows | Max full sequence 1,096 tokens, max response 1,022; no truncation. The first 150 response tokens are selected explicitly, with shorter responses using all tokens. |
| Independent upstream implementation agreement | Verified with tiny Qwen on RTX 3090 | Original upstream hook functions execute independently: captured-vector max error 0, score Spearman 1.0. This is not humorous-head reproduction. |
| Scores, controls, permutations, heatmaps | Implemented and integrated | All 420 assigned stimuli run with random tiny weights; E/A first/resp/user analyses complete with 19,999 permutations. Fixture tests cover 12 localizations and 6 comparisons. Synthetic scores are not findings. |
| uv environment/storage | Implemented | Default model/dev groups retain libraries across `uv run`; `scripts/uv_hdd.sh` routes packages, cache and weights to the hard disk. Repo `results/` points there on this host. |
| Imported reference repositories | Verified | 72 SMH and 41 Persona Vectors files match recorded upstream Git blobs; no nested Git metadata. |
| Upstream humorous reproduction | NOT_RUN | Judged humorous CSVs/vectors and judge credentials are absent. This remains independent of WS1 user-variant data. |
| E/A assigned localization | NOT_RUN on full checkpoint | All four pinned checkpoint shards are cached on the HDD. Full loading failed because another Visual Grounding process occupied 22.2 GiB of GPU memory. |
| Real assigned/user comparison | NOT_AVAILABLE | WS1 has not produced `data/user_variants.jsonl`; `--require-user` fails before loading weights. |

The pinned Qwen checkpoint is fully cached at
`/media/gaurav/Data21/eshaan/models/huggingface`; the uv environment/cache and
large result bundles are also on that hard disk. The latest full-model attempt
failed at checkpoint loading: `harsha`'s Visual Grounding evaluation, PID
1778350, was occupying about 22.2 GiB on the RTX 3090. No full-model extraction
or scientific result was produced. The incomplete `full-qwen-gpu` output is
preserved; use a new output name for the next attempt.
The completed tiny-model integration report is at
`results/ws1-verification/tiny-real-tokenizer-gpu/verification.json`.

The full verification command is:

```bash
bash scripts/uv_hdd.sh run python -m scripts.verify_ws2_on_ws1 \
  --model full --device cuda:0 --out results/ws1-verification/full-qwen-gpu-retry
```

After WS1 ships user variants, rerun `prepare-data --user-variants ...` and
`run --require-user ...` from `INTERFACES.md`. Regenerate incomplete QC cells
or explicitly choose a balanced slice; neither pairing nor QC is bypassed.
