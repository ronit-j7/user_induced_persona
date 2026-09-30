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

## Implementation and verification status (2026-09-29)

| Item | Status | Evidence |
| --- | --- | --- |
| Version 1 WS1/WS2/WS3 handoff | Implemented | Root `INTERFACES.md`, validator, sample JSONL. |
| Qwen hook and exact token positions | Verified on tiny Qwen CPU and RTX 3090 | `uv run python -m pytest -q` (7 passed); GPU smoke generated indexed activation, localization, and comparison bundles. |
| Real Qwen2.5-7B tokenizer/config | Verified | `scripts/verify_qwen_tokenizer.py` checked 3 distinct sample rows and model dimensions. |
| Paired scores, controls, permutations, heatmaps | Implemented, synthetic integration run | Both assigned and user sample analyses complete. Sample has 3 scenarios and random weights, so its numerical scores are not findings. |
| Imported reference repositories | Verified | 72 SMH and 41 Persona Vectors files match their recorded upstream Git blobs; no nested Git metadata. |
| Upstream humorous reproduction | NOT_RUN | Requires actual upstream judged CSVs/vector files and full Qwen weights. |
| E/A Big Five assigned localization | NOT_RUN | Requires WS1 real twin data and full Qwen weights. |

The RTX 3090 itself works with bf16. At the last check, about 8.8 GB VRAM was
free and the filesystem had about 11 GB free; the full bf16 7B checkpoint is
not cached. The full-model gates must wait for enough free VRAM/storage and
the upstream/WS1 inputs. Keep sample and full-model results separate.
