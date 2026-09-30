# Workstream 2 implementation plan

Scope: Qwen2.5-7B-Instruct reference implementation validation, Big Five assigned-persona
localization, reusable extraction/scoring, and neutral/random controls. The root
`INTERFACES.md` is the versioned handoff contract for all workstreams.

## Implementation sequence

1. Define version 1 JSONL and artifact contracts. Validate IDs, matched poles,
   QC, explicit system prompts, fixed responses, and neutral baselines. Keep
   illustrative fixtures separate from Workstream 1's real data.
2. Implement exact chat token spans and batch-size-one pre-o_proj hooks. Record
   first/user/response readouts, FP32 reductions, FP16 storage, and per-layer
   output projection weights. Save row indices, model/tokenizer revisions,
   dependency versions, input hashes, seed, and run status.
3. Implement paired contrasts, raw head contribution scores, both normalization
   conventions, score identity checks, scenario-clustered sign-flip tests,
   BH correction, layer SNR, heatmaps, and seeded control groups.
4. Execute unmodified original extraction/score functions independently on
   identical project inputs and report the tested sample and numerical errors.
   The original humorous experiment and judged-data replay are out of scope;
   the legacy CSV adapter is an optional utility, not a completion gate.
5. Exercise the whole pipeline on small sample twins and a randomly initialized
   tiny Qwen on CPU. Test GQA dimensions, hook removal, token boundaries,
   response sensitivity, prefix invariance, scoring and malformed data.
6. Document installation, CPU smoke run, GPU validation, real-data swap,
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

## Current completion criteria (2026-10-01)

- Independent original-reference function agreement on identical project inputs
  with full Qwen: captured-vector tolerances and head-score rho > 0.95; report
  the two tested rows, not a paper reproduction claim.
- Independent E/A assigned localization over all 784 heads, empirical published
  candidate ranks, heatmaps, significance, layer SNR and saved controls.
- Original humorous generation, judging and judged-data replay are OUT_OF_SCOPE
  under `Context_Documents/plan_team.md` and `plan_agent.md`.
- WS3 user localization/comparison and behavioral judging remain separate work.

## Implementation and verification status (2026-09-30)

| Item | Status | Evidence |
| --- | --- | --- |
| Version 1 WS1/WS2/WS3 handoff | Implemented | Root `INTERFACES.md`, validator, sample JSONL. |
| Current WS1 adapter | Verified on real inputs | `prepare-data` builds 420 rows from 20 scenarios and 20 system prompts; 100 matched assigned pairs per E/A trait. Canonical responses stay unchanged. |
| Current WS1 QC protocol | Verified | Accepts the exact `gen_user_variants.make_row` export, checks both forced-choice orders, and preserves authored-system provenance. Legacy numeric QC remains supported. |
| Qwen hooks and token positions | Verified on full Qwen on RTX 3090 | 20 tests pass without skips, including the selective-refresh guard. Full checkpoint passes response sensitivity and bit-identical first/user checks. |
| Real Qwen tokenizer/config | Verified on all 420 WS1 rows | Max full sequence 1,096 tokens, max response 1,022; no truncation. The first 150 response tokens are selected explicitly, with shorter responses using all tokens. |
| Independent upstream implementation agreement | Verified with full Qwen on RTX 3090 | Original upstream hooks match captured vectors exactly; score Spearman 1.0, maximum score error 1.34e-5. This is not humorous-head reproduction. |
| Scores, controls, permutations, heatmaps | Verified on full checkpoint | All 420 assigned stimuli complete E/A first/resp/user analyses with 19,999 permutations, 100 pairs and 20 independent scenarios per trait. Fixture tests cover 12 localizations and 6 comparisons. |
| uv environment/storage | Implemented | Default model/dev groups retain libraries across `uv run`; `Workstream_2/scripts/uv_hdd.sh` routes packages, cache and weights to the hard disk. Repo `results/` points there on this host. |
| Imported reference repositories | Verified | 72 SMH and 41 Persona Vectors files match recorded upstream Git blobs; no nested Git metadata. |
| Original humorous experiment/replay | OUT_OF_SCOPE | Excluded by the 2026-10-01 team scope; not a missing input or WS2 blocker. |
| E/A assigned localization | COMPLETE on full checkpoint | RTX 3090, bf16, pinned WS1 revision; peak GPU allocation 14.40 GiB. Six localization bundles and their controls/heatmaps are complete. |
| Real assigned/user comparison | WS3 | WS1 user variants are now generated. This A-only refresh does not run or rerun user experiments. |

The pinned Qwen checkpoint is fully cached at
`/media/gaurav/Data21/eshaan/models/huggingface`; the uv environment/cache and
large result bundles are also on that hard disk. The completed full run is at
`results/ws1-verification/full-qwen-gpu-retry/verification.json`. Every file in
its completed manifest was checked after the folder reorganization. Compact
measured evidence is checked in as `docs/full_qwen_verification_summary.json`.
An earlier memory-blocked attempt is retained separately as `full-qwen-gpu`.

The latest A prompts were refreshed on 2026-10-01: only 200 A system twins
were extracted again, while 220 E/neutral rows remain bit identical. The three
A localizations are at `results/ws2-refresh/A-system-prompts-2026-10-01-retry/`;
these supersede the September 30 A reports. E and reference validation are
retained from that earlier run. Current metrics, hashes and paths are in
`docs/assigned_refresh_2026-10-01.json`; all neutral controls are unchanged.

The full verification command is:

```bash
bash Workstream_2/scripts/uv_hdd.sh run python -m Workstream_2.scripts.verify_ws2_on_ws1 \
  --model full --device cuda:0 --out results/ws1-verification/full-qwen-gpu-next
```

For WS3, use the generated WS1 variants with `prepare-data --user-variants ...` and
`run --require-user ...` from `INTERFACES.md`. Regenerate incomplete QC cells
or explicitly choose a balanced slice; neither pairing nor QC is bypassed.
