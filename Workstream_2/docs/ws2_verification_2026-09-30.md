# WS2 verification: 2026-09-30

## Verified

- Current WS1 inputs: 20 scenarios, 20 authored E/A system prompts, complete
  canonical responses. Preparation produces 420 rows: 20 neutral and 400
  assigned twins, with 100 paired contrasts and 20 independent scenarios per
  trait. Source hashes are recorded in each prepared bundle.
- Regression suite: **19 passed, no skips**, executed through uv in the HDD
  verification environment. Coverage includes WS1's exact QC export, matched
  cells, token windows, causal readouts, hook cleanup, independent reference
  functions, future user comparisons, replay adapters, and manifest integrity.
- Real Qwen tokenizer: all 420 rows fit; longest sequence 1,096 tokens,
  longest response 1,022 tokens. Response extraction selects the first 150
  tokens, or the entire response when shorter, without changing stored text.
- RTX 3090 integration with **random tiny Qwen weights** completed all 420
  real input rows and six E/A assigned localizations using 19,999 permutations.
  First/user readouts were bit identical after changing the forced response.
  Independently executed original reference functions matched captured vectors
  exactly; score Spearman was 1.0. These synthetic results are software evidence.
- Full Qwen checkpoint revision `a09a35458c702b33eeacc393d103063234e8bc28`
  is cached, with all four weight shards on the HDD.
- **Full Qwen verification completed on the RTX 3090:** all 420 real WS1 rows,
  six E/A first/response/user localizations, 19,999 permutations, 100 matched
  pairs and 20 scenario clusters per trait. Peak GPU allocation was 14.40 GiB.
  First/user readouts were bit identical after changing the response. Original
  reference hooks matched vectors exactly; score Spearman was 1.0, maximum
  score error 1.34e-5. All completed artifact hashes were verified.
- After consolidation into `Workstream_2/`, all 19 tests pass without skips and
  the installed `ws2 prepare-data` command still produces the same 420 rows.
  Repeating E response localization through the moved installed CLI reproduces
  all seven saved arrays exactly, including permutation p values and BH q values.

## Full-model results and remaining gates

The first full checkpoint attempt failed while loading the first shard. At that point,
`harsha`'s Visual Grounding evaluation `eval_best_all_150k.py` (PID 1778350,
started at 20:59:46 local time) occupied about 22.2 GiB of GPU memory. The
remaining memory was insufficient. That attempt produced no localization.
The incomplete attempt is retained and has no completed verification manifest.

After the GPU was freed, `full-qwen-gpu-retry` completed successfully. Layer 19
response-readout ranks for heads 2, 4, and 27 were respectively 1, 3, and 2 for
both E and A. The number of heads with BH q < 0.05 across all 784 heads was
685 for E and 703 for A on this response readout. These are assigned-persona
results on the current 20 scenarios; they do not establish user-induced persona
effects or replace the separate humorous reproduction gate. Detailed metrics
and immutable manifest hashes are in `full_qwen_verification_summary.json`.

WS1 user rewrites are **NOT_AVAILABLE**; the future comparison path passes
fixture tests, but a real assigned/user comparison requires those exports.
Humorous paper-head reproduction is independently **NOT_RUN** because its
judged CSVs and saved upstream vectors are absent. Reference hook agreement on
WS1 inputs does not establish humorous-head recovery.

## Runtime and storage

The successful tests used torch 2.4.1+cu121, transformers 4.56.1, accelerate
1.10.1, numpy 2.1.2, and matplotlib 3.10.0. The temporary verification
environment on the HDD inherits the existing system torch installation; a
separate fully isolated locked uv environment is still being installed there.

The mounted HDD is `/dev/sdb1`, at `/media/gaurav/Data21`. New large files use:

| Artifact | Absolute location |
| --- | --- |
| Checkpoint | `/media/gaurav/Data21/eshaan/models/huggingface` |
| uv cache | `/media/gaurav/Data21/eshaan/cache/uv` |
| Locked environment | `/media/gaurav/Data21/eshaan/envs/user_induced_persona` |
| Verification environment | `/media/gaurav/Data21/eshaan/envs/ws2-verification-bootstrap` |
| Results | `/media/gaurav/Data21/eshaan/results/user_induced_persona` |

Repository `.venv` and `results` point to HDD locations. The completed full GPU
report is `results/ws1-verification/full-qwen-gpu-retry/verification.json`.
The earlier tiny report is `results/ws1-verification/tiny-real-tokenizer-gpu/verification.json`.
The incomplete full attempt is `results/ws1-verification/full-qwen-gpu`.
Use `bash Workstream_2/scripts/uv_hdd.sh ...` for subsequent installation and runs.
