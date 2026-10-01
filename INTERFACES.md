# AttentionSeekers shared interface (version 1)

This is the handoff contract for Data Creation (WS1), SMH localization (WS2),
and user-induced experiments (WS3). Change the schema version and this file
together when a breaking change is necessary. Code entry points live in the
`Workstream_2/attentionseekers` package; the CLI is `ws2` or `python -m attentionseekers`.

All WS2 support files are under `Workstream_2/` as described in its
`README.md`. The shared project `pyproject.toml` and `uv.lock` stay at the
repository root; package discovery points into `Workstream_2`. Python imports
remain `attentionseekers`. CLI defaults resolve WS1 inputs and the WS2 config
relative to the installed source, independently of the working directory.

## Environment and first run

From the repository root, use `uv` for dependency management:

```bash
uv sync --locked
uv run python -m Workstream_2.scripts.make_sample_ws2
uv run ws2 validate-data --data Workstream_2/samples/ws2_sample.jsonl --allow-sample
uv run python -m pytest -q
uv run python -m attentionseekers.smoke --out results/smoke-cpu
```

The `model` and `dev` dependency groups are enabled by default so `uv run`
retains the model/test libraries. The legacy `--extra model --extra test`
install command still works. Always execute tests with `python -m pytest`.

On this 3090 host, keep packages, model weights, and run artifacts on the
mounted hard disk:

```bash
bash Workstream_2/scripts/uv_hdd.sh sync --locked
bash Workstream_2/scripts/uv_hdd.sh run python -m pytest -q
```

The wrapper routes the uv cache to `/media/gaurav/Data21/eshaan/cache/uv`, the
environment to `envs/user_induced_persona`, and the HF cache to
`models/huggingface` under that same root. Set `WS2_STORAGE_ROOT` to your own
mounted data directory on another machine. On this host, the ignored repo
`results/` directory is a symlink to
`/media/gaurav/Data21/eshaan/results/user_induced_persona`.

`results/smoke-cpu` uses random tiny Qwen weights. It is a software smoke test,
never evidence of an SMH or a Big Five result. It creates activation,
localization, and comparison bundles. On a CUDA machine, add `--device cuda:0`
and use a different output directory. Every `--out` directory must be new;
the CLI refuses to overwrite previous runs.

The full Qwen config is `Workstream_2/configs/qwen.json`: GPU 0, bf16, response readout over
the first 150 response tokens, revision `a09a35458c702b33eeacc393d103063234e8bc28`
(the same checkpoint that WS1 used). Set `response_tokens` to null for the
full-response mean. Every bundle records the window and `is_synthetic`;
random-model integration results must not be presented as research findings.

## WS1 -> WS2: JSONL version 1

One UTF-8 JSON object per line. The checked-in sample at
`Workstream_2/samples/ws2_sample.jsonl` is executable schema documentation.
It is tagged `"is_sample": true`; real rows should omit this field or set it
to false, and must include passing QC. Never merge sample and real rows.

| Field | Type / rule |
| --- | --- |
| `id` | Unique, stable, nonempty string. Suggested `scenario|set|trait|pole|k`. |
| `scenario`, `domain` | Nonempty strings. Domain must be constant per scenario. |
| `set` | `neutral`, `sys_twin`, `user_variant`, or `factorial`. |
| `trait` | `E` or `A` for current experiments; `none` on neutral rows. |
| `sys_pole`, `user_pole` | One of `+`, `-`, `0`. `sys_twin` is sys +/- with user 0; `user_variant` is user +/- with sys 0; `neutral` is 0/0. |
| `sys_paraphrase`, `user_paraphrase` | Nonnegative integer. Set the unused side to 0. Use the same paraphrase numbers in both pole cells so they form explicit pairs. |
| `system`, `user` | Exact full message content, nonempty. Always supply a system message. Neutral system is `You are a helpful assistant.` |
| `forced_response` | Nonempty fixed assistant response for `resp` extraction. Identical for all rows of a scenario. `null` is allowed only if extracting `first`/`user` without `resp`. |
| `qc` | User rewrites: WS1's `{"passed":true,"forced_choice":{"rewrite_as_B":"B","rewrite_as_A":"A"}}` for a high pole, and the reverse pair (`A` then `B`) for a low pole (DECISIONS D4). Factorial rows may instead point at the source rewrite with `user_variant_id`; this attests to its checklist and code checks, with detailed evidence in WS1's attempts log. If `checklist` is included, all nine answers must be `yes`. Legacy numeric content/trait scores >=70 are also accepted. Authored system twins use `{"protocol":"authored_system_prompt","source_sha256":"..."}`. These record stimulus provenance, not a judge's empirical trait score. Neutral rows need no trait score. |
| `is_sample` | Boolean; only `true` for development fixtures, requiring `--allow-sample`. |

The validator checks IDs, schema, QC, domain and response consistency, neutral
messages, and allowed pole combinations. `matched_pairs` additionally checks
complete +/− paraphrase cells, identical fixed content, and balance across
scenarios. No mismatched row is silently dropped. For assigned localization,
positive and negative rows must share `(scenario, user_paraphrase,
sys_paraphrase)`, the same `user`, and the same `forced_response`. For user
localization they must share those keys, the same `system`, and response.

WS1 now provides 20 scenarios with complete Qwen-generated fixed responses
and 20 authored system prompts. The preparation step creates 420 rows:
20 neutral plus 400 system twins (100 +/- pairs per trait). It preserves the
exact canonical response for each scenario. WS1 now supplies user variants in
`Data_Creation/data/user_variants.jsonl`. Add `--user-variants` for WS3; the expected full
dataset is 660 rows with 3 user paraphrases per pole and 5 system paraphrases.

Build and run the available assigned dataset:

```bash
bash Workstream_2/scripts/uv_hdd.sh run ws2 prepare-data --out results/exp1/prepared
bash Workstream_2/scripts/uv_hdd.sh run ws2 preflight --data results/exp1/prepared/rows.jsonl \
  --out results/exp1/tokens
bash Workstream_2/scripts/uv_hdd.sh run ws2 run --data results/exp1/prepared/rows.jsonl \
  --config Workstream_2/configs/qwen.json --out results/exp1/assigned
```

For the generated user variants (WS3):

```bash
bash Workstream_2/scripts/uv_hdd.sh run ws2 prepare-data \
  --user-variants Data_Creation/data/user_variants.jsonl --out results/exp2/prepared
bash Workstream_2/scripts/uv_hdd.sh run ws2 run --data results/exp2/prepared/rows.jsonl \
  --config Workstream_2/configs/qwen.json --require-user --out results/exp2/full
```

`run` extracts once and produces E/A localization for all requested readouts,
control groups, heatmaps, SNR, and comparisons if user variants exist. Absent
user data is `NOT_AVAILABLE`; `--require-user` fails before loading weights.
Dropped rewrites that leave unmatched or unbalanced cells are rejected.
Regenerate those cells or explicitly select a complete balanced slice using
`prepare-data --scenarios code_01 ... --traits E`. No orphan row is dropped.

To repeat the current real-data software/hardware checks, including comparison
with independently executed unmodified reference extraction functions:

```bash
bash Workstream_2/scripts/uv_hdd.sh run python -m Workstream_2.scripts.verify_ws2_on_ws1 \
  --model full --device cuda:0 --out results/verification-full
```

Use `--model tiny` for random-weight integration testing. Reference function
agreement on WS1 stimuli is the WS2 implementation gate. The original humorous
experiment and its judged-data replay are OUT_OF_SCOPE under the 2026-10-01
team decision; no humorous CSVs, vectors or judge credentials are required.

The current 420-row assigned dataset completed full Qwen verification on the
RTX 3090. All six E/A/readout analyses completed, peak GPU allocation was
14.40 GiB, and the independent reference vector comparison was exact. Compact
results and manifest hashes are in
`Workstream_2/docs/full_qwen_verification_summary.json`. The completed HDD
report is `results/ws1-verification/full-qwen-gpu-retry/verification.json`.

## WS2 -> WS3: Python APIs

Stable public interfaces for version 1:

```python
from attentionseekers.config import load_config, ModelConfig
from attentionseekers.data import load_rows, validate_rows, matched_pairs
from attentionseekers.chat import encode, describe_encoding
from attentionseekers.extract import load_model, capture_encoded, extract_dataset, capture_hooks
from attentionseekers.heads import localize, head_contributions, control_groups, spearman
from attentionseekers.io import load_manifest, load_array, read_jsonl
from attentionseekers.prepare import prepare_dataset
from attentionseekers.pipeline import run_pipeline

config = load_config("Workstream_2/configs/qwen.json")
rows = load_rows("results/exp1/prepared/rows.jsonl")
model, tokenizer = load_model(config)
encoded = encode(tokenizer, rows[0], config.max_length, response_tokens=config.response_tokens)
readouts, _ = capture_encoded(model, config, encoded)
# readouts["first"], ["resp"], ["user"] are FP32 [layers, query_heads, head_dim]

bundle = extract_dataset(model, tokenizer, config, rows, "results/exp1/acts",
                         data_path="results/exp1/prepared/rows.jsonl")
```

`encode` tokenizes the template prefix and response separately and joins the
IDs. `first` is the final generation-prefix token, which predicts the first
assistant token. It is extracted in a prefix-only forward pass. `resp` is the
mean over the selected response window. The full fixed response is teacher-forced;
the default config selects its first 150 tokens (or all available tokens for a
shorter response). Index rows record `response_span` for the selected window,
`full_response_span`, `response_tokens_used`, and `response_tokens_total`.
`user` is the mean of user-content token positions. No special end-of-turn token
is appended to the fixed response.
Each hook reads the input to `model.model.layers[layer].self_attn.o_proj`,
which has `[1, tokens, query_heads * head_dim]`. `capture_hooks` is a context
manager and removes hooks even after exceptions. It never modifies activations.

For WS3 comparison, use independent assigned and user runs from the same
activation bundle and the same trait/readout:

```bash
uv run ws2 analyze --acts results/exp1/qwen-ea-acts --trait E \
  --source assigned --readout resp --out results/exp1/e-assigned
uv run ws2 analyze --acts results/exp1/qwen-ea-acts --trait E \
  --source user --readout resp --out results/exp1/e-user
uv run ws2 compare --assigned results/exp1/e-assigned \
  --user results/exp1/e-user --out results/exp1/e-comparison
```

Repeat for `A`, and optionally `first`. `localize` returns raw contribution
scores, upstream-style raw z scores, signed-log z scores, one-sided permutation
p values, BH q values, layer SNR, and activation contrasts. It averages
paraphrases within each scenario and flips entire scenario signs for tests.
The default 19,999 sign flips give a minimum Monte Carlo p of 0.00005, which
can support BH correction over 784 heads; use the saved permutation count when
reporting results. A tiny sample has too few scenarios for useful inference.
The head-score identity is checked in FP32. Head/layer numbers in all APIs and
files are zero indexed. Qwen SMHs: layer 19, heads 2, 4, 27. Control groups in
`metrics.json` are the SMHs, 5 seeded random triples, and the highest norm
triple from neutral first-token readouts. Groups can overlap.

For behavior generation, WS3 should use the same `system` and `user` messages
from the JSONL rows. The `forced_response` exists only for controlled
extraction, so do not feed it to free generation. The response-only judge and
its scores belong to WS3. WS3 should cite the activation bundle manifest hash
in its run metadata when joining behavior and activation rows by `id`.

## Artifact layout and verification

Each activation bundle is a new directory with:

```text
config.json                    exact model config, runtime versions, git/source hashes,
                               input-data hash, resolved model revision, mode, window,
                               is_synthetic and seed
index.jsonl                   original rows in tensor order plus row index and token spans
first.npy, resp.npy, user.npy  requested FP16 arrays [N, layers, query_heads, head_dim]
weights/layer_XX.npy          FP32 output projection [hidden_size, query_heads * head_dim]
manifest.json                 schema_version=1, status=complete, SHA256 of every file
```

An upstream replay bundle additionally has `output_resp.npy` (FP32
`[N,layers,hidden_size]`). Localization bundles save `raw.npy`, `raw_z.npy`,
`signed_log_z.npy`, `p.npy`, `q.npy`, `snr.npy`, `delta.npy`, `metrics.json`,
heatmaps, and `manifest.json`. `raw.npy` is `[layers, heads]`; `delta.npy` is
`[layers, heads, head_dim]`. The source bundle manifest SHA256 is in
`metrics.json`. Comparison bundles have `comparison.json` and `manifest.json`.

Before consuming a bundle, call `load_manifest(path)`; it rejects incomplete
or modified files. Always read `index.jsonl` for row order. Do not treat a
missing result as a zero effect. The sample and synthetic-model tags are
propagated through all downstream bundles. Older version 1 bundles without a
response window describe full-response readouts. Do not join bundles with
different windows or model revisions.

## Reference implementation agreement and current scope

The 2026-10-01 team decision in `Context_Documents/plan_team.md` and
`Context_Documents/plan_agent.md` excludes the original humorous generate/judge
experiment and judged-data replay. Those are OUT_OF_SCOPE, not missing WS2
completion gates. No judge calls or saved humorous vectors are required.

`reference_check.verify_reference_extraction` executes the unmodified imported
hooks and score functions independently on identical project inputs. The
recorded full-Qwen check used one E positive/negative pair: captured vectors
matched exactly, score Spearman was 1.0, and max raw-score error was 1.34e-5.
This is reference implementation validation, not original-paper reproduction.
The reference uses joint prompt+answer tokenization for this check; project
extraction continues to use the separately tokenized response convention above.

The published candidates remain layer 19 heads [2, 4, 27], zero indexed.
Localize all 784 heads independently on E/A system twins; report their ranks,
significance, heatmaps and controls without requiring those heads to dominate.
The original CSV replay adapter remains an optional legacy utility in `repro.py`
and the `reproduce` command, not part of the current workstream run.

## Refresh after system-prompt edits

Compare exact model inputs against the previous completed activation index.
Changing a system prompt invalidates every corresponding system-twin row's
first/response/user readouts. Unchanged E and neutral rows can be reused;
unchanged neutral first readouts preserve the previously selected controls.

```bash
bash Workstream_2/scripts/uv_hdd.sh run python -m Workstream_2.scripts.refresh_assigned \
  --previous results/ws1-verification/full-qwen-gpu-retry/pipeline/activations \
  --trait A --out results/ws2-refresh/A-system-prompts-next
```

This compares stable IDs and all input fields, rejects any change outside the
selected trait's system text, verifies the previous manifest/model settings,
and extracts only changed rows. It produces a combined activation bundle with
per-row reuse provenance, verifies reused arrays are bit identical, and runs
only the selected trait's three assigned readouts. Each run records previous
and changed bundle hashes. Use a new output directory for every run.
E localization and reference validation reports remain in their original
completed directories; they are not recomputed by this refresh command.

The 2026-10-01 A refresh is complete at
`results/ws2-refresh/A-system-prompts-2026-10-01-retry/`. It supersedes the
September 30 A reports; retain E and reference agreement from the earlier run.
The combined activation bundle has the latest 420 assigned/neutral rows.
Use `Workstream_2/docs/assigned_refresh_2026-10-01.json` for current artifact
paths, hashes, readout-specific ranks and unchanged-control verification.
