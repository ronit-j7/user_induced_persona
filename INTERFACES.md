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
| `qc` | User rewrites: WS1's `{"passed":true,"forced_choice":{"rewrite_as_B":"B","rewrite_as_A":"A"}}`; this attests to its checklist and code checks, with detailed evidence in WS1's attempts log. If `checklist` is included, all nine answers must be `yes`. Legacy numeric content/trait scores >=70 are also accepted. Authored system twins use `{"protocol":"authored_system_prompt","source_sha256":"..."}`. These record stimulus provenance, not a judge's empirical trait score. Neutral rows need no trait score. |
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
exact canonical response for each scenario. Current user variants are still
absent. Once WS1 supplies them, add `--user-variants`; the expected full
dataset is 660 rows with 3 user paraphrases per pole and 5 system paraphrases.

Build and run the available assigned dataset:

```bash
bash Workstream_2/scripts/uv_hdd.sh run ws2 prepare-data --out results/exp1/prepared
bash Workstream_2/scripts/uv_hdd.sh run ws2 preflight --data results/exp1/prepared/rows.jsonl \
  --out results/exp1/tokens
bash Workstream_2/scripts/uv_hdd.sh run ws2 run --data results/exp1/prepared/rows.jsonl \
  --config Workstream_2/configs/qwen.json --out results/exp1/assigned
```

After real user variants are generated:

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
agreement on WS1 stimuli and paper-head recovery on judged humorous data are
separate checks. The verification script does not claim the latter.

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

## Independent reproduction gate

The vendored, unmodified reference is in `Workstream_2/reference_repos/style-modulation-head`
(commit recorded in `Workstream_2/reference_repos/UPSTREAM.md`). Use its documented `uv`
environment, CSV generation, vector extraction, and head analysis on
`humorous`; preserve its outputs and model revision. It uses joint
`prompt + answer` tokenization and separately counts prompt tokens. Our
`reproduce` command deliberately replays that convention only for the upstream
comparison. Project datasets always use `encode` above.

From `Workstream_2/reference_repos/style-modulation-head`, with its own Python 3.10 `uv`
environment and judge credentials, the direct upstream commands are:

```bash
uv sync
uv run python src/eval/eval_persona.py \
  --model Qwen/Qwen2.5-7B-Instruct --trait humorous \
  --output_path data/eval_persona_extract/Qwen/Qwen2.5-7B-Instruct/humorous_pos_instruct.csv \
  --persona_instruction_type pos --assistant_name humorous \
  --judge_model gpt-4.1-mini-2025-04-14 --version extract
uv run python src/eval/eval_persona.py \
  --model Qwen/Qwen2.5-7B-Instruct --trait humorous \
  --output_path data/eval_persona_extract/Qwen/Qwen2.5-7B-Instruct/humorous_neg_instruct.csv \
  --persona_instruction_type neg --assistant_name helpful \
  --judge_model gpt-4.1-mini-2025-04-14 --version extract
uv run python src/generate_vec/generate_vec_head.py \
  --model_name Qwen/Qwen2.5-7B-Instruct \
  --pos_path data/eval_persona_extract/Qwen/Qwen2.5-7B-Instruct/humorous_pos_instruct.csv \
  --neg_path data/eval_persona_extract/Qwen/Qwen2.5-7B-Instruct/humorous_neg_instruct.csv \
  --trait humorous --save_dir data/persona_vectors/Qwen/Qwen2.5-7B-Instruct
uv run python src/generate_vec/generate_vec_block.py \
  --model_name Qwen/Qwen2.5-7B-Instruct \
  --pos_path data/eval_persona_extract/Qwen/Qwen2.5-7B-Instruct/humorous_pos_instruct.csv \
  --neg_path data/eval_persona_extract/Qwen/Qwen2.5-7B-Instruct/humorous_neg_instruct.csv \
  --trait humorous --save_dir data/persona_vectors/Qwen/Qwen2.5-7B-Instruct
uv run python -m src.head_analysis.head_contribution.main analyze_trait \
  --model_name Qwen/Qwen2.5-7B-Instruct \
  --vector_dir data/persona_vectors/Qwen/Qwen2.5-7B-Instruct \
  --trait humorous --vector_type response_avg
```

The upstream `scripts/generate_all_vectors.sh` in this imported commit refers
to a missing `src/save_model_attn_config.py`, so use the direct commands above.
`generate_vec_head.py` saves the attention config itself. The reference source
is preserved as imported; this project does not patch it silently.

```bash
uv run ws2 reproduce \
  --pos-csv /path/to/humorous_pos_instruct.csv \
  --neg-csv /path/to/humorous_neg_instruct.csv \
  --upstream-vectors /path/to/upstream/persona_vectors \
  --trait humorous --config Workstream_2/configs/qwen.json \
  --acts-out results/exp0/humorous-replay-acts \
  --report-out results/exp0/humorous-repro
```

The adapter uses the upstream row-paired filter (`positive trait >= 50`,
`negative trait < 50`, both coherence >= 50). It compares our independently
captured pre-o_proj and attention-output response vectors with upstream's
saved vector files and applies the same weights to both. `repro.json` records
both top-3 lists, layer-19 score Spearman, and vector RMSE. Claim the gate
only if the upstream top-3 is `{2,4,27}`, our top-3 matches it, and
rho > 0.95. These checks have not yet run on the full Qwen checkpoint.

The upstream source's current `normalize_matrix` z-scores raw scores despite
its signed-log description. Both normalizations are saved in our results;
upstream replay comparison uses raw scores. Reproduction and Big Five twins
use different data and must be reported separately.
