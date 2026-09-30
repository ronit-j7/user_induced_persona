# AttentionSeekers shared interface (version 1)

This is the handoff contract for Data Creation (WS1), SMH localization (WS2),
and user-induced experiments (WS3). Change the schema version and this file
together when a breaking change is necessary. Code entry points live in the
`attentionseekers` package; the CLI is `ws2` or `python -m attentionseekers`.

## Environment and first run

From the repository root, use `uv` for dependency management:

```bash
uv sync --extra model --extra test
uv run python scripts/make_sample_ws2.py
uv run ws2 validate-data --data Data_Creation/samples/ws2_sample.jsonl --allow-sample
uv run pytest -q
uv run python -m attentionseekers.smoke --out results/smoke-cpu
```

`results/smoke-cpu` uses random tiny Qwen weights. It is a software smoke test,
never evidence of an SMH or a Big Five result. It creates activation,
localization, and comparison bundles. On a CUDA machine, add `--device cuda:0`
and use a different output directory. Every `--out` directory must be new;
the CLI refuses to overwrite previous runs.

The full Qwen config is `configs/qwen.json`. `uv run ws2 extract ...` loads the
specified Hugging Face checkpoint on GPU 0 in bf16; this has only been tested
on a tiny random Qwen, so run the real Qwen gates below before research use.

## WS1 -> WS2: JSONL version 1

One UTF-8 JSON object per line. The checked-in sample at
`Data_Creation/samples/ws2_sample.jsonl` is executable schema documentation.
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
| `qc` | Required on real styled rows: `{"passed": true, "content": >=70, "trait": >=70}` with finite scores <=100. Extra fields are allowed. Neutral base rows are human-approved and do not require a trait score. |
| `is_sample` | Boolean; only `true` for development fixtures, requiring `--allow-sample`. |

The validator checks IDs, schema, QC, domain and response consistency, neutral
messages, and allowed pole combinations. `matched_pairs` additionally checks
complete +/− paraphrase cells, identical fixed content, and balance across
scenarios. No mismatched row is silently dropped. For assigned localization,
positive and negative rows must share `(scenario, user_paraphrase,
sys_paraphrase)`, the same `user`, and the same `forced_response`. For user
localization they must share those keys, the same `system`, and response.

WS1 needs to generate Qwen's fixed response for each base scenario and the
assigned/user variants. The existing 20 base scenarios have null responses and
cannot be used for `resp` extraction yet. A complete current run is 20
scenarios, E/A poles, 3 user paraphrases, 5 system paraphrases, plus one
neutral row per scenario. Send a first slice through the same validator.

```bash
uv run ws2 validate-data --data Data_Creation/data/ws2_ready.jsonl
uv run ws2 extract --data Data_Creation/data/ws2_ready.jsonl \
  --config configs/qwen.json --out results/exp1/qwen-ea-acts
```

## WS2 -> WS3: Python APIs

Stable public interfaces for version 1:

```python
from attentionseekers.config import load_config, ModelConfig
from attentionseekers.data import load_rows, validate_rows, matched_pairs
from attentionseekers.chat import encode, describe_encoding
from attentionseekers.extract import load_model, capture_encoded, extract_dataset, capture_hooks
from attentionseekers.heads import localize, head_contributions, control_groups, spearman
from attentionseekers.io import load_manifest, load_array, read_jsonl

config = load_config("configs/qwen.json")
rows = load_rows("Data_Creation/data/ws2_ready.jsonl")
model, tokenizer = load_model(config)
encoded = encode(tokenizer, rows[0], config.max_length)
readouts, _ = capture_encoded(model, config, encoded)
# readouts["first"], ["resp"], ["user"] are FP32 [layers, query_heads, head_dim]

bundle = extract_dataset(model, tokenizer, config, rows, "results/exp1/acts",
                         data_path="Data_Creation/data/ws2_ready.jsonl")
```

`encode` tokenizes the template prefix and response separately and joins the
IDs. `first` is the final generation-prefix token, which predicts the first
assistant token. It is extracted in a prefix-only forward pass. `resp` is the
mean of the fixed response-token positions. `user` is the mean of user-content
token positions. No special end-of-turn token is appended to the fixed response.
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
                               input-data hash, resolved model revision, mode and seed
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
missing result as a zero effect. The sample tag is propagated through all
downstream bundles.

## Independent reproduction gate

The vendored, unmodified reference is in `reference_repos/style-modulation-head`
(commit recorded in `reference_repos/UPSTREAM.md`). Use its documented `uv`
environment, CSV generation, vector extraction, and head analysis on
`humorous`; preserve its outputs and model revision. It uses joint
`prompt + answer` tokenization and separately counts prompt tokens. Our
`reproduce` command deliberately replays that convention only for the upstream
comparison. Project datasets always use `encode` above.

From `reference_repos/style-modulation-head`, with its own Python 3.10 `uv`
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
  --trait humorous --config configs/qwen.json \
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
