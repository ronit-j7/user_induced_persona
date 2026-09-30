# Data Creation

This folder builds the **user-style variants** (`user_variants.jsonl`). It rewrites each neutral scenario message in a high or low Extraversion or Agreeableness style, then quality-checks every rewrite with an LLM judge.

## Files

| File | What it is |
|---|---|
| `data/scenarios.jsonl` | The 20 base scenarios (4 domains × 5): intent, neutral user message `u0`, and forced response (null until Qwen runs). |
| `prompts.py` | **All prompt text used by the code:** trait definitions, rewriter prompt and examples, judge prompts. Edit prompts here. |
| `prompts/*.md` | Readable versions of the prompts, for review. Keep them in sync with `prompts.py`. |
| `prompts/sys_prompts.json` | The 20 system prompts for the assigned-persona set (5 per trait-pole, written in + / − pairs). |
| `llm.py` | API clients: DeepSeek (rewriter) and OpenAI (judge), plus the disk cache and token/cost tracking. |
| `gen_user_variants.py` | The pipeline. Run this. |
| `tests/smoke_test.py` | The full pipeline against fake APIs. No keys needed, no cost. |
| `GLOSSARY.md` | What "pole", "sys twin", "u0" and so on mean. |

## How it works

```
for each scenario × trait (E, A) × pole (+, −):            80 cells
    DeepSeek writes 3 paraphrases in one call
    for each paraphrase:
        1. code checks    length, unchanged-from-original, duplicate
        2. checklist      judge answers 9 yes/no questions (content, style boundaries, realism)
        3. forced choice  "which sounds more <pole>?" original vs rewrite, asked in both orders
        → accepted if everything passes
        → otherwise regenerated alone, with feedback listing every failed check
          (up to 3 attempts in total, then dropped and logged)
```

- **Models:** the rewriter is `deepseek-flash` (DeepSeek V4.1 Flash; the legacy name `deepseek-v4-flash` points to the same model), with thinking on and high effort. The judge is `gpt-6-luna`, with low reasoning effort for the checklist and none for the forced choice.
- **Different models on purpose:** LLM judges favour their own outputs.
- **Caching:** every successful API call is cached in `.cache/llm/`. Re-running never pays twice, and a crashed run resumes for free. **Delete the cache to force fresh generations.**

## Running

```bash
pip install openai tokenizers          # tokenizers is optional (for user_len_tokens)
export DEEPSEEK_API_KEY=...
export GPT_API_KEY=...                 # GPT_APT_KEY or OPENAI_API_KEY also work

cd Data_Creation
python gen_user_variants.py --dry-run                     # no API calls: prompt sizes + sample prompts
python gen_user_variants.py --scenarios code_01,explain_01,emo_01,brainstorm_01,code_02 --traits E   # first slice
python gen_user_variants.py                               # full set: 240 rows
python tests/smoke_test.py                                # sanity check, no keys needed
```

Useful flags: `--traits E,A` · `--poles +,-` · `--workers 8` · `--max-attempts 3` · `--no-tokenizer` · `--no-cache`

## Outputs

| Path | Contents |
|---|---|
| `data/user_variants.jsonl` | Accepted rows in the shared team schema. `id` looks like `code_01\|U\|E\|+\|k0`. |
| `data/qc/attempts.jsonl` | Every evaluated rewrite, with all judge answers, reasons and feedback, including failures. |
| `data/qc/dropped.jsonl` | Paraphrases that failed 3 attempts, with their last text and feedback. |
| `data/qc/summary.json` | Rows per pole, fail rate per check, attempts distribution, mean length per pole, duplicates. |
| `data/qc/usage.json` | Tokens and cost per model (DeepSeek at peak prices, so an upper bound). |
| `data/qc/manual_check.csv` | A blind 10% sample for humans: the same 9 questions plus the forced choice. |
| `data/qc/manual_check_key.csv` | Which of A/B is the rewrite. Keep it hidden from whoever fills in the sheet. |
| `data/qc/run_config.json` | Settings, arguments, git hash and timestamp of the run. |

## Worth knowing

- **`forced_response` is null** until Qwen generates the fixed replies. A later step fills it in.
- **`user_len_tokens`** needs the Qwen tokenizer, which is downloaded once from Hugging Face. Without it the field is null.
- **Length rule:** a rewrite must be between 0.5× and 2× the original's word count. Short originals may grow by up to 25 words.
- **Missing rows are expected:** a dropped paraphrase leaves its cell with fewer than 3 rows. `summary.json` lists the incomplete cells.
- **Settings** (models, reasoning effort, thresholds) live in `CONFIG` at the top of `gen_user_variants.py`.
