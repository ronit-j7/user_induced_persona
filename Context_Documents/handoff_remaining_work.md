# Handoff: Remaining Work Before the Report and Slides

Follow this top to bottom. The only code left is **scoring the overnight generations with the judge** and running two scripts on the scores. Everything else is reading, a 1-hour human check, and writing.

No GPU is needed for anything here. A laptop is enough.

---

## 0. Context in one minute

- **What we study.** Does Qwen2.5-7B-Instruct copy the personality of the user's message (it does for Extraversion: +13 points at mid-sub), and does that copying run through the "style heads" (layer 19, heads 2, 4, 27) that control a system-prompt persona?
- **What ran overnight (done).** The causal test. For each reply, we removed the direction that the user's enthusiasm pushes the model's activations in, in different places, and regenerated. Comparing each "arm" to the unedited baseline tells us where the copying lives.
- **What is missing.** The generations are plain text. Until the judge scores them, we do not know whether any edit reduced copying.
- **Read first:** `Context_Documents/story_end_submission.md` (the argument of the report, figure list, how to pick the ending). Background: `MIDSUB_RESULTS.md`, `Workstream_3/docs/ws3_summary.md`, `Context_Documents/plan_end_submission.md` (why each arm exists).

## 1. Where everything is

All overnight results are committed under `Workstream_3/ws3_results/ada/`.

| Path | What |
|---|---|
| `jobA/arms/<arm>/gen.jsonl` | Generations. Arms: `baseline`, `smh_l1`, `smh_l2`, `rand3_l1`, `baselineA` |
| `jobB/arms/<arm>/gen.jsonl` | Generations. Arms: `baseline`, `l19_l1`, `l19_l2`, `top10_l1`, `randdir19_l1`, `resid19_l1` |
| `job*/arms/<arm>/arm.json` | Settings of each arm (which heads, λ, decoding) |
| `job*/directions/` | The learned user-style directions, per fold |
| `jobA/analysis/` | Head analysis: system-prompt E and A, user E, and the wording-only check (`same-pole/metrics.json`) |
| `figures/wording_control.png` | Figure 2 of the story doc (already made) |

What the arm names mean:

| Arm | Edit | Question it answers |
|---|---|---|
| `baseline` | none | Reference. Both jobs have identical baselines. |
| `smh_l1` | Remove user direction from the 3 style heads | **Main test:** do the style heads carry the copying? |
| `smh_l2`, `l19_l2` | Flip the direction instead of removing it | Steering: can the direction push style at all? |
| `rand3_l1` | Same removal in 3 random heads (never 2, 4, 27) | Control for `smh_l1` |
| `l19_l1` | Removal across all 28 heads of layer 19 | Is it in layer 19 but outside the style heads? |
| `top10_l1` | Removal in the 10 heads the user moves most | Is it in other heads? |
| `randdir19_l1` | A random direction in layer 19 | Control: does any edit of this size matter? |
| `resid19_l1` | Removal from the whole layer-19 output | Positive control: can removal stop copying at all? |
| `baselineA` | none, Agreeableness user messages | For the pairwise Agreeableness judge |

Each E arm has 200 replies: 120 "mirroring" replies (enthusiastic vs reserved user, neutral system prompt) and 80 "2×2" replies (system prompt high/low × user high/low).

## 2. One-time setup (10 minutes)

```bash
git pull
# install uv if missing: https://docs.astral.sh/uv/getting-started/installation/
uv sync --no-group model          # skips torch; judging does not need it
```

Every command below is run from the repo root.

## 3. Get an OpenAI API key

The key that was shared in the group chat is **expired** (OpenAI returns `expired_secret_key`). Make a new one:

1. Log in at <https://platform.openai.com>.
2. **Settings → Billing**: make sure there is credit. The whole job costs about **$1–3**; adding $5 is plenty.
3. **API keys → Create new secret key.** If you use a project key, check that the project is allowed to use `gpt-4.1-mini` (Settings → Project → Limits / model access).
4. Copy it once. Put it **only** in your shell, never in a file in the repo and never in chat:

```bash
export OPENAI_API_KEY='sk-...'
```

The judge must be **`gpt-4.1-mini`**. That is what the mid-submission used, and the setting in `Workstream_3/configs/ws3.json`. Do not change the model, or the new numbers cannot be compared with the mid-sub +13.

**Test the key (costs a fraction of a cent):**

```bash
uv run --no-group model python -m ws3.judge \
  --generations Workstream_3/ws3_results/ada/jobA/arms/baseline/gen.jsonl \
  --out /tmp/judge_test --cache results/judge_cache --limit 2
```

Good: it prints `[progress] 8/8` and writes `/tmp/judge_test/scores.jsonl`. Bad: `{"status": "BLOCKED"...}` means the key is not exported in this shell; a `401` means the key is wrong or expired.

## 4. Score the baselines first and check the copying is there

```bash
for job in jobA jobB; do
  uv run --no-group model python -m ws3.judge \
    --generations Workstream_3/ws3_results/ada/$job/arms/baseline/gen.jsonl \
    --out Workstream_3/ws3_results/ada/judged/$job/baseline \
    --cache results/judge_cache
done
uv run --no-group model python -m ws3.intervention_report \
  --job A=Workstream_3/ws3_results/ada/judged/jobA \
  --job B=Workstream_3/ws3_results/ada/judged/jobB \
  --out Workstream_3/ws3_results/ada/report
```

The jobB baseline is identical to jobA's, so its scores come from the cache for free.

Look at `M=` on the `baseline` lines. That is the copying effect (high user minus low user, judge points). Mid-sub had 13, but with random sampling and 300-token replies; these are greedy with 150 tokens, so a different number is expected.

- **M around 5 or more:** good, continue.
- **M below about 3:** continue anyway, but tell the team: the removal test then has little to remove, and the report must say so.

## 5. Score every arm (about 30–60 minutes)

```bash
for arm_dir in Workstream_3/ws3_results/ada/job*/arms/*/; do
  job=$(basename "$(dirname "$(dirname "$arm_dir")")")
  arm=$(basename "$arm_dir")
  uv run --no-group model python -m ws3.judge \
    --generations "${arm_dir}gen.jsonl" \
    --out "Workstream_3/ws3_results/ada/judged/$job/$arm" \
    --cache results/judge_cache
done
```

About 7,700 judge calls in total (each reply is scored on Extraversion, Agreeableness, coherence and refusal). The judge only sees the task and the reply, never the user's message.

**If it stops** (network, rate limit, laptop sleep): run the same loop again. Finished replies are skipped and every API call is cached in `results/judge_cache`, so nothing is paid twice.

## 6. Pairwise Agreeableness (5 minutes)

Absolute Agreeableness is stuck near 91/100, so it cannot show copying. The pairwise judge asks "which of these two replies is warmer?" instead.

```bash
uv run --no-group model python -m ws3.pairwise --trait A \
  --generations Workstream_3/ws3_results/ada/jobA/arms/baselineA/gen.jsonl \
  --out Workstream_3/ws3_results/ada/pairwise/A --cache results/judge_cache
uv run --no-group model python -m ws3.pairwise --trait E \
  --generations Workstream_3/ws3_results/ada/jobA/arms/baseline/gen.jsonl \
  --out Workstream_3/ws3_results/ada/pairwise/E --cache results/judge_cache
```

Read `summary.json`: `mean_p_high` is how often the reply to the high-trait user was picked. 0.5 means no copying. The E run is a sanity check: it should come out clearly above 0.5.

## 7. Build the main table and figure

```bash
uv run --no-group model python -m ws3.intervention_report \
  --job A=Workstream_3/ws3_results/ada/judged/jobA \
  --job B=Workstream_3/ws3_results/ada/judged/jobB \
  --out Workstream_3/ws3_results/ada/report
```

Outputs in `Workstream_3/ws3_results/ada/report/`:

- `metrics.json`: every number, with confidence intervals.
- `intervention.png`: **the main figure** of the report.

How to read each printed line:

| Field | Meaning | What we hope to see |
|---|---|---|
| `M` | Copying in this arm (judge points) | Lower than the baseline for a working removal |
| `reduction` | Share of the baseline copying removed (0.3 = 30%) | Clearly above 0 for a working removal; near 0 for the random controls |
| `assigned` | Effect of the system-prompt persona in the 2×2 | About the same as the baseline (the edit must not wipe out the persona) |
| `user\|persona` | Copying of the user while a persona is set | Lower than baseline for a working removal |
| `coherence` | Judge's fluency score | Close to the baseline (the edit must not break the text) |

The confidence intervals for each arm versus its baseline are in `metrics.json` under `mirroring_vs_baseline` and `factorial_*_vs_baseline`. Use those in the report, not just the means.

Then pick the ending with the decision list in `story_end_submission.md` ("Possible endings").

## 8. Commit the scores

```bash
git add Workstream_3/ws3_results/ada/judged Workstream_3/ws3_results/ada/pairwise Workstream_3/ws3_results/ada/report
git commit -m "Judge overnight Ada arms and build the intervention report"
git push
```

Do not commit `results/judge_cache` (it is gitignored) or the API key.

## 9. Human 10% audit (about 1 hour, no code)

This checks that the data the whole project rests on is valid. It is the one item reviewers flagged at mid-sub that is still open.

1. Open `Data_Creation/data/qc/manual_check.csv` (24 rows) in a spreadsheet. **Do not open `manual_check_key.csv` until you are done.**
2. For each row, compare `original` and `rewrite` and fill columns `C1`–`R3` with `yes` or `no`. `yes` always means the rule is satisfied. The questions are in `Data_Creation/prompts/qc_judge_prompt.md` (C1–C4 content kept, L1 other trait unchanged (its question is in the `L1_question` column), L2 no added distress, R1–R3 realistic and safe).
3. For the forced choice, read `message_A` and `message_B` and write `A` or `B` in `choice_A_or_B` for the answer to `forced_choice_question`.
4. Then open `manual_check_key.csv` and count:
   - checklist pass rate: share of all `yes` answers;
   - forced-choice agreement: share of rows where your `choice_A_or_B` equals `expected_choice`.
5. Save the filled sheet as `Data_Creation/data/qc/manual_check_filled.csv`, commit it, and put the two numbers in the report's Data section.

## 10. Then write

- Report and slides follow `story_end_submission.md`: five beats, the figure list, and the Limitations list.
- Figures already available:
  - head heatmaps: `Workstream_3/ws3_results/ada/jobA/analysis/*/raw_z_heatmap.png`;
  - wording control: `Workstream_3/ws3_results/ada/figures/wording_control.png`;
  - the main result: `Workstream_3/ws3_results/ada/report/intervention.png` (after step 7).
- Citations to fix are listed in `plan_end_submission.md` (Report section).

## Checklist

- [ ] New API key, tested (step 3)
- [ ] Baselines scored, M checked (step 4)
- [ ] All arms scored (step 5)
- [ ] Pairwise A and E (step 6)
- [ ] `intervention_report` table and figure (step 7)
- [ ] Results committed (step 8)
- [ ] Human audit numbers (step 9)
- [ ] Ending chosen, report and slides written (step 10)

## Troubleshooting

| Message | Fix |
|---|---|
| `{"status": "BLOCKED", "reason": "OPENAI_API_KEY missing"}` | `export OPENAI_API_KEY=...` in the same terminal |
| `401` / `expired_secret_key` / `invalid_api_key` | Make a new key (step 3) |
| `model_not_found` or permission error for `gpt-4.1-mini` | Allow the model in the project's settings, or use a key from a project that has it |
| `429` / rate limit | Wait; the code retries automatically. If it keeps failing, add `--workers 4` |
| `insufficient_quota` | Add billing credit |
| `missing_E` above 0 in the report | A few replies got no usable score; rerun step 5 (cached calls are free). A handful is fine; report the count |
