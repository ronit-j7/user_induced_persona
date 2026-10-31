# AttentionSeekers: The Story for the Final Report

How to turn everything we have (mid-sub results plus the overnight Ada runs) into one argument a reader can follow. Use this to order the report sections and the slides. Numbers marked **[judge]** are not available until the overnight generations are scored.

---

## The one-sentence version

> Qwen2.5-7B copies a user's enthusiasm, and the "style heads" that control an assigned persona do notice the user, but the copying does not run through those heads.

Every section below either sets up this sentence or tests it. If the judged projection results change the last clause (see "Possible endings"), rewrite this sentence first and the rest follows.

## The arc in five beats

Each beat is a question, the answer, and the evidence. Keep the report in this order. Each beat ends with the question that motivates the next.

### Beat 1. Do the published style heads care about Big Five personality? (Workstream 2)

- **Setup.** Izawa et al. (ICML 2026) found three heads (layer 19, heads 2, 4, 27) that switch an assistant's persona, using personas like "humorous". Ours are Big Five traits.
- **Answer.** Yes. With the personality in the system prompt, the three heads are the top 3 of layer 19 for both Extraversion and Agreeableness. This held at mid-sub (bf16) and again overnight (fp16).
- **New check (overnight, Job A).** The "wording-only" control: keep the personality the same and only change the wording of the system prompt. The same three heads are still the top 3 of layer 19. So they react to *any* system-prompt change, not only to personality.
- **But the size differs, mostly for head 2.** The personality contrast moves head 2 by about 30 (E) and 27 (A) on the head-score scale. The largest wording-only change on *any* head is 4–16, so head 2 moves 2–8 times more for personality than for rewording. Heads 4 and 27 (about 9–16) are no larger than the biggest rewording change.
- **How to say it.** "The style heads are the system prompt's most sensitive heads in layer 19. Head 2 responds to personality well beyond rewording; heads 4 and 27 respond about as much to rewording as to personality." Do not say the three heads are personality-specific.
- **Next question.** If they react to the system prompt, do they react to the *user*?

Evidence: `MIDSUB_RESULTS.md` (WS2), `Workstream_3/ws3_results/ada/jobA/analysis/E-assigned`, `A-assigned`, `same-pole/metrics.json`.

### Beat 2. Does Qwen copy the user's personality? (Workstream 3, experiment 1)

- **Answer.** For Extraversion, yes: +13 points on a 0–100 judge scale (d_z 1.39) when the user writes enthusiastically. It comes from sociability (+18.5) and energy (+22.5), not assertiveness (−1.8). It nearly disappears for emotional advice (+2.6).
- **Agreeableness.** Qwen stays at about 91/100 however the user writes. That is a ceiling, so the honest claim is "no copying visible", not "no copying". The pairwise A judge from Job A (`baselineA`) is the fix: it asks which of two replies is warmer, which does not need room above 91. **[judge]**
- **Next question.** Copying exists. Where in the model does the user's style go?

Evidence: `Workstream_3/docs/ws3_summary.md`, mirroring table.

### Beat 3. Does the user's style reach the style heads? (Workstream 3, experiments 2 and 3)

- **Answer.** It reaches the same heads, weakly and in a different direction.
  - Same heads: the user contrast also ranks the style heads in the top 4 of layer 19 (mid-sub and overnight).
  - Weak: in the 2×2 mixed set, the system prompt explains about 69% of what the style heads do, and the user about 0.6%. Overnight, the user contrast moves head 2 by about 0.7 against about 30 for the system prompt.
  - Different direction: the user's push on the style heads is no more aligned with the system prompt's push than a random set of three heads.
- **Next question.** Weak is not the same as unused. Is that small signal what makes Qwen copy the user? That needs a causal test.

### Beat 4. Does the copying run through the style heads? (the causal tests)

This is the new result for the final submission.

- **Mid-sub test: switch the heads off.** Zeroing the three heads did not reduce copying (13 → 14 points), the same as zeroing random heads. That is a blunt test: it removes everything the heads do, not just the user's part.
- **Overnight test: remove only the user's direction.** For each group of heads, we learned the direction that user enthusiasm pushes activations in, then subtracted it during generation (λ = 1). Directions were learned on 15 scenarios and applied to the held-out 5 (4 folds), so nothing is tested on its own training data. Decoding was greedy, so each arm differs from its baseline only because of the intervention.
- **Arms and what each one rules out.**

| Arm | What is removed | Role in the story |
|---|---|---|
| `smh_l1` | User direction in the 3 style heads | The main test |
| `rand3_l1` | Same, in 3 random heads (never 2, 4, 27) | Is any 3-head removal enough? |
| `l19_l1` | Same, across all 28 heads of layer 19 | Is the signal in layer 19 but not in the style heads? |
| `top10_l1` | Same, in the 10 heads most moved by the user | Is it in other heads entirely? |
| `randdir19_l1` | A random direction in layer 19 | Does any edit of this size change copying? |
| `resid19_l1` | User direction in the whole layer-19 output | Positive control: can removing a direction stop copying at all? |
| `smh_l2`, `l19_l2` | The direction is flipped, not removed | Steering: can the direction push style at all? |

- **What we already know without the judge.**
  - Edits are real and targeted. Compared with the baseline, the style-head removal changes 61% of replies, all of layer 19 66%, and the residual control 98%. The random-head and random-direction controls change only 30% and 21%.
  - Replies stay fluent. Spot checks of the strongest arms read like normal answers.
  - Runs are reproducible. Both accounts produced byte-identical baselines (200/200), so greedy decoding worked.
  - A crude enthusiasm count (exclamation marks, emoji, words like "exciting") hints that the residual control removes most of the user effect, and the style-head removal does little. Treat this only as a hint. The judged numbers decide.
- **The judged result. [judge]** For each arm: copying (high minus low user, from the 120 mirroring replies), and from the 2×2 set: the system-prompt effect, the user effect under a system prompt, and the conflict gap. `ws3.intervention_report` builds the table and the figure.

### Beat 5. What it means

Pick the ending that matches the judged table (next section), then close with:

- what this says about how assistants handle *who is talking* versus *who they were told to be*;
- why it matters for persona safety: the system prompt is the strong, localized control; user influence is weaker and spread out, so it may not be fixable by editing three heads;
- the limitations (below).

---

## Possible endings (decide after judging)

Read the judged table in this order. Compare each arm only to the baseline from its own job.

1. **Does the residual control (`resid19_l1`) reduce copying?**
   - **No:** removing one direction at layer 19 does not stop copying anywhere. Ending: "Single-turn copying is not carried by one direction at layer 19." The style-head null stays a null, and you cannot say more about where copying lives.
   - **Yes:** go to 2.
2. **Does the style-head removal (`smh_l1`) reduce copying more than the random-head removal (`rand3_l1`)?**
   - **Yes:** "The style heads carry the user's style after all. Zeroing them at mid-sub was too blunt to show it." This would overturn the mid-sub null and becomes the headline.
   - **No:** go to 3.
3. **Do `l19_l1` or `top10_l1` reduce it (more than `randdir19_l1`)?**
   - **Yes:** "Copying uses the user direction, but through other heads, not the style heads." This is the most likely ending given the hints, and it fits the one-sentence version.
   - **No:** "Copying runs through the residual stream at layer 19, not through any head group we tested."

Also report, whatever the ending:

- **Persona preservation.** Does removing the user direction leave the system-prompt effect intact? If the system effect also collapses, the edit is too broad and the claim weakens.
- **Steering (λ = 2).** If flipping the direction makes replies *less* enthusiastic for enthusiastic users, the direction is causally about style even if removal alone is small.

---

## Figures and tables to make

| # | What | Source | Status |
|---|---|---|---|
| 1 | Head heatmaps, system E and A, style heads marked | `ws3_results/ada/jobA/analysis/*/raw_z_heatmap.png` | Exists |
| 2 | Trait vs wording-only effect on the style heads (bar chart) | `ws3_results/ada/figures/wording_control.png` | Exists |
| 3 | Mirroring by facet and domain | `ws3_summary.md` | Exists as table |
| 4 | System vs user share of style-head variation (69% vs 0.6%) | mid-sub factorial | Exists as numbers |
| 5 | Main result: copying per arm, with confidence intervals | `ws3.intervention_report` | **[judge]** |
| 6 | 2×2: system effect, user effect, conflict gap, per arm | `ws3.intervention_report` | **[judge]** |
| 7 | Pairwise agreeableness | `ws3.pairwise` | **[judge]** |

Figure 5 is the paper's main figure. Put it in the abstract and on the first results slide.

## What to say in Limitations

- Single-turn conversations only; no drift over a long chat.
- One model (Qwen2.5-7B-Instruct), two traits (E, A).
- The 2×2 conflict results use the sociability facet only (paraphrase 0).
- The overnight runs used fp16 on GTX 1080 Ti cards with greedy decoding and 150-token replies. Mid-sub used bf16 on an RTX 3090 at temperature 1 with 300 tokens. Compare arms only within the same job.
- The judge is an LLM (GPT-4.1-mini) and never sees the user's message. The human 10% audit (`Data_Creation/data/qc/manual_check.csv`) is the check on the data; report its numbers.
- Agreeableness is at a ceiling on the absolute scale; only the pairwise judge can speak to it.

## Section outline for the report

1. **Introduction:** assistants are told who to be (system prompt) but also hear who they talk to (user). Do they copy the user, and does it go through the same machinery as an assigned persona?
2. **Related work:** Izawa et al. (style heads), persona vectors, user modeling and sycophancy. Fix the citations listed in `plan_end_submission.md`.
3. **Data:** scenarios, user variants, system twins, the 2×2 set, QC and the human audit.
4. **Method:** head scores, the wording-only control, the judge (never sees the user), direction removal with cross-fitting, greedy decoding, controls.
5. **Results:** Beats 1–4 in order, one subsection each.
6. **Discussion:** Beat 5 and the chosen ending.
7. **Limitations and future work:** the list above, plus Llama, the other three traits and multi-turn drift.

## Still to do before the story is complete

- [ ] Judge every arm in both jobs (needs `OPENAI_API_KEY`; about an hour).
- [ ] Pairwise agreeableness on `jobA/arms/baselineA`.
- [ ] `ws3.intervention_report` → Figures 5 and 6.
- [ ] Human 10% audit.
- [ ] Pick the ending, rewrite the one-sentence version if needed, then write.
