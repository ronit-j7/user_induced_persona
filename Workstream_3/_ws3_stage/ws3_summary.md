# Workstream 3 results (Qwen2.5-7B-Instruct)

Single-turn user-style experiments on the RTX 3090. Model revision `a09a35458c702b33eeacc393d103063234e8bc28`, bf16, response readouts use the first 150 tokens of the fixed reply. Head numbers are zero-indexed. The published style-modulation heads (SMHs) are layer 19, heads 2, 4 and 27.

Large arrays stay on the hard disk under `results/ws3/` (`/media/gaurav/Data21/eshaan/results/user_induced_persona`). This note is the compact record. The judge was `gpt-4.1-mini` (resolved to `gpt-4.1-mini-2025-04-14`) with logprob-weighted 0–100 scores. Prompts contained the neutral task intent and the assistant reply only.

## User-style vs assigned heads

`ws2 run --require-user` extracted 660 rows (20 neutral, 400 system twins, 240 user variants) and localized both sources. Spearman correlation of raw head scores is about 0.56–0.62 at the first-token readout and about 0.78 at the response readout.

On the response readout, the SMHs rank in the top 4 of layer 19 for the user contrast as well as the assigned contrast (E user ranks 1, 2, 4; A user ranks 1, 2, 3). At the first token, only head 27 stays in the top 3 for the user contrast; heads 2 and 4 fall to ranks 10–14.

The cosine between assigned and user contrasts inside the SMH group is positive (E first 0.42, E response 0.23, A first 0.55, A response 0.38). Against 1,000 random 3-head groups in layer 19, that cosine sits at the 85th, 57th, 92nd and 71st percentiles. The response-readout SMH cosine is close to the average absolute cosine of a random triple, so shared rank is clearer than shared direction.

Between 669 and 703 of 784 heads pass BH q < 0.05 for both sources. A same-pole null, pairing system paraphrases k=0 and k=1 with the pole held fixed, marks 691–707 heads significant (assigned response contrasts were 685 for E and 703 for A). The test flags wording changes, not only the trait. Ranks and effect sizes are the quantities to report.

## Behavioral mirroring

540 free generations: every user paraphrase, two samples, plus three neutral samples per scenario. Temperature 1.0, 300 new tokens. The judge scored E, A, coherence and refusal on the reply plus the neutral intent.

| Trait | M (high − low) | d_z | Wilcoxon p | 95% CI |
| --- | --- | --- | --- | --- |
| Extraversion | 13.05 | 1.39 | 1.9e-5 | [8.98, 17.05] |
| Agreeableness | 0.48 | 0.31 | 0.097 | [−0.16, 1.16] |

Qwen mirrors extraversion and does not clearly mirror agreeableness. Mean E scores are 49.0 (high), 36.0 (low) and 33.7 (neutral). Mean A scores sit near 91 for high, low and neutral, so the model stays agreeable. Refusal rate is 0 on both poles. Replies average about 290 tokens, close to the 300-token cap, on every pole.

Extraversion mirroring is not uniform across facets (DECISIONS D3). Sociability (k=0) is +18.5 and energy (k=2) is +22.5. Assertiveness (k=1) is −1.8 (p = 0.67). By domain, E is about +14 to +19 on coding, explaining and brainstorming, and only +2.6 on emotional advice. Agreeableness stays within about one point in every domain, including emotional advice.

## Factorial regression (Extraversion, stretch)

360 factorial rows, head groups of 384 dimensions, predictors centered within scenario. Assigned persona dominates. On the response readout at the SMHs, R² is 0.69 for the assigned block and 0.006 for the user block. The user coefficient norm is still above a within-cell shuffle null (p = 0.001), and the signed cosine with the assigned pole is 0.20, CI [0.16, 0.23]. The first-token SMH cosine is 0.27, CI [0.22, 0.31], with user R² 0.003. Several random triples have a similar or larger cosine, and their user R² is also only a few percent. User style and the system prompt are not interchangeable as drivers of these activations.

## SMH ablation (stretch)

The same 540 prompts were regenerated with three heads zeroed at `o_proj` for the whole forward pass: the SMHs, and two random triples from the saved controls ([14, 17, 22] and [0, 1, 2]).

| Arm | E mirroring | A mirroring |
| --- | --- | --- |
| No ablation | 13.05 | 0.48 |
| SMH zeroed | 13.69 | 0.19 |
| Random [14, 17, 22] | 12.88 | 0.24 |
| Random [0, 1, 2] | 14.36 | 0.43 |

Zeroing the SMHs does not reduce single-turn extraversion mirroring relative to the random triples. Agreeableness stays near zero in every arm.

## What this supports

User style and assigned persona land on overlapping heads, and the SMHs are among the top response-readout heads for both. The behavioral effect is an extraversion effect, concentrated in sociability and energy. Agreeableness in the reply barely moves. The user direction in the factorial activation regression is small next to the system prompt, and turning the SMHs off does not remove the extraversion effect. Single-turn mirroring is real for extraversion and is not explained by a simple "the SMHs are the mirroring circuit" story.
