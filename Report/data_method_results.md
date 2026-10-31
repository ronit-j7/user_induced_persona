<!--
Draft of four sections of the mid-submission report: Dataset, Method, Results, Future Work and Timeline.
Numbering assumes Section 1 is the Introduction and Section 2 is Related Work.
Citations are author-year placeholders for the LaTeX pass.
-->

## 3 Dataset

No existing dataset holds the personality of a conversation fixed in one place while varying it in another. We therefore built one. It has a single property throughout: within a scenario, two inputs differ only in personality, never in what the user asks or in the assistant's reply.

### 3.1 Traits and scenarios

We study two Big Five traits, Extraversion (E) and Agreeableness (A), each with a high (+) and a low (−) pole. We define each pole through the facets of the BFI-2 (Soto & John, 2017): sociability, assertiveness and energy for Extraversion, and compassion, respectfulness and trust for Agreeableness. We use adjective markers from Saucier (1994) and linguistic cues of personality from Pennebaker & King (1999) and Mairesse et al. (2007). We chose the BFI-2 over the NEO-PI-R because the NEO-PI-R places warmth under Extraversion, which blurs the two traits we want to keep apart.

A scenario is one request a user might make. We wrote 20 scenarios, five in each of four domains: coding, explanation, emotional advice and brainstorming. The emotional-advice intents are adapted from the request categories in Li et al. (2025). Each scenario has a one-line intent and a neutral user message $u^0_b$ with no personality markers, for example "How do I reverse a singly linked list in Python?". Concrete details in $u^0_b$, such as the programming language or the user's nationality, pin down the request so that rewrites cannot drift in content.

### 3.2 Fixed responses

Every row of a scenario $b$ shares one assistant response $r_b$. We generated $r_b$ with Qwen2.5-7B-Instruct itself, by greedy decoding from the neutral system prompt "You are a helpful assistant." and $u^0_b$. We store the full response, capped at 1,024 tokens. Responses range from 90 to 1,024 tokens, with a median of about 630. One response reached the cap and was cut at its last sentence boundary.

Attention is causal, so the activation at response token $t$ depends only on the tokens before it. Averaging over the first $N$ tokens of the full teacher-forced response is therefore identical to forcing an $N$-token response. Storing the full response lets the readout length be chosen at analysis time; we use $N = 150$. A separate 150-token generation matched the first 150 tokens of the full generation for all 20 scenarios, which confirms that decoding is deterministic.

### 3.3 User-style variants

A user variant rewrites $u^0_b$ in one pole of one trait while asking for exactly the same thing. For each scenario, trait and pole we generated three paraphrases $k \in \{0, 1, 2\}$, which gives $20 \times 2 \times 2 \times 3 = 240$ variants. Each paraphrase leads with one facet of the trait (Table 1). The low pole uses the low end of the same facet, so paraphrase $k$ of the two poles forms a facet-matched pair.

**Table 1.** The facet each paraphrase leads with. The low pole uses the low end of the same facet.

| $k$ | Extraversion | Agreeableness |
|---|---|---|
| 0 | Sociability | Compassion |
| 1 | Assertiveness | Respectfulness |
| 2 | Energy | Trust |

A generator model, DeepSeek-V4.1-Flash, wrote all three paraphrases of a cell in one call. Its prompt gave the pole's definition and text cues, the facet for each paraphrase, and four out-of-domain examples written as full rewrites. The prompt forbids adding facts, adding constraints on the answer, or keeping the original wording and appending a styled sentence.

Each variant then passed three checks:

1. **Automatic checks.** The rewrite must be 0.5 to 2 times the length of $u^0_b$ (or at most 25 words longer, for very short messages). It must not contain $u^0_b$ word for word, and it must not be a near-duplicate of $u^0_b$ or of another paraphrase.
2. **Checklist.** A judge model, GPT-6 Luna, answers nine yes/no questions with a one-sentence reason each. Four check content: the same request, every concrete detail kept, no new facts, and no new requirements on the answer. Two check style boundaries: the other trait unchanged, and no added distress. Three check realism: plausible as a real message, no offensive language, and no self-answer or explicit naming of the trait. We use decomposed yes/no questions rather than a single 0–100 score because they are more reliable for LLM judges (Cook et al., 2024; Qin et al., 2024; Lee et al., 2024).
3. **Trait check.** The judge sees $u^0_b$ and the rewrite side by side and picks the writer who is more extraverted (or more agreeable). The question is asked twice with the order swapped, to cancel position bias (Zheng et al., 2023). A high-pole rewrite passes if it is chosen both times; a low-pole rewrite passes if the original is chosen both times. We always ask from the high end of the trait because asking "which is more reserved?" let the judge pick the shorter message: a plain ten-word original won against every low-Extraversion rewrite in a pilot.

A rewrite that failed any check was regenerated on its own, with the judge's reasons for every failed check as feedback, for up to five attempts. The generator and the judge are different models because LLM judges favour their own outputs (Panickssery et al., 2024).

All 240 variants passed: 143 on the first attempt, 82 on the second and 15 on the third to fifth. High-pole variants are longer than low-pole ones, at 35.9 against 24.7 tokens for Extraversion and 39.7 against 29.2 for Agreeableness. We record each variant's length so that analyses can control for it.

### 3.4 System-prompt twins

A system twin keeps $u^0_b$ and $r_b$ and moves the personality into the system prompt. We wrote five system prompts per trait and pole in the format of Chen et al. (2025) and Izawa et al. (2026), "You are a [X] assistant. [instruction]", with each high-pole prompt written in parallel to its low-pole counterpart. This gives $20 \times 2 \times 2 \times 5 = 400$ twins, plus 20 neutral rows with the neutral system prompt.

We checked that the prompts change Qwen's behaviour as intended. On four scenarios, one per domain, a judge compared each persona response with the neutral response. The low-Agreeableness prompts made Qwen withhold help rather than only change its tone ("Use a recursive approach. Google it."); only 13 of 20 low-Agreeableness responses still helped. User-side low-Agreeableness messages still ask sincerely for help, so this would have confounded the comparison between the two sources. We added "Still help the user fully with their request." to all ten Agreeableness prompts. Afterwards, 14 of 15 low-Agreeableness responses to task requests helped. On emotional-advice requests, low-Agreeableness responses remain dismissive, since withholding sympathy is close to what low compassion means there. The same sentence weakened the low-Extraversion prompts (16 of 20 responses read as reserved, against 19 of 20 without it), so the Extraversion prompts do not carry it.

### 3.5 Factorial set

The factorial set crosses the system prompt {+, −, none} with the user message {+, −} for each trait. System prompt $k$ is paired with user paraphrase $k$, which gives $20 \times 3 \times 2 \times 3 = 360$ rows per trait.

## 4 Method

### 4.1 Model and activation readouts

We study Qwen2.5-7B-Instruct in bf16, with 28 layers of 28 query heads each and a head dimension of 128. Izawa et al. (2026) report three style modulation heads (SMHs) for this model: heads 2, 4 and 27 of layer 19, zero-indexed. (Their paper numbers these layer 20, heads 3, 5 and 28.)

For head $i$ in layer $\ell$, we record the slice $z_{\ell,i} \in \mathbb{R}^{128}$ of the input to the output projection $W^O_\ell$. We read it at three positions:

- **first:** the last token of the prompt, which predicts the first response token. We take it from a prompt-only forward pass, so the forced response cannot affect it.
- **resp:** the mean over the first 150 tokens of the teacher-forced $r_b$. This is the readout Izawa et al. (2026) use.
- **user:** the mean over the tokens of the user message.

We tokenize the chat-template prefix and the response separately and concatenate the token IDs, so no token merges across the boundary. Activations are averaged in fp32. To check our implementation, we ran exactly two inputs through both our code and the original implementation of Izawa et al. (2026): one scenario's high-Extraversion system twin and its low-Extraversion counterpart. For this check our code followed their conventions, tokenizing prompt and response jointly and averaging over the full response. The recorded activations were identical, and the head scores agreed with a Spearman correlation of 1.0 and a maximum absolute difference of $1.3 \times 10^{-5}$. This confirms that our hooks and scoring compute the same quantities as theirs. It is not a reproduction of their experiments.

### 4.2 Head localization

For one trait and one source of personality, a pair is two rows that differ only in pole. Assigned contrasts pair system twins (100 pairs per trait); user contrasts pair user variants (60 pairs per trait). For each scenario $b$, we average the pair differences $z_+ - z_-$ over its paraphrases to get $d_b$, and take the mean over scenarios, $\delta = \frac{1}{B}\sum_b d_b$. The contribution score of head $i$ in layer $\ell$ follows Izawa et al. (2026):

$$s_{\ell,i} = \Big\langle W^O_{\ell,i}\,\delta_{\ell,i},\ \sum_j W^O_{\ell,j}\,\delta_{\ell,j} \Big\rangle,$$

the projection of the head's write onto the layer's total write. The scores of a layer sum to the squared norm of the layer's total write, so they decompose a layer, and we rank heads within their layer. For significance, we flip the signs of whole scenarios $d_b$ (19,999 permutations, one-sided) and correct across all 784 heads with the Benjamini–Hochberg procedure (Benjamini & Hochberg, 1995). Paraphrases are repeated observations of a scenario, so we resample scenarios, not paraphrases.

To compare the two sources, we report the Spearman correlation of head scores over all 784 heads, and the SMH ranks under each source. We also compute the cosine between the assigned and user contrasts $\delta^A$ and $\delta^U$ restricted to the three SMHs, a 384-dimensional vector, and compare it with 1,000 random three-head groups from layer 19.

As a control for the significance test, we built same-pole contrasts that pair two different system prompts of the same pole ($k = 0$ against $k = 1$). These differ in wording but not in trait.

### 4.3 Behavioural mirroring

We sampled Qwen's response to every user variant twice and to every neutral message three times, with the neutral system prompt, temperature 1.0 and at most 300 new tokens. This gives 540 responses. Following Izawa et al. (2026), GPT-4.1-mini scores each response from 0 to 100 for Extraversion and for Agreeableness, as the probability-weighted mean over its top-20 integer tokens. The judge sees only the response and the scenario's neutral intent, never the user's message, so it cannot score the user's style instead of the model's. It also scores coherence and refusal.

For trait $t$, the mirroring effect is

$$M_t = \frac{1}{B}\sum_{b=1}^{B}\left(\bar y_{b,+} - \bar y_{b,-}\right),$$

where $\bar y_{b,\pm}$ is the mean judged score of trait $t$ over scenario $b$'s responses to the high- or low-pole user variants. We test $M_t$ with a Wilcoxon signed-rank test over the 20 scenarios and report Cohen's $d_z$ and a 95% bootstrap confidence interval from 5,000 scenario resamples.

### 4.4 Factorial regression

For a three-head group $G$ in layer 19, $Z_G \in \mathbb{R}^{n \times 384}$ stacks the group's activations over the factorial rows of one trait. We code the system prompt with two orthogonal contrasts: pole, $c_1 \in \{+1, -1, 0\}$ for {+, −, none}, and presence, $c_2 \in \{1, 1, -2\}$. We code the user message as $c_U = \pm 1$ and add the interactions $c_1 c_U$ and $c_2 c_U$. After centering $Z_G$ and the codes within each scenario, we fit a multi-output least-squares regression. We report the variance explained by the assigned block ($c_1, c_2$), the user block ($c_U$) and the interaction block. We also report the signed cosine between the coefficient vectors of $c_1$ and $c_U$, with a 95% CI from 1,000 scenario bootstraps.

We test whether the user coefficient is nonzero by shuffling $c_U$ within each scenario and system-prompt cell (1,000 shuffles). We do not use this shuffle as a null for the cosine: under shuffling the coefficient keeps pointing along the true user direction with random sign, so the null distribution sits near the observed value. We fit the regression for the SMHs, the three heads of layer 19 with the largest mean norm on neutral inputs, and five random triples. We ran it for Extraversion only.

### 4.5 Ablation

To test whether the SMHs carry mirroring, we zeroed their slices of the output-projection input at every position during generation and repeated the mirroring measurement. As comparisons, we zeroed two random three-head groups from layer 19 instead.

## 5 Results

### 5.1 The style heads respond to both sources

On the response readout, the three SMHs are the top three heads of layer 19 for both assigned traits, and remain in the top four for user style (Table 2). Over all 784 heads, assigned and user head scores correlate at a Spearman $\rho$ of about 0.78. At the first token, the user contrast keeps only head 27 near the top, and the correlation falls to 0.56–0.62.

**Table 2.** Ranks of the three SMHs among the 28 heads of layer 19 (1 = largest contribution), as heads 2 / 4 / 27.

| Readout | Source | Extraversion | Agreeableness |
|---|---|---|---|
| resp | assigned | 1 / 3 / 2 | 1 / 3 / 2 |
| resp | user | 1 / 2 / 4 | 1 / 2 / 3 |
| first | assigned | 2 / 3 / 1 | 2 / 3 / 1 |
| first | user | 14 / 10 / 2 | 14 / 13 / 3 |

The significance test does not separate heads. Between 669 and 722 of the 784 heads pass $q < 0.05$ in every localization we ran. The same-pole contrasts, which change only the wording of the system prompt, mark 691 to 707 heads. The test therefore detects any consistent change in the input, and we interpret ranks and effect sizes instead. We have not yet measured the SMH ranks under the same-pole contrasts. Without that number, the ranks in Table 2 show that the SMHs respond strongly to these system prompts, but not that they respond to personality rather than to wording.

The two sources move the SMHs in only partly shared directions. Inside the SMHs, the cosine between the assigned and user contrasts is 0.42 and 0.55 at the first token (Extraversion and Agreeableness), at the 85th and 92nd percentiles of random three-head groups. On the response readout it falls to 0.23 and 0.38, at the 57th and 71st percentiles, close to the mean absolute cosine of a random group (0.24 and 0.34). The SMHs are among the most responsive heads for both sources, but they do not carry an unusually shared direction.

### 5.2 Qwen mirrors extraversion but not agreeableness

Qwen's responses are clearly more extraverted when the user writes in an extraverted style (Table 3). The effect is one-sided: responses to introverted users score 36.0, close to the 33.7 for neutral users, while responses to extraverted users score 49.0. Agreeableness does not move. Responses score about 91 for every user pole, including the neutral one, so they sit near the ceiling of the scale, and the test cannot detect a smaller change. No response was a refusal, and responses average about 290 tokens for every pole.

**Table 3.** Judged trait of Qwen's responses (0–100), by the style of the user's message. $M$ is the high-minus-low mirroring effect.

| Trait | Neutral | Low pole | High pole | $M$ [95% CI] | $d_z$ | Wilcoxon $p$ |
|---|---|---|---|---|---|---|
| Extraversion | 33.7 | 36.0 | 49.0 | 13.0 [9.0, 17.1] | 1.39 | $1.9 \times 10^{-5}$ |
| Agreeableness | 90.8 | 90.9 | 91.4 | 0.5 [−0.2, 1.2] | 0.31 | 0.097 |

Extraversion mirroring depends on the facet the user expresses. Sociable messages ($k = 0$) raise the effect to 18.5 points ($p = 2 \times 10^{-4}$) and energetic messages ($k = 2$) to 22.5 points ($p = 1.9 \times 10^{-6}$). Assertive messages ($k = 1$) produce no effect (−1.8, $p = 0.67$). The effect also depends on the domain: it is 14.0 to 19.4 points for coding, explanation and brainstorming, and 2.6 points for emotional advice. With five scenarios per domain, the domain differences are descriptive.

### 5.3 The system prompt dominates the style heads

In the factorial regression for Extraversion, the system prompt explains far more of the SMH activations than the user's style (Table 4). On the response readout, the assigned block explains 69% of the variance and the user block 0.6%. The user coefficient is still reliably nonzero ($p \le 0.003$ in every head group, against a minimum of 0.001 with 1,000 shuffles), and it points partly in the same direction as the assigned pole coefficient (cosine 0.20, CI [0.16, 0.23]).

**Table 4.** Factorial regression for Extraversion. $R^2_A$ and $R^2_U$ are the variance explained by the assigned and user blocks. The cosine is between the assigned-pole and user coefficients. The random range spans the five random triples.

| Readout | Group | $R^2_A$ | $R^2_U$ | Cosine [95% CI] |
|---|---|---|---|---|
| resp | SMHs | 0.69 | 0.006 | 0.20 [0.16, 0.23] |
| resp | random triples | 0.22–0.71 | 0.007–0.029 | 0.00–0.29 |
| first | SMHs | 0.72 | 0.003 | 0.27 [0.22, 0.31] |
| first | random triples | 0.10–0.70 | 0.008–0.125 | 0.07–0.56 |

These numbers do not single out the SMHs. Several random triples show a similar or larger cosine, and the SMHs have the smallest user $R^2$ of any group. One head drives the large assigned $R^2$: every group that contains head 2 explains 65–72% of the variance with the assigned block, and groups without it explain 10–40%. Two of the comparison groups, the highest-norm triple (heads 0, 2 and 17) and one random triple (heads 0, 1 and 2), contain head 2, so they are not independent of the SMHs. The assigned block also combines two effects: the pole of the system prompt, and whether a persona prompt is present at all. We have not separated them.

### 5.4 Ablating the style heads does not reduce mirroring

Zeroing the SMHs during generation leaves Extraversion mirroring unchanged (Table 5). The effect is 13.7 points with the SMHs ablated, against 13.0 without ablation and 12.9 and 14.4 with random triples ablated, and the confidence intervals overlap almost entirely. Agreeableness mirroring stays within one point of zero in every condition. One random triple (heads 0, 1 and 2) contains head 2, so it is not a clean control. Because the SMH ablation itself produced no reduction, this does not change the conclusion.

**Table 5.** Mirroring effect $M$ (points) with three heads of layer 19 zeroed.

| Heads zeroed | Extraversion $M$ [95% CI] | Agreeableness $M$ |
|---|---|---|
| none | 13.0 [9.0, 17.1] | 0.5 |
| SMHs (2, 4, 27) | 13.7 [10.2, 17.1] | 0.2 |
| random (14, 17, 22) | 12.9 [8.4, 17.7] | 0.2 |
| random (0, 1, 2) | 14.4 [9.0, 19.7] | 0.4 |

### 5.5 Answers to the research questions so far

All results so far come from single-turn conversations. Prior work observed persona drift building up over multi-turn conversations (Lu et al., 2026; Sprejer et al., 2026), so we treat these answers as preliminary until the multi-turn experiments in Section 6 are complete.

**RQ1: Is user-induced persona drift driven by a specific, localized computation, or is it a generalized consequence of processing stylized context?** Not yet answered. Single-turn mirroring exists for Extraversion (Section 5.2). The heads that respond most to user style are the style modulation heads (Section 5.1), but zeroing them leaves mirroring unchanged (Section 5.4). We have therefore ruled out one candidate location without finding the computation itself.

**RQ2: Do user-induced mirroring and assigned persona share the style modulation heads, or do they use different causal pathways?** Partly answered. Both sources engage the same heads: the SMHs rank among the top four of layer 19 for both, and head scores correlate at $\rho \approx 0.78$ across all heads. In a single turn, however, the SMHs are not necessary for mirroring. The evidence so far favours shared responsive heads with a different causal pathway for mirroring. Two checks remain before we can claim this: SMH ranks under wording-only contrasts, and the ablation in multi-turn conversations.

**RQ3: Are the two components geometrically separable within this circuit, and can that separability suppress user-induced drift while preserving the assigned persona?** The first part is partly answered. Inside the SMHs, the assigned and user contrasts are distinct but correlated: their cosine is 0.20–0.55 depending on trait and readout, and no higher than for random head groups. The user component is also small next to the assigned one ($R^2$ of 0.006 against 0.69). The second part has not been tested. Because the SMHs are not necessary for single-turn mirroring, a projection applied at the SMHs is unlikely to suppress it. We will choose the intervention site after locating the mechanism (Section 6).

## 6 Future Work and Timeline

The project ends in late October 2026. The work below is ordered by dependency. The first two items extend the evidence base that every later conclusion rests on.

### 6.1 Five traits, two models and a sharper agreeableness measure

We will extend the dataset to all five Big Five traits by adding Openness, Conscientiousness and Neuroticism. Each needs pole definitions from the BFI-2 facets, rewrite examples and system prompts; the generation and quality-check pipeline carries over unchanged. We will also double the number of scenarios to 40, so that per-domain and per-facet effects can be tested rather than only described. All experiments will be repeated on Llama-3.1-8B-Instruct, whose reported style heads are layer 13, heads 23, 29 and 31 (zero-indexed).

The agreeableness result shows that absolute 0–100 ratings fail when a model sits near the top of the scale. We will replace them with pairwise judgements: the judge sees two responses to the same request, one to a high-pole user and one to a low-pole user, and picks the more agreeable one in both orders. This is the same protocol we use to check the user variants (Section 3.3). We will also count explicit markers, such as politeness formulas and hedges, as a judge-free measure.

### 6.2 Multi-turn mirroring

Prior work found persona drift accumulating over multi-turn conversations, while all our results are single-turn. We will build conversations of five to ten turns in which the user keeps a consistent style, with and without an assigned persona, and track the judged trait of each response. At each turn we will also record the readouts of Section 4.1. We will repeat the ablation of Section 4.5 on these conversations, since the style heads may become necessary only once the user's style has built up over several turns.

### 6.3 Is mirroring asymmetric, and does alignment cause it?

Qwen mirrors enthusiasm but not reservedness, and shows no measurable shift towards rudeness. If this pattern holds across five traits and two models, assistants mirror only some poles of a trait. One explanation is that preference training keeps the assistant persona away from negative traits. We will test this by comparing Qwen2.5-7B with Qwen2.5-7B-Instruct on the same stimuli. If the base model mirrors both poles and the instruction-tuned model only one, alignment training is the likely source of the asymmetry.

### 6.4 Does an assigned persona resist the user?

The factorial set (Section 3.5) has so far been used only for activation analysis. We will also generate and judge responses for it, which measures how well an assigned persona holds against a user writing in the opposite style. For example, it tells us whether an assistant told to be introverted stays introverted with an enthusiastic user. This is the behavioural quantity an intervention must preserve.

### 6.5 Locating the mirroring mechanism

The ablation of Section 5.4 tells us where single-turn mirroring is not computed. To find where it is, we will follow the style information from the user's message into the response:

- **Attention knockout.** Mirroring requires the response to read style from the user's tokens. We will block attention from response positions to user positions, one head or layer at a time, and measure the drop in mirroring.
- **Activation patching.** We will copy activations from a run with a high-pole user into a run with a low-pole user, one component at a time, and measure the change in the response. A linear probe trained on judged responses will score the change, so that each patch does not require a judge call.
- **Adoption or perception.** Within one user pole, we will test whether the strength of the user direction in the activations predicts the judged trait of the response. A positive relationship means the model adopts the style. No relationship means the model only represents the user's style. Sprejer et al. (2026) leave this question open.

### 6.6 Intervening at the mechanism

Once the mechanism is located, we will remove the user component there, by projecting out the estimated user direction at the identified heads or layers during generation. We will compare this with steering the residual stream and with an instruction not to adapt to the user's style. Each intervention is scored on three outcomes: the reduction in mirroring, the preservation of the assigned persona (Section 6.4) and the coherence of the response. If no localized mechanism emerges, residual-stream steering becomes the main intervention, and RQ1 is answered in favour of a distributed computation.

### 6.7 Timeline

| Period | Work |
|---|---|
| Oct 5–11 | Pairwise agreeableness measure; definitions, prompts and data for the three new traits; scenarios expanded to 40 |
| Oct 12–18 | Five traits on Qwen; Llama-3.1-8B on all traits; base vs. instruction-tuned comparison; behavioural factorial |
| Oct 19–25 | Multi-turn conversations, including the multi-turn ablation; attention knockout |
| Oct 26–31 | Activation patching and adoption vs. perception; intervention at the located mechanism; final report |
