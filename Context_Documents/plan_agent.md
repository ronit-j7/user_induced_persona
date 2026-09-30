# AttentionSeekers: Implementation Plan for Coding Agents

> **Status: SUGGESTION, not a spec.** The team may change scope, traits, models, or methods. When something here conflicts with a newer instruction from the team, the team wins. When something is ambiguous, **ask instead of guessing**, especially for anything listed under *Open decisions*.
>
> You are assumed to have read the project's **interim document** and the **full proposal** ("Causal Disentanglement of User-Induced and Assigned Persona in Language Models"). This file does not repeat their motivation. It gives (1) what we think is wrong or underspecified in the current design, (2) the direction we recommend, and (3) concrete specs, interfaces, and acceptance tests.
>
> A reference implementation of the analysis maths and hooks lives in `persona_pipeline.py`. Its numpy analysis functions are unit-tested (`python persona_pipeline.py --selftest`). **Its torch hook and generation code has NOT been executed on a GPU.** Treat it as a starting point and verify positions and shapes first.

---

## 1. Current scope (mid-submission)

- **Model:** `Qwen/Qwen2.5-7B-Instruct` only. Llama-3.1-8B-Instruct comes later, so write model-agnostic code with a config dict.
- **Traits:** Extraversion (E) and Agreeableness (A), each with a high (+) and low (−) pole. Design for 5 traits.
- **Scope decision (2026-10-01):** do **not** run the original humorous reproduction or its judged-data replay. Replace that gate with implementation agreement against original reference functions on identical project inputs, plus independent E/A assigned-persona localization. Original-paper reproduction is out of scope, not a remaining WS2 blocker. This does not remove WS1 QC or WS3 behavioral judging.
- **Priorities, in order:**
  1. Validate extraction against the reference implementation and localize assigned E/A persona heads.
  2. Behavioral mirroring check.
  3. Assigned vs. user head localization.
  4. Small factorial regression (stretch).
  5. SMH ablation (stretch).
  6. Projection intervention (post mid-sub).

---

## 2. Issues we found in the current proposal

These are ordered by how much they affect results. Each one comes with a suggested fix.

### 2.1 Teacher-forcing is inert at the proposed readout position (critical)
The proposal teacher-forces a fixed assistant response, then reads activations "at the first assistant token, immediately following the final user token." Under causal attention, $h_t = f(x_{\le t})$. The activation at the last generation-prompt token cannot depend on response tokens that come after it, so teacher-forcing has no effect on the measured quantity.

**Fix:** record three readouts per forward pass (Section 6.2):
- `first`: the last token of the generation prompt.
- `resp`: the mean over teacher-forced response tokens. This matches Izawa et al.'s persona-vector definition.
- `user`: the mean over user-content tokens.

### 2.2 "3 sampled completions" are not replicates for first-token activations (critical)
In the factorial stage, the proposal samples 3 completions per condition and counts 1,800 conversations per model. Sampling happens after the first-token position, so the 3 samples give **identical** `first` activations. Effective n is 600, not 1,800, and treating them as independent inflates significance.

**Fix:** use **paraphrases** of the user message (and of the system prompt) as the replicates for activation analyses. Sampled completions are only relevant for behavioral scores and for `resp` readouts on free generations.

### 2.3 Head indexing: the paper is 1-indexed, the code is 0-indexed (critical, silent)
Izawa et al. report Qwen SMHs as "layer 20, heads 3/5/28" and Llama as "layer 14, heads 24/30/32." Their repo (`github.com/Omusubi0123/style-modulation-head`) is 0-indexed throughout. Its README says `--layer 19` means the 20th block, and its example steers Qwen with `--layer 19 --head_indices 2,4,27`. Llama has 32 heads, so a head numbered 32 can only be 1-indexed.

**Use:**
- Qwen: `model.model.layers[19]`, heads `[2, 4, 27]`
- Llama: `model.model.layers[13]`, heads `[23, 29, 31]`

**Treat these as published reference candidates.** In Exp 0, verify implementation behavior and report their empirical ranks on our E/A twins; do not assume the published heads must dominate every new contrast or claim original-paper reproduction.

### 2.4 Selection bias / circularity in the SMH comparison (important)
The SMHs were selected using **system-prompt** persona contrasts, on six personas (evil, sycophancy, hallucination, humorous, passionate, loser), with **model-generated responses that differ across conditions**. Comparing "assigned" and "user" subspaces only inside heads chosen for assigned persona biases the comparison toward assigned persona. Also, "SMHs don't respond to user style" is confounded with "SMHs don't encode Big Five traits."

**Fix:**
- (a) Rerun localization for Big Five traits.
- (b) Localize heads **independently** for the user-style contrast, using the same head contribution score.
- (c) Use a **symmetric twin design** where the assigned and user contrasts share the same scenarios and the same forced response, and differ only in where the style comes from.

### 2.5 Judge contamination in behavioral scoring (important)
If the judge sees the user message when scoring the assistant's trait, it will partly score the user's style.

**Fix:** the judge sees **only** the assistant response plus a neutral one-line description of the intent.

### 2.6 Regression is underspecified (moderate)
- There are no scenario fixed effects. Content varies across scenarios and will dominate $Z$. **Fix:** center $Z$ and $X$ within scenario.
- The factor coding is unspecified. **Fix:** use orthogonal contrast codes (Section 7.4) so the $R^2$ contributions are additive in a balanced design.
- "Principal angles between subspaces" collapses to a single cosine per trait, since each trait contributes one direction. The **sign** of that cosine is the most informative quantity (positive = mirroring-aligned), and principal angles discard it. **Fix:** report the signed per-trait cosine first, and principal angles across traits second.
- MANOVA is infeasible because $d = 384$ exceeds n per cell. **Fix:** use permutation tests on coefficient norms.

### 2.7 An invalid null for the cosine (moderate; easy to get wrong)
Shuffling user labels within cells is **not** a valid null for $\cos(\beta_A, \beta_U)$. Under shuffling, $\beta_U$ still points along the true user direction with random sign and scale, so the null distribution concentrates near $\pm$ the true cosine. We hit this while testing the reference code.

**Fix:**
- Use shuffling only to test whether $\|\beta_U\|$ differs from zero.
- For the cosine, use a scenario-bootstrap CI, a split-half reliability noise ceiling (report disattenuated values), and the random-direction baseline $\mathbb{E}|\cos| \approx \sqrt{2/(\pi d)}$.

### 2.8 Tokenization boundary (minor, silent)
Izawa's repo tokenizes `prompt + response` as a single string and counts prompt tokens separately. A merge across the boundary shifts positions by one.

**Fix:** tokenize the prefix (via `apply_chat_template(..., add_generation_prompt=True)`) and the response (`add_special_tokens=False`) separately, and concatenate the **ids**.

### 2.9 Default system prompt injection (minor, silent)
The Qwen2.5 chat template injects "You are Qwen, created by Alibaba Cloud. You are a helpful assistant." when no system message is given.

**Fix:** always pass an explicit system message. The neutral one is `"You are a helpful assistant."`

### 2.10 Scope mismatch with the cited drift literature (framing)
Lu et al. (Assistant Axis) define persona drift as departure from the **default Assistant persona**, driven by meta-reflection and emotionally vulnerable users. It is not the degradation of a custom system-prompt persona. Our design is single-turn.

**Fix:** frame the results as **single-turn personality mirroring**, with multi-turn drift as an extension.

### 2.11 Trait-pole confounds (watch)
- Low Agreeableness ("blunt, demanding") overlaps with rudeness and may trigger refusal or safety behavior.
- High Neuroticism in the emotional-advice domain resembles Lu's drift trigger.
- User-message **length** varies with pole.

Log refusals and length, and include `log(user_len_tokens)` as a nuisance covariate in robustness checks.

### 2.12 Citation/writing issues (for the human writers, not agents)
- Wrong title and authors for Lu et al.
- Cite Izawa et al. as ICML 2026 (arXiv 2603.13249).
- Subramani 2022 is misattributed for "residual steering hurts coherence."
- The Beckmann & Butlin sentence is ambiguous.
- BibTeX author fields are broken ("et al. Izawa").

---

## 3. Recommended direction

**Core idea:** treat user style and assigned persona as two sources of the **same stylistic variable**. Measure where each enters the network and in what direction, using a **symmetric** design, before attempting any intervention.

**Hypotheses, pre-registered informally:**

- **H1 (localization):** user-style contrasts produce significant head contribution scores concentrated in a small set of heads.
- **H2 (shared substrate):** those heads overlap with the assigned-persona SMHs, measured by Spearman $\rho$ over all heads and top-k Jaccard.
- **H3 (alignment):** within the SMHs, $\cos(\beta_A^{(t)}, \beta_U^{(t)}) > 0$, meaning user style pushes the heads in the same direction as the equivalent system prompt. This is the mechanistic signature of mirroring.
- **H4 (adoption vs. perception):** the per-sample projection onto $\hat u$ predicts the judged trait of free generations within a user pole.
- **H5 (causal):** projecting out $\hat u$ at the SMHs reduces mirroring more than the same projection at control heads, at comparable coherence cost.

**Interpretation guide:**
- H2 and H3 true means shared substrate and aligned direction. Separability (RQ3) would then be *hard*, and projection would also damage the assigned persona. That is a legitimate finding.
- H1 true but H2 false means a separate pathway. Redirect RQ3 to the user-style heads.
- If the behavioral mirroring effect is near zero, report that and consider multi-turn stimuli.

---

## 4. Suggested repo layout

```
attentionseekers/
  configs/
    qwen.yaml            # model facts + SMH indices (0-indexed) + paths
    traits.yaml          # trait definitions, pole descriptions, judge rubrics
  data/
    scenarios.jsonl      # 20 base scenarios
    user_variants.jsonl  # Stage-1 user-style items
    sys_twins.jsonl      # assigned-persona twin items
    factorial.jsonl      # A x U items
    qc/                  # judge outputs, rejected items, manual-check sheet
  src/
    config.py
    chat.py              # encode(): token-exact positions
    extract.py           # capture_heads(): writes acts/*.npy
    heads.py             # contrasts, head scores, permutation, BH
    factorial.py         # regression, cosines, bootstrap, reliability
    generate.py          # free generation (+ optional hooks)
    judge.py             # logprob-weighted 0-100 judge, caching
    intervene.py         # projection / ablation hooks
    plots.py
  scripts/               # one entry point per experiment: exp0_validate.py ... exp5_intervene.py
  results/<exp>/<run_id>/  # config snapshot, metrics.json, figures, logs
  tests/
  persona_pipeline.py    # reference (port pieces into src/)
```

**Conventions:**
- Every run writes `results/<exp>/<run_id>/config.json` (git hash, seed, model, dtype, data file hashes).
- Seeds are fixed (`seed=0` default).
- The judge is cached by `sha256(prompt)` on disk. Never pay twice.

---

## 5. Environment facts (verified from Izawa et al.'s repo and model configs)

| | Qwen2.5-7B-Instruct | Llama-3.1-8B-Instruct |
|---|---|---|
| layers / query heads / KV heads | 28 / 28 / 4 (GQA) | 32 / 32 / 8 (GQA) |
| head_dim / d_model | 128 / 3584 | 128 / 4096 |
| SMH (0-indexed) | layer 19, heads [2, 4, 27] | layer 13, heads [23, 29, 31] |

- **Hook point:** `model.model.layers[l].self_attn.o_proj`, `register_forward_pre_hook`. The input `args[0]` has shape `[B, T, H*head_dim]`; head `i` is the slice `[i*128:(i+1)*128]`. With GQA, the pre-`o_proj` tensor is still per **query** head.
- **`o_proj` has no bias** in both models, so head writes are exactly linear. Per-head write matrix: `W_O[l][:, i*128:(i+1)*128]`, shape `[d_model, 128]`.
- **Precision:** load in bf16, cast captured activations to fp32 before any averaging, store as fp16 `.npy`.
- **Memory:** full per-head `W_O` for Qwen in fp32 is about 1.4 GB. Load only the layers you need if tight.
- **Judge in Izawa's repo:** `gpt-4.1-mini-2025-04-14`, 0–100 scale, logprob-weighted over the top-20 tokens. Reuse it for comparability unless the team decides otherwise.

---

## 6. Data and extraction specs

### 6.1 JSONL schema (one row per model input)

```json
{
  "id": "code_03|U|E|+|k1",
  "scenario": "code_03", "domain": "coding",
  "set": "user_variant | sys_twin | factorial | neutral",
  "trait": "E", "user_pole": "+|-|0", "sys_pole": "+|-|0",
  "user_paraphrase": 1, "sys_paraphrase": 0,
  "system": "You are a helpful assistant.",
  "user": "...",
  "forced_response": "...",           // null for free-generation rows
  "qc": {"content": 92.1, "trait": 88.4, "passed": true, "attempts": 1},
  "user_len_tokens": 57
}
```

`id` must be unique and deterministic. Analyses join on `(scenario, trait, user_paraphrase, sys_paraphrase)`.

### 6.2 Construction procedure

1. **Scenarios.** 20 total: 4 domains (coding, explaining, emotional advice, brainstorming) × 5. Each has an `intent` and a neutral `user` message ($u^0_b$). Humans or an LLM draft them, and humans approve.
2. **Forced response $r_b$.** Qwen generates it from `(system=s0, user=u0_b)`. Use greedy decoding, `max_new_tokens=150`, truncate at a sentence boundary, and store the text. The same $r_b$ is reused for **every** row of scenario $b$ in the user-variant and sys-twin sets.
3. **User variants.** For trait $t$, pole $p$, and paraphrase $k \in \{0, 1, 2\}$: an LLM rewrites $u^0_b$ in pole-$p$ style, preserving the intent exactly. Prompt the rewriter with the pole description from `traits.yaml`.
4. **Sys twins.** User = $u^0_b$. System = one of 5 paraphrased pole-$p$ persona prompts per trait, in Izawa's format ("You are a [X] assistant. [instruction]"). Also include neutral rows (`sys_pole=0`, `user_pole=0`).
5. **Factorial.** One trait first: sys ∈ {+, −, 0} × user ∈ {+, −} × 20 scenarios × 3 user paraphrases = 360 rows. The `forced_response` is null for free generation, and $r_b$ is used for the teacher-forced `resp` readout.
6. **QC.** The judge scores `content` (same intent as $u^0_b$?) and `trait` (clear pole-$p$ expression?) on 0–100. Pass at ≥ 70 on both, retry up to 3 times with judge feedback, then drop. Export a random 10% to `qc/manual_check.csv` for humans.

**Checks to run after building:**
- Balance per cell.
- Every scenario has all poles and paraphrases.
- No duplicate user texts.
- Report the distribution of `user_len_tokens` per pole.

### 6.3 Encoding (`chat.encode`)
- Prefix ids: `apply_chat_template([sys, user], tokenize=True, add_generation_prompt=True)`.
- `t0 = len(prefix) - 1`.
- Response ids: `tokenizer(resp, add_special_tokens=False)`, appended to the prefix. `resp_span = [len(prefix), len(prefix)+len(r))`, which excludes the end-of-turn token.
- `user_span`: locate by diffing against the template rendered with empty user content (approximate is fine). Assert that decoding the span returns the user text, allowing for whitespace.

**Acceptance test:** for 3 random rows, print `decode(ids[t0])`, `decode(ids[resp_span])[:40]`, and `decode(ids[user_span])[:40]`. For Qwen, `t0` should decode to the newline after `<|im_start|>assistant`.

### 6.4 Extraction (`extract.capture_heads`)
- Batch size 1. Put pre-hooks on all layers' `o_proj`. Store `first`, `resp`, and `user` as `[N, L, H, 128]` fp16 arrays plus an index file mapping row → `id`.
- Output: `acts/<set>/<readout>.npy`, `acts/<set>/index.jsonl`.
- Rough cost for Qwen: about 200 KB per row per readout, which is fine.

**Acceptance tests:**
- `resp` readouts differ when `forced_response` differs.
- `first` readouts are **bit-identical** for two rows that share (system, user) but differ in `forced_response`. This demonstrates issue 2.1 and guards against position bugs.

---

## 7. Experiments

### Exp 0: Implementation validation and assigned-persona localization (priority 1)
- **Excluded:** the original `humorous` generate–judge–filter experiment and replay on its judged examples. Do not generate those CSVs, make judge calls for that reproduction, or require saved upstream vectors to complete WS2.
- **Step 1.** Compare our extraction and head scoring with independently executed, unmodified original reference functions on identical project inputs using full Qwen. Match token IDs and the full-response averaging convention for this check. Record the sample IDs, model revision, reference source hashes, vector errors, and raw-score agreement; do not use a self-comparison.
- **Validation criterion:** captured vectors agree within recorded numerical tolerances and raw head-score Spearman correlation is $\rho > 0.95$. State the tested sample size rather than implying validation on the authors' full dataset.
- **Step 2.** Run independent localization on our **sys_twins** (E, A) with the `resp` readout. Score all 784 heads and report the ranks of published candidates {2, 4, 27} in `layers[19]`, whether they appear in the top-3/top-10, significance results, and layer comparisons. Report the measured outcome even if those candidates do not survive.
- **Step 3.** Define and save layer-19 controls: five seeded random 3-head groups and the highest-norm 3 heads selected from neutral `first` activations.
- **Output:** an implementation-agreement report, E/A assigned-persona heatmaps and metrics, saved controls, and reusable verified extraction code. Do not label the implementation check as original-paper reproduction.
- **Recorded status:** complete under this revised scope. Full-Qwen reference agreement on one E positive/negative pair matched vectors exactly, with score Spearman 1.0 and maximum score error approximately 1.34e-5. Assigned E/A response localization ranked heads 2, 27, and 4 first, second, and third respectively in layer 19; controls are saved. Evidence: `../Workstream_2/docs/full_qwen_verification_summary.json` and `../results/ws1-verification/full-qwen-gpu-retry/`.

### Exp 1: Assigned vs. user head localization (priority 3)
For trait $t$, readout $r \in$ {first, resp}, and source $X \in \{A, U\}$:

- **Paired differences** over matched (scenario, paraphrase) pairs: $D_p = z_{+} - z_{-}$, with $\delta^X = \operatorname{mean}_p D_p$, shape `[L, H, 128]`.
- **Head score:** $s_{\ell,i} = \langle W^O_{\ell,i}\delta_{\ell,i},\ \sum_j W^O_{\ell,j}\delta_{\ell,j}\rangle$.
  - **Unit test:** $\sum_i s_{\ell,i} = \|\sum_j W^O_{\ell,j}\delta_{\ell,j}\|^2$.
- **Normalized score:** $\operatorname{sign}(s)\log(1+|s|)$, then z-score within layer (Izawa's convention).
- **Layer SNR:** $\|W^O_\ell\delta_\ell\| / \sqrt{\mathbb{E}_n\|W^O_\ell(z_{n,\ell}-\bar z_\ell)\|^2}$.
- **Significance:** sign-flip permutation over pairs (1000 permutations), one-sided, BH-FDR at 0.05 over all $L \times H$ heads.
- **Comparison metrics:**
  - Spearman $\rho(s^A, s^U)$
  - top-3 and top-10 Jaccard
  - rank of each SMH in $s^U$
  - $\cos(\delta^A_G, \delta^U_G)$ for $G$ = SMH, vs. the percentile among 1000 random 3-head groups in layer 19, vs. the top-norm group (3 heads with the largest mean $\|o_{19,i,t_0}\|$ on neutral rows)
- **Output:** `metrics.json` plus heatmaps (A and U side by side) and SNR curves.

The A contrast comes from sys_twins (same user $u^0$, same $r_b$). The U contrast comes from user_variants (same system $s_0$, same $r_b$). This symmetry is the point, so don't mix in Izawa's generated-response vectors here.

### Exp 2: Behavioral mirroring (priority 2, parallelizable)
- **Generation:** free generation from (s0, user variant, paraphrase k=0). Temperature 1.0, 3 samples, `max_new_tokens=300`.
- **Judging:** the judge scores trait $t$ on the response **only**, with the neutral intent as context. It also scores coherence and flags refusals.
- **Metric:** $M_t = \operatorname{mean}_b(\bar y_{b,+} - \bar y_{b,-})$. Test with Wilcoxon signed-rank over 20 scenario pairs, and report $d_z$ and a bootstrap CI.
- **Also report:** the refusal rate per pole, and the mean response length per pole.

### Exp 3: Factorial regression (priority 4, stretch)
- **Data:** the factorial set for 1 trait, readouts `first` and `resp`. Head groups: SMH, top-norm, 5 random triples, each with $Z_G \in \mathbb{R}^{n\times384}$.
- **Codes** (orthogonal in a balanced design):

  | A level | $c_1$ (pole) | $c_2$ (present) |
  |---|---|---|
  | + | +1 | +1 |
  | − | −1 | +1 |
  | 0 | 0 | −2 |

  Plus $c_U = \pm1$ and the interactions $c_1c_U$ and $c_2c_U$.
- **Fit:** center $Z$ and $X$ within scenario, then multi-output OLS.
- **Report:**
  - Additive $R^2$ for the A, U, and A×U blocks, and the residual.
  - Signed $\cos(\beta_{A,\text{pole}}, \beta_U)$ with a scenario-bootstrap 95% CI (1000 resamples).
  - Split-half reliabilities $r_A$ and $r_U$ (Spearman–Brown corrected) and the disattenuated cosine.
  - The random baseline $\sqrt{2/(\pi\cdot384)} \approx 0.041$.
  - A shuffle test on $\|\beta_U\|$ and on $\|\beta_I\|_F$ (shuffle $c_U$ within scenario × A cells).
- **Multi-trait (later):** principal-angle cosines $\sigma(Q_A^\top Q_U)$.
- **Do not** use label shuffling as the null for the cosine (see 2.7).

### Exp 4: Behavior link, adoption vs. perception (post mid-sub or stretch)
- **Data:** the A=0 factorial rows with free generations.
- **Projection:** $p_n = \langle z_{n,G} - \bar z_{b(n),G},\ \hat u^{(-b)}\rangle$, with $\hat u$ estimated **leave-one-scenario-out**.
- **Analysis:** Spearman $\rho(p_n, y_n)$ within each user pole, pooled with Fisher $z$.

### Exp 5: Interventions (post mid-sub; SMH zero-ablation is an allowed stretch before)
- **Zero-ablation (stretch):** zero the SMH slices at all positions and rerun Exp 2. Compare against 5 random triples.
- **Projection:** at the `o_proj` pre-hook, for positions $\ge t_0$, apply $z_G \leftarrow z_G - \lambda QQ^\top(z_G-\mu_0)$ with $\lambda \in \{0.5, 1, 1.5\}$.
  - $\mu_0$ is the neutral-condition mean.
  - $Q$ is estimated from the `resp` readout on 15 train scenarios, and evaluated on 5 held-out scenarios.
  - The hook must track absolute position across KV-cache steps. See `make_projection_hook`, and call `.reset()` before each `generate`.
- **Arms:** SMH, 5 random triples, top-norm, and a random direction within the SMH subspace.
- **Metrics:** mirroring reduction $1 - M^{\text{int}}/M^{\text{base}}$, assigned-persona preservation under (s_{t,+}, u^0), and coherence.

---

## 8. Judge spec
- Model: `gpt-4.1-mini-2025-04-14` (pending team decision). `temperature=0`, `max_tokens=1`, `logprobs=True`, `top_logprobs=20`. The score is the probability-weighted mean over integer tokens 0–100. Return `None` if the total mass on integer tokens is below 0.25 (a refusal or malformed output).
- Rubrics live in `configs/traits.yaml`, one per trait pole, adapted from Izawa's format ("Focus only on whether the model exhibits this trait… Respond with a number between 0 and 100").
- The coherence rubric is Izawa's, verbatim.
- **Never include the user message in trait-scoring prompts.**
- Cache every call. Log token usage.

---

## 9. Testing and sanity checklist
- [ ] `python persona_pipeline.py --selftest` passes. It covers the head-score identity, planted-head recovery, the planted cosine (0.6) recovered within its CI, and principal angles.
- [ ] Position decode test (6.3) passes for both models.
- [ ] The `first` readout is identical across different forced responses (6.4).
- [x] Exp 0 validates extraction/scoring against independently executed reference functions on the recorded full-Qwen project sample.
- [x] Exp 0 reports all-head E/A assigned-persona localization, published-SMH ranks, heatmaps, and saved controls; original-paper reproduction is not a completion gate.
- [ ] Head-score identity holds on real activations (relative error below 1e-4 in fp32).
- [ ] Every figure in `results/` can be regenerated from `acts/` plus `config.json` alone.

---

## 10. Pitfalls to avoid
- Using paper head/layer numbers directly (1-indexed).
- Omitting the system message (Qwen injects its own).
- Tokenizing prompt+response as a single string.
- Averaging in bf16.
- Batch padding with position bugs. Stick to bs=1 for extraction.
- Letting the judge see the user message.
- Treating sampled completions as replicates for `first` readouts.
- Using within-cell label shuffles as a null for cosines.
- Estimating a direction and evaluating on the same scenarios (use leave-one-scenario-out or held-out splits).
- Silently changing trait, pole, or scenario definitions. Flag it to the team instead.

---

## 11. Open decisions (ask the team; don't decide unilaterally)
1. Judge model and budget.
2. Final trait list if E or A proves problematic (e.g., refusal rate above 10% for low-A).
3. Whether the forced response is truncated to a fixed token length.
4. The GPU platform. This affects batch size and whether full-layer `W_O` fits in memory.
5. Whether to add multi-turn stimuli if the single-turn mirroring effect is weak.
