# Literature review: user-induced and assigned persona in language models

**Project:** AttentionSeekers / *Causal Disentanglement of User-Induced and Assigned Persona in Language Models*  
**Review date:** 2 October 2026  
**Primary model in the current study:** Qwen2.5-7B-Instruct  
**Current empirical scope:** single-turn Extraversion and Agreeableness; behavioral mirroring, head localization, factorial geometry, and head ablation

## Executive conclusion

The strongest defensible version of this project is not:

> We discovered that language models mirror users, or that persona is represented by a small set of attention heads.

Both claims now have substantial prior art. The strongest version is:

> We use a symmetric, content-controlled design to ask whether stylistically equivalent signals from two instruction sources—system-assigned persona and user-expressed personality—share a mechanism in Qwen2.5-7B-Instruct. We find substantial overlap in where the two signals are represented, but weak equivalence in how they are represented and no causal evidence that the published Style Modulation Heads are necessary for single-turn user-style mirroring.

That story is both novel and consistent with the completed results:

- Qwen behaviorally mirrors **Extraversion** but not **Agreeableness** in this setup.
- Assigned-persona and user-style head scores are correlated, especially in response-token readouts.
- The published Style Modulation Heads (SMHs) rank highly for both sources at the response readout.
- The signed assigned/user directions within the SMHs are only moderately aligned, and the factorial user contribution is small relative to the system-prompt contribution.
- Zeroing the three SMHs does not reduce Extraversion mirroring relative to random-head controls.

The literature therefore supports a nuanced conclusion: shared localization is evidence of a common representational neighborhood, not proof of a shared causal circuit. The project's most valuable result is the separation of **representation**, **source equivalence**, and **causal necessity**.

## 1. Review question and selection method

### 1.1 Questions used to search the literature

The review treats the project as the intersection of four literatures:

1. **Behavioral adaptation:** Do assistants mirror a user's personality, style, stance, or conversational behavior?
2. **Persona representation:** Are personas or traits represented as directions or low-dimensional regions in activation space?
3. **Mechanistic localization:** Can persona/style processing be localized to layers, heads, or causal pathways?
4. **Measurement and causal validity:** When is a Big Five score, probe, steering result, patch, or ablation a valid basis for a mechanistic claim?

The search emphasized ACL, EMNLP, NAACL, EACL, TACL, ICLR, ICML, and NeurIPS, plus a small number of directly relevant workshop papers and peer-reviewed journals. Preprints were retained only when unusually close to the project's central question.

### 1.2 Inclusion criteria

A paper was favored if it did at least one of the following:

- directly studied user-induced personality/style adaptation;
- compared conversational roles or sources of control;
- analyzed persona or character traits in internal activations;
- localized persona/style behavior to transformer components;
- introduced a method used by this project, such as contrastive activation directions or activation patching;
- established construct-validity requirements for personality measurement;
- exposed a failure mode that constrains the project's causal language.

### 1.3 Narrowing rubric

The final ten were selected with the following weighted rubric. Scores are comparative, not claims of absolute paper quality.

| Criterion | Weight | Question |
|---|---:|---|
| Direct relevance | 35% | Does it study the same phenomenon, source comparison, model family, or intervention site? |
| Narrative necessity | 25% | Does the paper perform a unique job in the project's argument? |
| Methodological value | 20% | Does it justify or challenge the study design and interpretation? |
| Evidence/venue strength | 10% | Is it peer reviewed at a strong venue or otherwise unusually well supported? |
| Non-redundancy | 10% | Does it add something not already supplied by another selected paper? |

## 2. Broad candidate pool

The pool below is intentionally broader than a conventional related-work section. “Core” means likely to be cited in the main narrative; “supporting” means useful for methods, discussion, or rebuttal; “background” means credible context but not essential to the central claim.

### 2.1 Direct work on mirroring, adaptation, and drift

| Priority | Paper | Venue/status | Why it was considered |
|---|---|---|---|
| Core | Xing, Niu, and Srivastava, [*Chameleon LLMs: User Personas Influence Chatbot Personality Shifts*](https://aclanthology.org/2025.emnlp-main.875/) | EMNLP 2025 | Strong behavioral precedent for user-persona-induced shifts across multiple models and Big Five-related traits. |
| Core | Sprejer, Martínez-Suñé, and Bianchi, [*White-Box Monitoring for Personality Mirroring in Conversational AI*](https://openreview.net/pdf?id=rVlyID0TzV) | ICLR 2026 CAO Workshop | Closest prior work: activation-space detection of natural personality mirroring in multi-turn Gemma conversations. |
| Core | Lu et al., [*The Assistant Axis: Situating and Stabilizing the Default Persona of Language Models*](https://proceedings.mlr.press/v306/lu26g.html) | ICML 2026 | Establishes persona drift along a latent axis and shows monitoring and intervention, but studies departure from the default Assistant persona. |
| Supporting | Frisch and Giulianelli, [*LLM Agents in Interaction: Measuring Personality Consistency and Linguistic Alignment in Interacting Populations of Large Language Models*](https://aclanthology.org/2024.personalize-1.9/) | PERSONALIZE at EACL 2024 | Early evidence that persona-conditioned agents linguistically align with conversational partners and vary in consistency. |
| Supporting | Hong et al., [*Measuring Sycophancy of Language Models in Multi-turn Dialogues*](https://aclanthology.org/2025.findings-emnlp.121/) | Findings of EMNLP 2025 | Treats user-driven conformity as a multi-turn dynamic and supplies concrete behavioral metrics. |
| Supporting | Sharma et al., [*Towards Understanding Sycophancy in Language Models*](https://proceedings.iclr.cc/paper_files/paper/2024/hash/0105f7972202c1d4fb817da9f21a9663-Abstract-Conference.html) | ICLR 2024 | Shows that human-feedback-trained assistants match user beliefs, and that preference data can reward that behavior. |
| Supporting | Shah, Mishra, and Silpasuwanchai, [*Too Nice to Tell the Truth: Quantifying Agreeableness-Driven Sycophancy in Role-Playing Language Models*](https://aclanthology.org/2026.acl-long.1421/) | ACL 2026 | Connects agreeableness to sycophancy and gives a safety consequence for personality manipulation. |
| Background | Tseng et al., [*Two Tales of Persona in LLMs: A Survey of Role-Playing and Personalization*](https://aclanthology.org/2024.findings-emnlp.969/) | Findings of EMNLP 2024 | Useful taxonomy separating assigned role-playing persona from user-oriented personalization. |

### 2.2 Persona and personality as activation-space structure

| Priority | Paper | Venue/status | Why it was considered |
|---|---|---|---|
| Core | Chen et al., [*Persona Vectors: Monitoring and Controlling Character Traits in Language Models*](https://arxiv.org/abs/2507.21509) | 2025 preprint / ICLR 2026 OpenReview version | Establishes automated trait directions used for monitoring, training-shift prediction, and control. It is the most direct residual-space precursor to both Sprejer and Izawa. |
| Core | Rimsky et al., [*Steering Llama 2 via Contrastive Activation Addition*](https://aclanthology.org/2024.acl-long.828/) | ACL 2024 | Foundational, peer-reviewed contrastive activation steering method; clarifies what a difference-in-means direction can and cannot establish. |
| Core/near-core | Bhandari et al., [*Activation-Space Personality Steering: Hybrid Layer Selection for Stable Trait Control in LLMs*](https://aclanthology.org/2026.eacl-long.300/) | EACL 2026 | Direct Big Five activation-space work with low-rank structure and layer selection. |
| Supporting | Konen et al., [*Style Vectors for Steering Generative Large Language Models*](https://aclanthology.org/2024.findings-eacl.52/) | Findings of EACL 2024 | Shows that recorded hidden activations can support graded style control. |
| Supporting | Zhang et al., [*Personalized Text Generation with Contrastive Activation Steering*](https://aclanthology.org/2025.acl-long.353/) | ACL 2025 | Extracts personalized writing style from histories and explicitly frames content/style entanglement as a central problem. |
| Supporting | Heyman and Vandeputte, [*Steer Like the LLM: Activation Steering that Mimics Prompting*](https://proceedings.mlr.press/v306/heyman26a.html) | ICML 2026 | Demonstrates that prompt-induced interventions are position dependent; useful when discussing why system and user prompts need not reduce to the same constant direction. |
| Supporting | Scherrer et al., [*Evaluating Language Model Character Traits*](https://aclanthology.org/2024.findings-emnlp.77/) | Findings of EMNLP 2024 | Offers a behaviorist definition of character traits and warns against unnecessary anthropomorphism. |
| Background | Hu and Collier, [*Quantifying the Persona Effect in LLM Simulations*](https://aclanthology.org/2024.acl-long.554/) | ACL 2024 | Quantifies how much persona variables affect simulated judgments; useful for keeping claims about “persona” proportional. |

### 2.3 Component- and circuit-level persona mechanisms

| Priority | Paper | Venue/status | Why it was considered |
|---|---|---|---|
| Core | Izawa et al., [*Steering at the Source: Style Modulation Heads for Robust Persona Control*](https://proceedings.mlr.press/v306/izawa26a.html) | ICML 2026 | The direct mechanistic baseline and source of the Qwen layer-19 heads 2, 4, and 27 used by this project. |
| Core | Ye, Cui, and Hadfield-Menell, [*Prompt Injection as Role Confusion*](https://proceedings.mlr.press/v306/ye26b.html) | ICML 2026 | Shows that style can dominate explicit conversational role tags in latent representations; a strong mechanistic bridge for source leakage. |
| Core | Poonia and Jain, [*Dissecting Persona-Driven Reasoning in Language Models via Activation Patching*](https://aclanthology.org/2025.findings-emnlp.1335/) | Findings of EMNLP 2025 | Gives component-level causal evidence for assigned-persona effects across MLPs, MHA layers, and individual heads. |
| Supporting | Conmy et al., [*Towards Automated Circuit Discovery for Mechanistic Interpretability*](https://proceedings.neurips.cc/paper_files/paper/2023/hash/34e1dbe95d34d7ebaf99b9bcaeb5b2be-Abstract-Conference.html) | NeurIPS 2023 | Establishes a standard circuit-localization workflow and emphasizes behavior-specific metrics and datasets. |
| Supporting | Mueller et al., [*MIB: A Mechanistic Interpretability Benchmark*](https://proceedings.mlr.press/v267/mueller25a.html) | ICML 2025 | Provides modern standards for evaluating circuit localization and causal-variable discovery. |
| Supporting | [*Causal Head Gating: A Framework for Interpreting Roles of Attention Heads in Transformers*](https://proceedings.neurips.cc/paper_files/paper/2025/hash/a7f530e11fa19e9551b7a51dbd0f336f-Abstract-Conference.html) | NeurIPS 2025 | Shows that sparse task-sufficient head circuits can coexist and head roles can depend on interactions, weakening one-head/one-function stories. |
| Supporting | Geiger et al., [*Causal Abstractions of Neural Networks*](https://proceedings.neurips.cc/paper/2021/hash/4f5c422f4d49a5a807eda27434231040-Abstract.html) | NeurIPS 2021 | Formal foundation for relating neural interventions to high-level causal variables. |

### 2.4 Personality measurement and construct validity

| Priority | Paper | Venue/status | Why it was considered |
|---|---|---|---|
| Core | Serapio-García et al., [*A Psychometric Framework for Evaluating and Shaping Personality Traits in Large Language Models*](https://www.nature.com/articles/s42256-025-01115-6) | Nature Machine Intelligence 2025 | Strongest construct-validity support for using Big Five traits while also showing that reliability depends on model and prompting configuration. |
| Supporting | Jiang et al., [*PersonaLLM: Investigating the Ability of Large Language Models to Express Personality Traits*](https://aclanthology.org/2024.findings-naacl.229/) | Findings of NAACL 2024 | Demonstrates that assigned Big Five personas affect both questionnaires and open-ended writing, including human perceptibility. |
| Supporting | Zheng et al., [*LMLPA: Language Model Linguistic Personality Assessment*](https://aclanthology.org/2025.cl-2.6/) | Computational Linguistics 2025 | Explicitly adapts personality assessment to generated language rather than relying only on human self-report questionnaires. |
| Supporting | Gupta et al., [*Do LLMs Have Distinct and Consistent Personality? TRAIT*](https://aclanthology.org/2025.findings-naacl.225/) | Findings of NAACL 2025 | Uses scenario-based psychometric tests and is useful for discussing consistency and prompt sensitivity. |
| Supporting | [*Evaluation Drift in LLM Personality Induction: Are We Moving the Goalpost?*](https://aclanthology.org/2026.lrec-1.881/) | LREC 2026 | Warns that personality evaluations can become unstable under rephrasing and post-training. |
| Background | Karra et al., [*Estimating the Personality of White-Box Language Models*](https://aclanthology.org/2022.emnlp-main.220/) | EMNLP 2022 | Earlier white-box personality-estimation work; historically useful but less direct than current activation studies. |

### 2.5 Validity limits for causal interpretability

| Priority | Paper | Venue/status | Why it was considered |
|---|---|---|---|
| Core | Makelov et al., [*Is This the Subspace You Are Looking for? An Interpretability Illusion for Subspace Activation Patching*](https://openreview.net/forum?id=Ebt7JgMHv1) | ICLR 2024 | Shows that a successful intervention can activate a dormant parallel pathway and therefore fail to identify the mechanism used naturally. |
| Core/near-core | Zhang and Nanda, [*Towards Best Practices of Activation Patching in Language Models: Metrics and Methods*](https://openreview.net/pdf?id=Hf17y6u9BC) | ICLR 2024 | Demonstrates that patching results can change substantially with corruption and metric choices. |
| Supporting | Mueller et al., [*MIB*](https://proceedings.mlr.press/v267/mueller25a.html) | ICML 2025 | Makes precision, concision, and causal pathway recovery explicit evaluation goals. |
| Supporting | Asiaee, [*Certified Interventional Fidelity*](https://proceedings.mlr.press/v337/asiaee26d.html) | UAI 2026 | Adds formal estimands and uncertainty to intervention claims; especially relevant to finite sample and adaptive analysis concerns. |

## 3. The ten most important papers

The papers are ordered to build the argument, not by prestige or citation count.

### 1. Chameleon LLMs: User Personas Influence Chatbot Personality Shifts

**Citation:** Jane Xing, Tianyi Niu, and Shashank Srivastava. EMNLP 2025. [Paper](https://aclanthology.org/2025.emnlp-main.875/)

**What it establishes.** In controlled, prolonged interactions, multiple LLMs change their measured personality in response to the simulated user's persona. Agreeableness, Extraversion, and Conscientiousness are especially susceptible, while Emotional Stability and Intellect are more stable.

**Why it is essential.** This is the best peer-reviewed behavioral anchor for the project's phenomenon. It means the project should not claim to be the first demonstration of personality mirroring. Instead, it moves from “does adaptation occur?” to “where and how does a user-carried signal interact with an assigned persona?”

**What it does not answer.** It uses behavioral before/after personality measurement over prolonged interaction. It does not isolate fixed content, inspect individual heads, compare equivalent user/system trait manipulations, or causally test a proposed circuit.

**Connection to our results.** Our Extraversion result agrees with its broad adaptation finding. Our null Agreeableness result does not directly contradict it because our design is single-turn, model-specific, and response-scored rather than a pre/post questionnaire. The contrast is useful evidence that mirroring is trait-, model-, and protocol-dependent.

**Use in the paper.** First paragraph of related work and discussion of why single-turn trait-specific results should not be generalized to stable personality change.

### 2. White-Box Monitoring for Personality Mirroring in Conversational AI

**Citation:** Eitan Sprejer, Agustín E. Martínez-Suñé, and Bruno Bianchi. CAO Workshop at ICLR 2026. [Paper](https://openreview.net/pdf?id=rVlyID0TzV)

**What it establishes.** Projecting Gemma-2-27B-it activations onto personality-trait principal components detects large user-mediated persona shifts across 2,940 simulated multi-turn conversations. Surface text features and an independent LLM judge move in the same broad direction. Topic alone also explains substantial variance.

**Why it is essential.** This is the closest prior work and should be treated as such. It establishes both the behavioral phenomenon and a white-box signature, while explicitly leaving open whether the activation shift is the model adopting a persona or simply representing the styled user text still present in context.

**What it does not answer.** The principal-component projection is model-level and correlational. It does not localize the effect to heads, compare user-carried and system-carried versions of the same trait under fixed content, or intervene on the proposed substrate.

**Connection to our results.** Our first-token, user-token, and response-token readouts directly address the adoption-versus-perception ambiguity. The moderate assigned/user head-score overlap supports shared processing, while the small factorial user contribution and negative ablation result argue against treating a projection shift as proof that the model uses the same circuit to generate mirrored behavior.

**Use in the paper.** State the gap immediately after summarizing the paper: *white-box detectability does not identify source-specific causal mechanism*.

### 3. The Assistant Axis: Situating and Stabilizing the Default Persona of Language Models

**Citation:** Christina Lu, Jack Gallagher, Jonathan Michala, Kyle Fish, and Jack Lindsey. ICML 2026. [Paper](https://proceedings.mlr.press/v306/lu26g.html)

**What it establishes.** Persona space has a dominant “Assistant Axis” across several models. Certain multi-turn contexts—especially emotionally vulnerable and meta-reflective conversations—move activations away from the default Assistant region. Steering or constraining this axis can stabilize behavior and reduce persona-based jailbreak susceptibility.

**Why it is essential.** It provides the strongest modern evidence that persona drift is geometrically monitorable and causally controllable. It also demonstrates that topic/domain is not a nuisance to ignore: it is a major driver of persona trajectories.

**What it does not answer.** “Persona drift” here means departure from the post-trained default Assistant identity. It is not degradation of a custom Big Five system persona, and it is not the same construct as single-turn stylistic mirroring.

**Connection to our results.** Our domain heterogeneity—especially weak Extraversion mirroring in emotional advice—should be reported rather than averaged away. But the paper must not use Lu et al. to label our single-turn effect “drift.” The accurate term is **single-turn user-style mirroring**.

**Use in the paper.** Motivation and scope boundary. It supports the practical importance of persona stability while requiring terminological discipline.

### 4. Persona Vectors: Monitoring and Controlling Character Traits in Language Models

**Citation:** Runjin Chen, Andy Arditi, Henry Sleight, Owain Evans, and Jack Lindsey. 2025. [Paper](https://arxiv.org/abs/2507.21509)

**What it establishes.** Natural-language trait descriptions can be converted into activation directions that monitor and control traits such as evil, sycophancy, and hallucination. These directions predict personality shifts caused by fine-tuning and support both post-hoc and preventative interventions.

**Why it is essential.** This is the clearest basis for treating character traits as manipulable activation-space objects. It underlies the trait-space projections used by Sprejer et al. and provides data/methodological foundations for Izawa et al.

**What it does not answer.** A residual-stream direction does not identify where a trait originates, whether the same direction is used for user and system sources, or whether steering recruits the mechanism used during normal generation.

**Connection to our results.** Our factorial regression shows why a single direction is insufficient for the source-comparison question: assigned and user sources can affect overlapping components yet have very unequal explained variance and only modest signed alignment.

**Use in the paper.** Foundation for persona geometry, followed by the sentence: *geometry alone does not establish source equivalence or causal necessity*.

### 5. Steering at the Source: Style Modulation Heads for Robust Persona Control

**Citation:** Yoshihiro Izawa, Gouki Minegishi, Koshi Eguchi, Sosuke Hosokawa, and Kenjiro Taura. ICML 2026. [Paper](https://proceedings.mlr.press/v306/izawa26a.html)

**What it establishes.** Persona/style control can be localized to a sparse set of attention heads, identified by layer-wise geometry and head contribution scores. On Qwen2.5-7B-Instruct, the reported set corresponds to zero-indexed layer 19, heads 2, 4, and 27. Targeted steering of these heads achieves a better trait/coherence trade-off than broad residual-stream steering.

**Why it is essential.** It supplies the exact mechanistic hypothesis, scoring method, reference implementation, and candidate heads tested by this project. This is the paper against which the project makes its most direct empirical contribution.

**What it does not answer.** Its heads are selected using system-prompt persona contrasts and traits such as humorous, passionate, loser, evil, sycophantic, and hallucinating. That does not establish that the same heads are necessary for Big Five user-induced mirroring. Localization based on a system source also creates selection bias if the evaluation is subsequently restricted to those heads.

**Connection to our results.** We strengthen the comparison by using symmetric system/user twins, fixed responses, independent localization, all-head rank correlations, same-layer random controls, and a direct zero-ablation test. The results partly generalize Izawa et al.—the heads remain highly ranked—but falsify the simplest extension: zeroing them does not reduce Extraversion mirroring.

**Use in the paper.** Central baseline. Be exact that we validated extraction against reference functions on project inputs and assessed the reported heads on E/A; we did not reproduce the original trait experiment.

### 6. Prompt Injection as Role Confusion

**Citation:** Charles Ye, Jasmine Cui, and Dylan Hadfield-Menell. ICML 2026. [Paper](https://proceedings.mlr.press/v306/ye26b.html)

**What it establishes.** Models infer “who is speaking” partly from spoofable stylistic cues rather than reliably preserving tag-defined source. Text that sounds like a trusted role can occupy the trusted role's representational space, and role-confusion scores predict attack success before generation.

**Why it is essential.** Although the task is prompt injection rather than personality mirroring, it gives the strongest mechanistic reason to expect user style to cross a formal role boundary. It turns the project's source-comparison question from an isolated curiosity into an instance of a broader problem: interface-level role separation need not survive in latent space.

**What it does not answer.** Authority confusion under adversarial injection is not equivalent to natural style mirroring. A shared role-probe geometry also does not imply a shared generation circuit.

**Connection to our results.** The overlap in assigned/user head rankings is compatible with role/style leakage. The small user coefficient and null SMH ablation add an important qualification: latent source confusion can coexist with different causal pathways or distributed downstream computation.

**Use in the paper.** Mechanistic motivation for comparing equivalent style cues placed in system versus user messages. Avoid claiming that our experiment tests prompt injection.

### 7. Dissecting Persona-Driven Reasoning in Language Models via Activation Patching

**Citation:** Ansh Poonia and Maeghal Jain. Findings of EMNLP 2025. [Paper](https://aclanthology.org/2025.findings-emnlp.1335/)

**What it establishes.** Assigned demographic personas can change objective-task reasoning. Activation patching suggests early MLP layers enrich persona-token representations and middle attention layers use them to shape outputs; some individual heads disproportionately attend to identity cues.

**Why it is essential.** It is a rare peer-reviewed paper connecting persona assignment to component-level causal analysis rather than only probes or output scores.

**What it does not answer.** It studies assigned demographic identities and MMLU behavior, not user-expressed Big Five style. Its identified components are therefore evidence that persona effects can be mechanistically localized, not evidence for the specific SMH mechanism.

**Connection to our results.** It motivates interventions but also highlights why “persona” is too broad a mechanistic category. Demographic identity, system-assigned writing style, and user-induced mirroring can rely on different circuits even if all produce persona-like output changes.

**Use in the paper.** Related work on causal persona analysis and a justification for separating traits, sources, and behaviors.

### 8. A Psychometric Framework for Evaluating and Shaping Personality Traits in Large Language Models

**Citation:** Gregory Serapio-García et al. Nature Machine Intelligence, 2025. [Paper](https://www.nature.com/articles/s42256-025-01115-6)

**What it establishes.** Big Five-like personality measurements can be reliable and construct-valid for some LLMs under specific prompting conditions, especially for larger instruction-tuned models. Personality expression can be shaped at multiple levels using a broad adjective set.

**Why it is essential.** It is the strongest defense of the project's use of Big Five as a model-independent trait taxonomy. Just as importantly, it prevents overclaiming: validity belongs to a measurement protocol and model configuration, not automatically to any LLM output labeled with a trait.

**What it does not answer.** Reliable measurement does not imply human-like psychological possession, temporal stability, or a unique neural mechanism. Much of the validation relies on questionnaire-style administration rather than task-conditioned assistant replies.

**Connection to our results.** Our response-only judge, multiple domains, matched facets, and behavioral generation are better treated as measurements of **perceived trait expression** than proof that Qwen “has” a personality. Agreeableness saturation near 91 across all conditions is a measurement/result ceiling that must be reported, not interpreted as evidence of no internal processing.

**Use in the paper.** Dataset and evaluation rationale; terminology should remain “trait expression” or “perceived personality,” not intrinsic personality.

### 9. Steering Llama 2 via Contrastive Activation Addition

**Citation:** Nina Rimsky et al. ACL 2024. [Paper](https://aclanthology.org/2024.acl-long.828/)

**What it establishes.** Contrastive Activation Addition constructs a behavioral direction from mean activation differences between positive and negative examples and adds it during inference to steer behavior. It demonstrates graded control across several behaviors while largely preserving capabilities.

**Why it is essential.** It is the cleanest peer-reviewed methodological ancestor for difference-in-means activation directions used throughout the persona-vector and SMH literature.

**What it does not answer.** A direction that steers behavior need not be the natural causal mediator of that behavior. Difference vectors can also contain content, length, lexical, or prompt-template effects.

**Connection to our results.** The project's fixed-response twin design, matched scenarios, facet reporting, user-length checks, and source-separated contrasts are defenses against these confounds. The fact that hundreds of heads—including same-pole wording controls—are significant reinforces why ranks, effect sizes, and causal interventions matter more than significance alone.

**Use in the paper.** Methodological lineage and a concise explanation of why contrastive geometry requires controlled pairs.

### 10. Is This the Subspace You Are Looking for? An Interpretability Illusion for Subspace Activation Patching

**Citation:** Aleksandar Makelov, Georg Lange, Atticus Geiger, and Neel Nanda. ICLR 2024. [Paper](https://openreview.net/forum?id=Ebt7JgMHv1)

**What it establishes.** A subspace intervention can have the expected causal effect while activating a dormant parallel pathway that is not the pathway normally responsible for model behavior. Intervention success can therefore create a false impression of mechanistic identification.

**Why it is essential.** This paper supplies the epistemic guardrail the project needs. Head ranking, probing, steering, patching, and ablation answer different questions. None alone licenses “this is the circuit.”

**Connection to our results.** Our null ablation is informative precisely because it tests necessity under natural prompting and compares against random-head controls. It does not prove the heads are irrelevant: redundancy, compensation, incomplete intervention, or distributed computation remain possible. The defensible conclusion is narrower: **the tested three-head set is not necessary for the measured single-turn mirroring effect under this intervention**.

**Use in the paper.** Limitations and causal-language policy. This citation will make the negative intervention result look principled rather than disappointing.

## 4. Why these ten, and why not the nearest alternatives?

| Near miss | Reason it did not displace a top-ten paper | Where to cite it |
|---|---|---|
| Bhandari et al., *Activation-Space Personality Steering* (EACL 2026) | Highly relevant Big Five geometry, but the final ten already contain stronger anchors for psychometric validity, contrastive steering, and source-specific persona geometry. | Related work on Big Five activation steering. |
| Jiang et al., *PersonaLLM* (Findings of NAACL 2024) | Important behavioral evidence, but Serapio-García et al. provides the more rigorous construct-validity foundation and Chameleon provides the direct interaction result. | Big Five induction and human perceptibility. |
| Zhang and Nanda, *Best Practices of Activation Patching* (ICLR 2024) | Extremely useful methods paper, but Makelov et al. contributes the sharper conceptual warning needed for the project's causal claims. | Methods/limitations; cite both if activation patching is added. |
| Mueller et al., *MIB* (ICML 2025) | Strong evaluation standard, but less directly connected to the current experiment than the selected persona and intervention papers. | Future work on systematic circuit recovery. |
| Frisch and Giulianelli, *LLM Agents in Interaction* (workshop 2024) | Early and direct, but smaller and less conclusive than Chameleon and Sprejer. | Historical bridge from linguistic alignment to LLM mirroring. |
| Sharma et al., *Towards Understanding Sycophancy* (ICLR 2024) | Important safety context, but stance agreement is distinct from stylistic mirroring. | Motivation and Agreeableness discussion. |
| Shah et al., *Too Nice to Tell the Truth* (ACL 2026) | Directly connects Agreeableness to a safety outcome, but studies assigned role-playing personas rather than user-induced style. | Discussion of why Agreeableness control matters. |
| Heyman and Vandeputte, *Steer Like the LLM* (ICML 2026) | Excellent evidence that prompt effects are position-dependent; however, the project does not currently train a steering replacement model. | Discussion of why a constant direction may underfit source/position effects. |
| Causal Head Gating (NeurIPS 2025) | Valuable warning about interacting and non-unique sparse circuits, but it is not persona-specific. | Limitations of single-head-set interpretations. |
| Scherrer et al., *Evaluating Language Model Character Traits* (Findings of EMNLP 2024) | Strong conceptual framing, but less directly useful than the psychometric framework for defending the actual trait measurements. | Terminology and anti-anthropomorphism statement. |

## 5. The literature-backed story for the paper

### 5.1 Recommended narrative arc

1. **Conversational adaptation is real but heterogeneous.** Behavioral studies show that user personas can change model personality and that models linguistically align with partners. Effects vary across traits, models, domains, and protocols.
2. **Persona shifts are visible in activation geometry.** Persona vectors, white-box mirroring, and the Assistant Axis show that character and persona shifts occupy structured activation-space directions.
3. **But a latent shift has multiple interpretations.** It can reflect perception of user text, adoption in the assistant's generation policy, topic effects, or a mixture. White-box detection alone does not distinguish these.
4. **Sparse head localization creates a testable causal hypothesis.** Izawa et al. identify SMHs under system-assigned persona control. Role-confusion work makes it plausible that style can cross formal role boundaries.
5. **Our design makes the source comparison symmetric.** The same scenario and fixed response are used while style is placed in the system or user message; sources are localized independently; readouts distinguish user processing, the first assistant position, and response-token processing.
6. **Our results support overlap without equivalence.** The two sources rank many of the same heads highly, but their directions and explained variances differ.
7. **The causal test rejects the simplest circuit story.** Removing the published SMHs does not diminish Extraversion mirroring compared with random-head controls.
8. **Conclusion:** a shared representational locus does not imply a single, necessary mirroring circuit. User-induced persona is better modeled as a source- and trait-dependent influence on distributed generation dynamics.

### 5.2 A compact gap statement

> Prior work has shown, separately, that conversational partners can shift an LLM's expressed personality, that such shifts are detectable in activation space, and that system-assigned persona control can be localized to sparse Style Modulation Heads. What remains unresolved is whether an equivalent trait carried by the user and by the system prompt enters the model through the same representational and causal pathway. Existing mirroring studies do not localize a circuit, while existing circuit studies select components using assigned personas. We close this gap with a symmetric source-controlled design, independent head localization, and a targeted necessity test.

### 5.3 Suggested contribution statement

> We contribute (1) a counterfactual dataset that places matched Big Five style cues in either the system or user role while holding task content and assistant response fixed; (2) a source-symmetric comparison of head-level localization across user and assigned persona signals in Qwen2.5-7B-Instruct; (3) a response-only behavioral evaluation showing robust single-turn Extraversion mirroring but no clear Agreeableness mirroring; and (4) a controlled ablation showing that the published Style Modulation Heads are not necessary for the observed Extraversion mirroring effect.

## 6. How the current evidence should be interpreted

| Current observation | Supported statement | Statement to avoid |
|---|---|---|
| E high-minus-low = 13.05, paired effect size 1.39 | Qwen expresses more Extraversion in replies to high- than low-Extraversion user phrasings in this single-turn dataset. | Qwen's personality becomes extraverted. |
| A high-minus-low = 0.48, CI crosses zero; all means near 91 | The study does not detect clear Agreeableness mirroring, and the judge/model combination has a high baseline that may limit headroom. | Qwen does not process user Agreeableness. |
| Assigned/user head-score Spearman correlations up to about 0.78 | The two source contrasts share substantial head-level ranking structure, especially over response tokens. | The same heads implement both behaviors. |
| Published SMHs are top-ranked for user and assigned response contrasts | SMHs are a shared representational locus for these controlled contrasts. | SMHs are the user-mirroring circuit. |
| SMH direction cosines are positive but modest; user R² is tiny relative to assigned R² | Source effects are partially aligned but not interchangeable in the tested head group. | System and user prompts encode the same persona vector. |
| SMH zero ablation leaves E mirroring intact | The tested three-head set is not necessary for the observed single-turn Extraversion effect under this intervention. | The SMHs play no role in persona or style. |
| Hundreds of heads pass BH correction, including same-pole wording controls | Wording/style manipulations are distributed and statistical significance is not a specificity criterion; ranks and effect sizes are more informative. | Persona is represented in hundreds of dedicated heads. |

## 7. Threats to defensibility and literature-backed repairs

### 7.1 “Personality” versus perceived linguistic style

The project manipulates phrasing and scores generated text. The safe construct is **perceived personality-trait expression in language**, not a persistent inner personality. Use Serapio-García et al., Jiang et al., and Scherrer et al. to make this explicit.

### 7.2 Single-turn mirroring versus persona drift

Lu et al. and Chameleon examine changes over extended interactions. Our result is a conditional single-turn response effect. Reserve “drift” for longitudinal departure from a reference persona and present multi-turn evaluation as future work.

### 7.3 Trait realization and lexical shortcuts

The current Agreeableness variants reuse politeness markers, while each paraphrase index intentionally emphasizes a different facet. Treat facet as a factor, report facet-level effects, audit lexical overlap, and avoid calling the three paraphrases exchangeable replicates. Contrastive steering literature makes content/style control central, not optional.

### 7.4 Judge validity and ceiling effects

The response-only judge avoids direct leakage from the user message, which is a major strength. Still, the Agreeableness score near 91 in all conditions suggests a ceiling or a strong post-training prior. Report raw distributions, blinded human validation on a stratified subset, and judge sensitivity to prompt or model choice if feasible.

### 7.5 Localization is not circuit identification

Head contribution scores, rank overlap, and significant contrasts identify associated components. A necessity/sufficiency claim needs interventions and carefully defined metrics. Makelov et al., Zhang and Nanda, MIB, and Causal Head Gating collectively justify conservative language, random controls, and tests of alternative or redundant pathways.

### 7.6 Null ablation is bounded, not universal

Zeroing pre-`o_proj` outputs is a meaningful necessity test, but a null can result from redundancy, compensation, distributed representation, or mismatch between the ablated component and the relevant direction within it. Report exactly when and where the hook operates and avoid extending the result beyond Qwen2.5-7B-Instruct, E/A, and the tested prompts.

## 8. High-value additional analyses suggested by the literature

These are ordered by likely payoff for the final paper.

1. **Human-check the behavioral judge on a stratified subset.** Sample E/A, high/low/neutral, all domains, and all facets. Report ordinal agreement and high-minus-low effect replication.
2. **Report facet-level effects as first-class results.** The existing Extraversion result is driven by sociability and energy, not assertiveness. This improves construct validity and explains why an aggregate Big Five label is incomplete.
3. **Quantify lexical confounds.** Regress scores and head contrasts on length and simple lexical markers; repeat after removing greetings, “please,” “thank you,” exclamation marks, and obvious energy markers where possible.
4. **Add head-set sufficiency or directional intervention only if carefully scoped.** A causal steering experiment could test whether adding the user direction at SMHs increases mirroring, but Makelov et al. means success should be described as controllability, not proof of the natural mechanism.
5. **Test alternative top user-localized heads.** The SMH ablation asks whether Izawa's system-selected heads are necessary. Ablating independently user-selected heads would test whether mirroring has a different sparse locus.
6. **Test combined or path-aware interventions.** Causal Head Gating and MIB suggest that interacting sets or edges may be more faithful than isolated-head removal.
7. **Replicate on another model only after the core claims are frozen.** Role confusion and circuit-localization results can vary sharply across model families. A Llama replication would test portability, not merely increase sample size.
8. **Add a short multi-turn extension with the same controlled facets.** This would connect the single-turn result to Chameleon, Sprejer, and Lu without conflating the two in the main claim.

## 9. Proposed related-work structure

### Paragraph 1: behavioral mirroring and linguistic alignment

Cite Xing et al.; Sprejer et al.; Frisch and Giulianelli; optionally Sharma et al. Distinguish stylistic mirroring from stance sycophancy and from longitudinal drift.

### Paragraph 2: activation-space persona geometry

Cite Chen et al.; Lu et al.; Rimsky et al.; Bhandari et al. State that these methods establish monitorable/steerable directions but not source identity or component-level origin.

### Paragraph 3: component-level persona mechanisms

Cite Izawa et al.; Poonia and Jain; Ye et al. Explain why system/user source equivalence is plausible but untested.

### Paragraph 4: measurement and causal validity

Cite Serapio-García et al.; Makelov et al.; optionally Zhang and Nanda or MIB. State the project's policy: behavior, representation, controllability, and necessity are reported separately.

## 10. Citation corrections for the existing proposal

- Replace “Lu (2026)” with the full ICML citation: Christina Lu, Jack Gallagher, Jonathan Michala, Kyle Fish, and Jack Lindsey, *The Assistant Axis: Situating and Stabilizing the Default Persona of Language Models*.
- Replace “et al. Izawa” with Yoshihiro Izawa, Gouki Minegishi, Koshi Eguchi, Sosuke Hosokawa, and Kenjiro Taura, and cite ICML 2026, not only the earlier workshop/preprint version.
- Replace the shortened Izawa title with *Steering at the Source: Style Modulation Heads for Robust Persona Control*.
- Cite Sprejer et al. accurately as a CAO Workshop paper at ICLR 2026, not as a full ICLR conference paper.
- Do not attribute the claim “residual-stream steering reduces coherence” to Subramani et al. (2022). The direct evidence for the trait/coherence trade-off in this project line is Izawa et al.; CAA and later steering papers provide broader context.
- Update Ye, Cui, and Hadfield-Menell from “Anonymous (2026)” or preprint-only status to ICML 2026.
- Clarify that Persona Vectors and the Assistant Axis study character/persona directions at the residual-stream level, whereas this project compares source-specific head-level effects.

## 11. One-paragraph defensible positioning

> Conversational language models adapt to user personas and styles, and recent work has detected these shifts both behaviorally and in activation space. Separately, persona-control studies have identified low-dimensional residual directions and a sparse set of Style Modulation Heads under system-prompt interventions. These findings do not establish that a trait expressed by the user and the same trait assigned by the system prompt share a causal mechanism: activation shifts may reflect perception of styled context, source-dependent routing, or adoption in the generated response. We test this distinction with matched Big Five contrasts that place style in either the user or system role while controlling task content and assistant responses. Across Qwen2.5-7B-Instruct, the two sources show overlapping head-level localization, but their activation directions and explained variances differ; moreover, ablating the published SMHs does not reduce observed single-turn Extraversion mirroring. The results separate shared representational sensitivity from causal necessity and argue against a simple unified-SMH account of user-induced persona.

## 12. Final shortlist at a glance

| # | Paper | Unique job in the story |
|---:|---|---|
| 1 | Xing et al., *Chameleon LLMs* | Peer-reviewed behavioral mirroring baseline |
| 2 | Sprejer et al., *White-Box Monitoring* | Closest activation-based mirroring work and explicit open gap |
| 3 | Lu et al., *Assistant Axis* | Persona drift geometry, causal control, and scope boundary |
| 4 | Chen et al., *Persona Vectors* | Trait directions as monitorable and steerable objects |
| 5 | Izawa et al., *Steering at the Source* | Exact SMH hypothesis and mechanistic baseline |
| 6 | Ye et al., *Role Confusion* | Why stylistic signals may cross formal source boundaries |
| 7 | Poonia and Jain, *Persona-Driven Reasoning* | Component-level causal persona precedent |
| 8 | Serapio-García et al., *Psychometric Framework* | Defense and limits of Big Five measurement |
| 9 | Rimsky et al., *Contrastive Activation Addition* | Methodological foundation for difference directions |
| 10 | Makelov et al., *Interpretability Illusion* | Guardrail against equating intervention with natural mechanism |

## 13. Bottom-line assessment

The project is defensible if it embraces its negative causal result. The literature already makes it unsurprising that persona and style are visible in activations and that a few heads can be useful intervention sites. What is not established—and what this project tests unusually well—is whether source-matched user and system style are mechanistically identical. The evidence says **no simple equivalence**: overlap in localization is real, but the user signal is weaker, directionally imperfect, behaviorally trait-specific, and not abolished by removing the published SMHs. That is a clearer and more credible contribution than claiming discovery of a mirroring circuit.
