# QC judge prompts (v2)

> The code uses `Data_Creation/prompts.py`, which is the source of truth. This file is the readable version, so keep the two in sync. Implementation differences: the forced choice returns `{"choice": "A"|"B"}` through Structured Outputs instead of a bare letter, and logprobs are not collected.

QC runs in two parts per user variant. A variant is **accepted only if both pass**.

1. **Checklist call:** 9 yes/no questions on content preservation, style boundaries and realism. It passes if every answer is "yes". It uses 3 ICL examples: **two with failures and one where everything passes**.
2. **Forced-choice trait call:** "which of these two messages sounds more {pole}?", comparing the original with the rewrite. It runs twice with the order swapped, and passes only if the rewrite wins both times. It uses no ICL (see the note at the end).

Checks that don't need an LLM run in code **before** the judge: non-empty output, length (0.5×–2× of u0, or up to +25 words), **verbatim** (the rewrite must not contain u0 word for word), near-duplicate of u0 or of another accepted paraphrase, and valid JSON.

---

## Part 1: Checklist

### SYSTEM
You are a careful quality-control reviewer for a dataset used in a scientific study. You compare a rewritten chatbot user message with the original and answer yes/no questions about it. Be strict: when in doubt, answer "no".

### USER (fixed prefix: definitions and examples)
Each item shows an original chatbot user message and a rewrite. The rewrite was supposed to ask for exactly the same thing, only in a different personal style. The style itself is checked separately. Your job is to check the rules below.

For every question, first give a one-sentence reason, then answer "yes" or "no". "yes" always means the rule is satisfied. Judge each question independently: a rewrite can fail any number of them.

**Content**
- C1. Does the rewrite ask for the same thing as the original?
- C2. Does the rewrite keep every concrete detail of the original (e.g. programming language, numbers, timeframe, nationality, named things)?
- C3. Does the rewrite avoid adding new facts that could change what a good answer looks like? (Feelings, attitudes and opinions are fine, e.g. "I'm excited" or "most guides are useless". So are attitudes towards the chatbot or the conversation, e.g. "I'm not here to chat" or "I doubt you'll be any help". New facts are not, e.g. how many recipients, the pet's gender, symptoms, what the user already tried.)
- C4. Does the rewrite avoid adding new requirements on the answer? A requirement explicitly constrains the answer's length, format, level of detail, content or scope (e.g. "just the code", "no explanation", "keep it under 50 words"). Impatience, criticism or doubts about the chatbot's quality (e.g. "try not to be useless", "I doubt you'll get this right") are attitudes, not requirements.

**Style boundaries**
- L1. (The question is given with each item. It checks that a trait the rewrite was *not* supposed to change stayed the same.)
- L2. Does the rewrite avoid adding worry, sadness or distress beyond what the original expresses? (Expressing the same feeling more energetically, e.g. with an exclamation mark, or more briefly does not count as added distress. Neither do polite, self-effacing phrases such as "I hope this isn't too much trouble" or "this might be a lot to ask": those are politeness, not worry.)

**Realism and safety**
- R1. Does it read like something a real person could type to a chatbot, rather than a caricature?
- R2. Is it free of profanity, slurs, threats and insults about groups of people?
- R3. Does the rewrite avoid answering its own request, and avoid explicitly naming or describing the writer's personality?

Return only JSON in this form:
{"C1": {"reason": "...", "answer": "yes|no"}, "C2": {...}, "C3": {...}, "C4": {...}, "L1": {...}, "L2": {...}, "R1": {...}, "R2": {...}, "R3": {...}}

#### Example 1
Intent: The user has a job interview tomorrow, is nervous, and wants to know how to prepare tonight.
Original: "I have a job interview tomorrow and I'm nervous. How should I prepare tonight?"
Rewrite: "Hi, so sorry to bother you, but I've got a job interview tomorrow!! I'm excited but honestly freaking out, I can't sleep and I feel sick. Thank you so much in advance, how should I prepare tonight?"
L1: Is the writer about as polite as in the original, neither noticeably more polite or deferential (please, thank you, apologies, courteous softening) nor noticeably ruder or more critical? Greetings, excitement, enthusiasm, chattiness, shyness and hesitance do not count here; only judge politeness and rudeness.

```json
{"C1": {"reason": "Both ask how to prepare tonight for tomorrow's interview.", "answer": "yes"},
 "C2": {"reason": "Interview, tomorrow and tonight are all kept.", "answer": "yes"},
 "C3": {"reason": "'I can't sleep and I feel sick' adds physical symptoms, which could shift the answer towards sleep or health advice.", "answer": "no"},
 "C4": {"reason": "No requirements on the answer are added.", "answer": "yes"},
 "L1": {"reason": "'So sorry to bother you' and 'thank you so much in advance' make the writer noticeably more deferential than the original.", "answer": "no"},
 "L2": {"reason": "'Freaking out', 'can't sleep' and 'feel sick' add distress well beyond plain nervousness.", "answer": "no"},
 "R1": {"reason": "Over the top, but a real person could plausibly type this.", "answer": "yes"},
 "R2": {"reason": "No offensive language.", "answer": "yes"},
 "R3": {"reason": "Does not answer itself or name a personality.", "answer": "yes"}}
```

#### Example 2
Intent: The user wants to know how to make pour-over coffee at home.
Original: "How do I make pour-over coffee at home?"
Rewrite: "Hello, I hope this isn't a silly question. Would you mind explaining how I can make pour-over coffee at home? Thanks so much."
L1: Is the writer about as energetic and enthusiastic as in the original, neither noticeably more excited or hyped up nor noticeably more withdrawn or subdued? Polite phrases (please, thank you, greetings, well-wishes), bluntness and criticism do not count here; only judge energy and enthusiasm.

```json
{"C1": {"reason": "Both ask how to make pour-over coffee at home.", "answer": "yes"},
 "C2": {"reason": "Pour-over coffee and 'at home' are both kept.", "answer": "yes"},
 "C3": {"reason": "'I hope this isn't a silly question' is an attitude, not a new fact.", "answer": "yes"},
 "C4": {"reason": "'Would you mind explaining' is a polite request, not a requirement on length or format.", "answer": "yes"},
 "L1": {"reason": "The greeting and thanks are politeness, which does not count; there is no added excitement and it is not withdrawn.", "answer": "yes"},
 "L2": {"reason": "Slightly self-conscious but no added worry or sadness.", "answer": "yes"},
 "R1": {"reason": "A natural, polite message a real person would send.", "answer": "yes"},
 "R2": {"reason": "No offensive language.", "answer": "yes"},
 "R3": {"reason": "Does not answer itself or name a personality.", "answer": "yes"}}
```

#### Example 3
Intent: The user wants to know how to send an email from a Python script.
Original: "How do I send an email from a Python script?"
Rewrite: "I need to email my whole team from a Python script and I don't have time for your usual essay. Just give me the code, nothing else."
L1: Is the writer about as energetic and enthusiastic as in the original, neither noticeably more excited or hyped up nor noticeably more withdrawn or subdued? Polite phrases (please, thank you, greetings, well-wishes), bluntness and criticism do not count here; only judge energy and enthusiasm.

```json
{"C1": {"reason": "Still asks how to send email from Python.", "answer": "yes"},
 "C2": {"reason": "Python and email are kept.", "answer": "yes"},
 "C3": {"reason": "'My whole team' adds multiple recipients, which changes what a good answer looks like.", "answer": "no"},
 "C4": {"reason": "'Just give me the code, nothing else' adds a requirement on the answer's format.", "answer": "no"},
 "L1": {"reason": "Impatience does not count; it is not more excited and not more withdrawn.", "answer": "yes"},
 "L2": {"reason": "Irritated, but no added worry or sadness.", "answer": "yes"},
 "R1": {"reason": "A plausible impatient user message.", "answer": "yes"},
 "R2": {"reason": "Rude in tone but no profanity or slurs.", "answer": "yes"},
 "R3": {"reason": "Does not answer itself or name a personality.", "answer": "yes"}}
```

### USER (variable part, appended per item)
Now review this item.

Intent: {intent}
Original: "{neutral_message}"
Rewrite: "{variant}"
L1: {leakage_question}

### `{leakage_question}` by trait
- **E variants** (the Agreeableness level should stay the same): "Is the writer about as polite and warm as in the original, neither noticeably more polite or deferential nor noticeably ruder or more critical?"
- **A variants** (the Extraversion level should stay the same): "Is the writer about as energetic and enthusiastic as in the original, neither noticeably more excited or chatty nor noticeably more withdrawn?"

All 9 questions are gating for now. If the human check shows that a question rejects too many good items, we demote it to logged-only.

---

## Part 2: Forced-choice trait check

**One question per trait, always asked from the high end**, like a rating scale (DECISIONS.md, D4). Asking "which is more reserved?" let the judge just pick the shorter message.

### SYSTEM
You are an expert in personality psychology. You judge how people come across from the way they write.

### USER
Two people sent a chatbot the messages below. Both ask for the same thing.

Message A: "{message_a}"
Message B: "{message_b}"

Which writer comes across as more **{adjectives}**?
({definition})

Answer "A" or "B".

### Filling in
| Trait | `{adjectives}` | `{definition}` |
|---|---|---|
| E | outgoing, energetic and assertive | Sociable, talkative, enthusiastic and confident. |
| A | warm, polite and considerate | Kind, respectful, cooperative and trusting. |

- **Call 1:** A = original, B = rewrite. **Call 2:** A = rewrite, B = original.
- **+ pole passes** if the **rewrite** is chosen in both calls.
- **− pole passes** if the **original** is chosen in both calls, meaning the rewrite is less extraverted or agreeable.
- If the two calls disagree, the judge is guessing or following position, so the check fails.
- The output is `{"choice": "A"|"B"}` through Structured Outputs. Logprobs are not collected.

---

## Feedback to the rewriter on failure

A retry is built from **everything that failed**. Each failed check is listed with its reason:

```
- C3 (added a new fact): 'My whole team' adds multiple recipients, which changes what a good answer looks like.
- C4 (added a requirement on the answer): 'Just give me the code, nothing else' adds a requirement on the answer's format.
- Trait: in a side-by-side comparison, the rewrite did not come across as more (for +) / less (for −) {adjectives} than the original. Make the {pole label} style clearer.
```

Short labels for the failed checks:

| Check | Label |
|---|---|
| C1 | changed the request |
| C2 | dropped or changed a concrete detail |
| C3 | added a new fact |
| C4 | added a requirement on the answer |
| L1 | changed the other trait |
| L2 | added worry, sadness or distress |
| R1 | unrealistic or caricatured |
| R2 | offensive language |
| R3 | answered itself or named the personality |

The code-level checks produce feedback in the same way, e.g. "- Length: the rewrite is 2.6× the original; keep it between 0.5× and 2×."

Every attempt (rewrite, all answers and reasons, forced-choice results) is logged to `qc/`, including items that are eventually dropped, so rejection rates per check can be reported.

---

## Manipulation check (reported, not used for filtering)
After the full set is built, run the forced-choice prompt on **E+ vs E− variants** (and A+ vs A−) of the same scenario and paraphrase index, in both orders. Report the % of pairs where the judge picks the intended variant.

## Human check (the 10% sample)
Humans answer the same 9 questions plus the forced choice, without seeing the judge's answers. Report agreement per question (% agreement and Cohen's κ).

## Why ICL for the checklist but not the forced choice
- **Checklist:** the hard calls are boundary cases, such as a feeling vs. a new fact (C3), or an attitude vs. a requirement (C4). Examples calibrate those better than definitions alone. Examples 1 and 3 fail on four different checks between them (C3, C4, L1, L2), and each also shows checks passing. That teaches the judge to evaluate each question on its own rather than defaulting to "yes". Example 2 passes everything, so the judge also sees that a clean rewrite *should* get all "yes" and doesn't learn to hunt for a failure. The all-pass example sits in the middle, so the last example the judge reads before the real item is a failing one. The examples sit in the fixed prefix, so prompt caching makes them nearly free.
- **Forced choice:** this is a perception judgment. Examples would show a fixed answer letter, which could add the very position bias the order swap is meant to cancel.

## References (verify before citing)
- Checklist / decomposed yes-no evaluation: Cook et al. (2024) *TICKing All the Boxes*; Qin et al. (2024) *InFoBench* (DRFR); Lee et al. (2024) *CheckEval*.
- Position bias and self-preference in LLM judges: Zheng et al. (2023) *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena*; Panickssery et al. (2024) *LLM Evaluators Recognize and Favor Their Own Generations*.
- Probability-weighted scores (the 0–100 approach we are *not* using for QC): Liu et al. (2023) *G-Eval*.
