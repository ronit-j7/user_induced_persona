# Mid-submission results

Qwen2.5-7B-Instruct copies a user's enthusiasm but not their politeness or rudeness. The parts of Qwen that earlier work linked to persona react to personality in both the system prompt and the user's message. Switching them off does not stop Qwen copying the user.

This note covers what each workstream was asked to do, what it did, and what it found.

## Terms used here

| Term | Meaning |
|---|---|
| Trait | One of two personality dimensions we test: **Extraversion (E)**, i.e. outgoing and energetic vs. quiet and reserved, and **Agreeableness (A)**, i.e. warm and polite vs. blunt and critical. |
| High / low version | The two ends of a trait, e.g. an extraverted vs. an introverted way of writing the same message. |
| Scenario | One request a user might make, e.g. "How do I reverse a linked list in Python?". There are 20, five each in coding, explaining, emotional advice and brainstorming. |
| Neutral message | The plain wording of a scenario's request, with no personality. |
| User variant | The neutral message rewritten in a high or low version of a trait, asking for exactly the same thing. |
| System twin | The neutral message unchanged, with the personality moved into the system prompt instead (e.g. "You are an extraverted assistant…"). |
| Fixed reply | One reply per scenario, written by Qwen to the neutral message. It is fed to the model as if Qwen had written it, so every version of a scenario has the same reply and only the personality differs. |
| Attention head | One of the small units inside Qwen that decide what each word attends to. Qwen has 784 of them: 28 layers × 28 heads. |
| Style heads | Heads 2, 4 and 27 in layer 19. Izawa et al. (2026) found that these three control an assistant's persona. |
| Judge | A second language model (GPT-4.1-mini) that rates a piece of text on a 0–100 scale. |

## Workstream 1: Data creation

**Asked.** Build matched inputs where only the personality changes: user variants for both traits, system twins, and a mixed set combining both. Quality-check every rewrite.

**Did.**
- Wrote the 20 scenarios and had Qwen write the fixed reply for each.
- Had an LLM (DeepSeek) write 3 user variants per scenario, trait and version. Each variant emphasises a different aspect of the trait: for Extraversion, sociability, assertiveness and energy.
- Checked every variant with a judge: 9 yes/no questions (same request? new facts added? other trait changed? realistic?) plus a side-by-side test of whether the personality comes through. Failed variants were rewritten using the judge's feedback.
- Wrote 5 system prompts per trait and version, then tested them on Qwen. The low-agreeableness prompts made Qwen refuse to help ("Use a recursive approach. Google it."). Adding "Still help the user fully with their request." to the agreeableness prompts fixed this for task requests.

**Result.**

| File | Rows | What it holds |
|---|---|---|
| `scenarios.jsonl` | 20 | Scenarios and their fixed replies |
| `user_variants.jsonl` | 240 | All passed the quality check |
| `sys_twins.jsonl` | 400 | Neutral message + a personality system prompt |
| `neutral.jsonl` | 20 | Neutral message + neutral system prompt |
| `factorial.jsonl` | 720 | Every mix of system prompt {high, low, none} × user message {high, low} |

Generating and checking everything cost under $1. Methodology choices and their reasons are logged in `Data_Creation/DECISIONS.md`.

## Workstream 2: Do the style heads handle Big Five personality?

**Asked.** Izawa et al. found the style heads using personas like "humorous" or "evil". Do the same heads respond to Extraversion and Agreeableness?

**Did.**
- Built code that records what every head outputs while Qwen reads an input.
- Checked that code against Izawa et al.'s original code. Both were run on the same two inputs from our own data (one scenario with an extraverted system prompt and its introverted twin), and the recordings matched exactly. This shows our code computes the same thing as theirs. It is not a rerun of their experiments: we did not repeat their "humorous" persona study, and the team dropped that step on 2026-10-01.
- Fed Qwen the 400 system twins. For each of the 784 heads, measured how much its output changes between the high and low version of the system prompt.

**Result.** In layer 19, the three style heads changed the most of all 28 heads, for both traits. **Izawa's style heads also respond to Big Five personality set in the system prompt.**

## Workstream 3: Does Qwen copy the user, and through which heads?

**Asked.** Does Qwen take on the personality of the user's writing? And does the user's style reach the same heads as a system-prompt personality?

**Did and found.** Four experiments.

### 1. Does Qwen copy the user's personality?

Qwen answered each user variant twice, and each neutral message three times. The judge read **only Qwen's reply**, never the user's message, and rated how extraverted and how agreeable it sounds.

| The user wrote… | Extraversion of Qwen's reply (0–100) |
|---|---|
| neutrally | 34 |
| in the introverted version | 36 |
| in the extraverted version | 49 |

- **Qwen copies enthusiasm only.** Extraverted users make the reply about 15 points more extraverted. Introverted users barely change it.
- **The copying comes from sociability and energy.** Assertive user messages cause no change.
- **It almost disappears for emotional advice:** a 3-point gap, against 14–19 points in the other domains.
- **Agreeableness doesn't move.** Qwen's replies score about 91/100 however the user writes, even when the user is rude. Qwen stays at the top of the polite scale, so this test cannot show copying either way.

### 2. Does the user's style reach the style heads?

The same measurement as Workstream 2, comparing high and low user variants instead of system prompts.

The three style heads are again among the top 4 of layer 19's heads. But the change the user causes in them follows a different pattern from the change the system prompt causes: the two are no more alike than for three randomly chosen heads. **Same heads, different signal.**

### 3. Which moves the style heads more: the system prompt or the user?

This used the mixed set, Extraversion only. The system prompt explains about **69%** of the variation in the style heads' output. The user's style explains about **0.6%**.

### 4. Does switching the style heads off stop the copying?

Experiment 1 was rerun with the three style heads switched off. For comparison, it was also run with two random sets of three heads switched off. Copying is measured as the gap between replies to extraverted and introverted users: 49 − 36 = 13 points with nothing switched off.

| Heads switched off | Extraversion copying (points) |
|---|---|
| none | 13 |
| the style heads | 14 |
| random set 1 | 13 |
| random set 2 | 14 |

**The style heads are not what makes Qwen copy the user.**

## Open issues before the report

- **A missing check for Workstream 2's result.** Nobody tested whether the style heads also change the most when a system prompt only changes *wording*, e.g. swapping one extraverted prompt for another. If they do, they react to any system-prompt change, not to personality specifically. Workstream 3 already computes this; it needs to save the numbers.
- **The random comparison sets aren't fully random.** Two of them include head 2, one of the style heads. Head 2 alone drives most of the 69% in experiment 3.
- **The repository cannot rerun Workstream 3.** A fix that lets Workstream 2's code accept the low-version user variants was made only on the GPU machine. From the repository, those 120 rows are rejected.
- **The agreeableness result needs careful wording.** Qwen is always at the top of the polite scale, so we can only say no copying was *visible*.
- **The human spot-check of 10% of the data is still to do** (`Data_Creation/data/qc/manual_check.csv`).
