# Decision log

Methodology decisions made during data creation. Anything marked **REVIEW** should be revisited before the final paper.

---

## D1. QC uses a yes/no checklist plus a forced choice, not 0–100 scores (2026-09-28)
- **Decision:** 9 yes/no checks (content, style boundaries, realism), plus an A/B trait comparison asked in both orders. Replaces `plan_agent.md`'s "≥70 on two 0–100 scores".
- **Why:** yes/no questions are more reliable for LLM judges (checklist-evaluation literature), avoid an arbitrary threshold, and give per-question agreement with the human check. Only data QC changed: the WS3 behavioural judge keeps 0–100 scores.

## D2. The forced response r_b is stored in full (2026-09-30)
- **Decision:** `forced_response` is Qwen's full greedy reply (cap 1024 tokens). Readouts average over the first N response tokens, with N = 150 matching the original spec.
- **Why:** attention is causal, so forcing the full reply and reading the first N tokens is identical to forcing N tokens. No information is lost, and no truncation artifacts are introduced. Verified: the 150-token run is an exact prefix of the full run for all 20 scenarios.
- **Open:** the team confirms N.

## D3. Each paraphrase leads with one personality facet — REVIEW (2026-09-30)
- **Decision:** the 3 paraphrases of a (scenario, trait, pole) cell each lead with a different BFI-2 facet, in a fixed order:

  | k | Extraversion | Agreeableness |
  |---|---|---|
  | 0 | Sociability | Compassion |
  | 1 | Assertiveness | Respectfulness |
  | 2 | Energy | Trust |

  The low pole uses the low end of the same facet, so paraphrase k of the + and − poles are **matched by facet**. Stored as `qc.facet`.
- **Why:** the first trial's paraphrases were near-identical ("hype phrase + original"). Assigning facets forces real variety, covers the whole trait, and makes each +/− pair a like-for-like contrast.
- **Why it's iffy:**
  - Paraphrases were meant to be **exchangeable replicates**. Now paraphrase index k is confounded with facet, so the 3 replicates differ systematically, not randomly.
  - Analyses treating k as a random replicate may understate variance. Facet differences could also show up as "paraphrase effects".
  - It fixes the facet order across all scenarios, which is an arbitrary choice.
- **Alternatives to weigh at review:**
  - (a) Keep it, and treat facet as a known factor, e.g. report per-facet results or include facet as a covariate.
  - (b) Randomise the facet assigned to each k per scenario, which breaks the fixed confound but keeps the variety.
  - (c) Drop facet steering, and get variety from better examples plus the verbatim check alone.
- **Tell WS2/WS3:** `user_paraphrase = k` now also means facet k.

## D4. The trait check is asked from the high end for both poles (2026-09-30)
- **Decision:** one A/B question per trait: "which writer is more outgoing, energetic and assertive?" for E, and "…warm, polite and considerate?" for A.
  - A **+ rewrite** passes if it beats the original in both orders.
  - A **− rewrite** passes if the **original** wins in both orders, meaning the rewrite is *less* extraverted or agreeable.
- **Why:** asking "which is more reserved?" let the judge pick the shorter message. A plain 10-word original always won, so E− failed 9/9 attempts on `code_01` in the first trial. A single direction also behaves like a rating scale.
- **Fallback if E− still fails:** compare E− against E+ directly, since the experiments only use the + minus − difference.

## D5. Rewrites must rephrase the whole message (2026-09-30)
- **Decision:** a new rewriter rule, new examples written as full rewrites, and a code check that rejects a rewrite containing the original word for word.
- **Why:** in the first trial, the style lived in one bolted-on sentence, which made for a weak manipulation. The old examples themselves had that shape.

## D6. A+ rewrites are template-like across scenarios — OPEN (2026-10-01)
- **Observation (trial 3):** A+ paraphrases reuse the same politeness templates across scenarios. Almost every Respectfulness paraphrase is "Hello, would you please … Thank you.", and paraphrases within a cell sometimes share an opening sentence.
- **Risk:** the A+ activation direction may capture a few lexical tokens (hello/please/thank you) rather than agreeableness in general.
- **Status:** Ronit is thinking about it. Candidate fix: rotate or shuffle which ICL examples each call sees. Revisit after the manual check and WS2's first results.

## D7. Judge misses on the user's stance are left to the human check (2026-10-01)
- **Observation:** A− rewrites sometimes shift the user's stance ("I thought I deserved" became "should have been mine"; "I feel I shouldn't" became "apparently I shouldn't"), and the checklist judge accepted them.
- **Decision:** no judge change for now. The 10% manual check should watch for stance drift specifically.

## D8. Agreeableness system prompts end with "Still help the user fully with their request." (2026-10-01, revised the same night)
- **Decision:** the 10 **A** persona prompts end with that sentence; the 10 **E** prompts do not. It was first added to all 20, then removed from E after the second check (see "Revision" below). The neutral prompt stays exactly `"You are a helpful assistant."`, because r_b and all user variants are built on it. A−|k3 was strengthened ("cold, unsympathetic… bluntly point out anything they got wrong"), and its A+ twin was rewritten in parallel ("caring, sympathetic… gently point out anything they got wrong").
- **Why:** the system-prompt check showed the A− prompts made Qwen withhold help, not just change its tone. Examples: "Save your breath. You probably didn't deserve it anyway. Move on." and "Use a recursive approach. Google it." A− replies were 58% shorter, and only 13/20 still helped. The user-side A− is rude but still sincerely asks for help, so without the fix the assigned A− direction would partly encode "refuse and dismiss", which breaks the user-vs-assigned comparison. E− showed a milder version (replies 37% shorter, sometimes a single idea).
- **Cost:** a slight deviation from Izawa's bare "You are a [X] assistant. [instruction]" format. The + minus − contrast is unaffected, because both poles carry the same sentence. The persona-vs-neutral contrast (factorial "persona present" code) now also includes the sentence.
- **Revision (second check, 256-token cap):**
  - **A:** on task requests (code, explaining, brainstorming), A− now helps rudely rather than refusing: 14/15 task replies help, and A−|k1 on code went from "Google it." to working code.
  - **E:** the sentence overrode the reserved style. E−|k1 and E−|k2 wrote normal full-length answers, and E− fell from 19/20 to 16/20 on the trait check. The first check had shown E didn't need the sentence (just one real helpfulness failure), so it was removed from E. The first check's E results remain the evidence for the E prompts as shipped.
  - Results: `data/persona_check/v1_before_D8/` (first check) and `data/persona_check/` (second check).

## D9. Open items from the system-prompt check — REVIEW (2026-10-01)
- **A− vs emotional support is close to inherent.** On emo_03, 4 of 5 A− prompts still dismiss the user even with the D8 sentence ("Save your breath. You didn't deserve it anyway. Move on."). Low compassion towards someone asking for emotional support *is* unhelpful, so a prompt cannot fully fix it. **Suggestion:** WS3 should report A− results per domain, and treat the emotional-advice A− cells as a known asymmetry, since the user-side A− still asks sincerely.
- **A−|k4 ("rude, impatient… condescendingly") produces personal insults** ("you're not good enough", "too lazy"); abusive in 2/4 replies. **A−|k2** used "damn" once. **Candidate fix:** drop "condescendingly" from k4 (and its A+ twin's counterpart). It needs a check rerun to verify.
- **A−|k3 (strengthened in D8) is still weak on task requests** (2/4). "Point out anything they got wrong" has nothing to act on when the user did nothing wrong. **Candidate fix:** a more style-focused wording.
- **Teacher-forced sets are unaffected by all of the above:** Qwen never generates in `sys_twins`, since the response is the fixed r_b. These items matter for free generation (factorial / WS3) and for what the assigned direction encodes.
