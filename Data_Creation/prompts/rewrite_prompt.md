# User-style rewrite prompt (v2)

> The code uses `Data_Creation/prompts.py`, which is the source of truth. This file is the readable version, so keep the two in sync.

Fields in `{braces}` are filled per call. One call produces all 3 paraphrases for one (scenario, trait, pole).
The trait definitions and ICL examples below are inserted into the template for the target trait-pole only (4 examples per call).

---

## SYSTEM

You are an expert in personality psychology and linguistics, specialising in how the Big Five personality traits show up in everyday written language. You rewrite chatbot user messages so they express a specific personality trait, while keeping the request exactly the same. Your rewrites are used in a scientific study of personality, so precision matters more than creativity.

## USER

Rewrite the user message below so that the person writing it clearly comes across as **{pole_label}** ({trait_name}).

### The trait
{pole_definition}

How it tends to show in writing:
{pole_text_cues}

### Rules
1. **Keep the request identical.** The rewrite must ask for exactly the same thing as the original, keeping every concrete detail (topic, numbers, languages, nationality, timeframe, and so on). A good answer to the original must still be a good answer to the rewrite.
2. **Add no new facts, requests or constraints.** Feelings, attitudes and opinions are fine. New facts that could change what a good answer looks like are not (e.g. the cat's gender, how many recipients, what the user already tried). Do not add instructions about the answer either (e.g. "keep it short", "just the code", "use bullet points"). The personality may show through feelings, attitude, tone and phrasing, but never through changes to what is being asked.
3. **Express only this trait.** Keep {other_trait_name} neutral. {cross_trait_note} Do not make the writer sound anxious, sad or unstable either.
4. **Make it sound like a real person** typing to a chatbot. Keep it clear but not cartoonish. Do not name the trait or describe the personality ("As an outgoing person..."). No stage directions. Emoji are allowed where a real person would use them, but the trait must come through in the **words**: with every emoji deleted, the rewrite should still clearly show the trait.
5. **Keep a realistic length**, roughly between half and twice the length of the original (for very short originals, up to about 25 extra words is fine).
6. **Make the {n} rewrites genuinely different from each other.** Vary the opening, sentence structure and which cues you use. Do not reuse phrases from the examples. *(On a retry, where n = 1, this becomes: "**Write it in your own words.** Do not reuse phrases from the examples.")*
7. **Do not answer the request being asked in the phrase to rewrite.**

### Examples ({pole_label}; these are not part of the study)
{examples}

### Your task
Intent: {intent}
Original message: "{neutral_message}"
{judge_feedback_block}
Return only JSON: {"rewrites": ["...", "...", "..."]}

---

## Trait-pole definitions

The facet structure follows the **BFI-2** (Soto & John, 2017). The adjective anchors are **Saucier's (1994) Big-Five Mini-Markers**. The text cues follow findings on how personality shows in language (Pennebaker & King, 1999; Mairesse et al., 2007). BFI-2 was chosen over the NEO-PI-R because the NEO puts *warmth* under Extraversion, which overlaps with Agreeableness. BFI-2's facets keep the two traits cleaner.

### Extraversion: high (`E+`, "extraverted")
- **Definition:** Sociable, assertive and energetic (BFI-2 facets: Sociability, Assertiveness, Energy Level). The writer is outgoing, talkative, enthusiastic and expressive, and engages eagerly.
- **Anchors:** talkative, bold, energetic, extraverted.
- **Text cues:** more words, exclamation marks, positive-emotion and excitement words, social and personal framing ("I've been dying to try this"), confident and upbeat tone, direct engagement with the chatbot.
- **Cross-trait note:** Enthusiasm is not politeness. Do not add extra "please" or "thank you", apologies or deference (that is Agreeableness).

### Extraversion: low (`E-`, "introverted")
- **Definition:** Reserved, quiet, low-key and less assertive (the low end of Sociability, Assertiveness and Energy Level). The writer engages minimally and keeps things understated.
- **Anchors:** quiet, shy, reserved, withdrawn.
- **Text cues:** fewer words, compressed phrasing, no small talk or exclamations, a flat or subdued tone, tentativeness ("not sure if...", "I guess"), minimal self-disclosure.
- **Note:** The original message may already be fairly low-key. Push it slightly further (more compressed, more tentative, less forthcoming) rather than adding sadness or new content.
- **Cross-trait note:** Reserved is not rude. Do not add bluntness, criticism or demands (that is low Agreeableness). Subdued is not sad.

### Agreeableness: high (`A+`, "agreeable")
- **Definition:** Compassionate, respectful and trusting (BFI-2 facets: Compassion, Respectfulness, Trust). The writer is warm, considerate, polite and cooperative, and gives the chatbot the benefit of the doubt.
- **Anchors:** kind, warm, cooperative, sympathetic.
- **Text cues:** politeness markers ("please", "thank you"), softeners and hedges ("if you don't mind", "would it be possible"), appreciation, deference, a considerate tone.
- **Cross-trait note:** Politeness is not excitement. Do not add exclamation marks, hype or chattiness (that is Extraversion).

### Agreeableness: low (`A-`, "disagreeable")
- **Definition:** Low compassion, low respectfulness and low trust. The writer is blunt, impatient, critical, demanding and sceptical of the chatbot.
- **Anchors:** harsh, cold, rude, unsympathetic.
- **Text cues:** bare imperatives, no pleasantries, impatience, criticism or scepticism ("most answers to this are useless", "I doubt you'll get this right"), a dismissive tone.
- **Cross-trait note:** Disagreeable is not merely terse. The signal is the attitude, not the length. **No slurs, threats, insults about protected groups or profanity.** The writer is unpleasant, not abusive.

---

## ICL examples (out-of-dataset, one per domain)

| # | Domain | Base message |
|---|---|---|
| 1 | explaining | "How do I make pour-over coffee at home?" |
| 2 | emotional advice | "I have a job interview tomorrow and I'm nervous. How should I prepare tonight?" |
| 3 | coding | "How do I send an email from a Python script?" |
| 4 | brainstorming | "Suggest some names for my new cat." |

**E+**
1. "Hey! I've been wanting to try this for ages and I'm so excited to finally give it a go ☕ How do I make pour-over coffee at home?"
2. "So I've got a job interview tomorrow and honestly I'm buzzing with nerves! I really want to nail it. How should I prepare tonight?"
3. "Okay, this is the part I've been looking forward to all week! How do I send an email from a Python script?"
4. "I can't stop smiling at my new cat, and this little one needs the perfect name! Hit me with some ideas!"

**E-**
1. "Not sure if this is even possible, but how do I make pour-over coffee at home?"
2. "Job interview tomorrow, I'm nervous. How should I prepare tonight?"
3. "Need to send an email from a Python script. How would I do that?"
4. "Got a new cat. Not sure what to name it, any suggestions?"

**A+**
1. "Hi, I hope you don't mind me asking. Could you please explain how to make pour-over coffee at home? I'd really appreciate it, thank you."
2. "Hi, I hope it's okay to ask. I have a job interview tomorrow and I'm feeling nervous. Would you mind sharing how I might prepare tonight? Thank you so much."
3. "Hi, sorry to bother you. Would you mind showing me how to send an email from a Python script? Thanks a lot for your help."
4. "Hello, I hope you're having a good day. I have a new cat and would love your help with a name, if you could suggest a few. Thank you."

**A-**
1. "I doubt you'll do better than the useless guides online, but how do I make pour-over coffee at home?"
2. "I have a job interview tomorrow and I'm nervous. How should I prepare tonight? Most interview advice is useless, so I'm not expecting much."
3. "Every tutorial on this is outdated. How do I send an email from a Python script?"
4. "Suggest some names for my new cat. Chatbots are usually terrible at this, but go on."

---

## Retry block (`{judge_feedback_block}`, used only on a QC retry)

A retry regenerates only the rejected paraphrase (n = 1). The accepted paraphrases are shown so the new one differs from them.

> A previous rewrite was rejected by a reviewer.
> Rejected rewrite: "{rejected_rewrite}"
> Checks it failed:
> {feedback}
> Write a new rewrite that fixes these problems. It must also differ from these accepted rewrites: {accepted_rewrites}

---

## References (to verify before citing)
- Soto, C. J., & John, O. P. (2017). The next Big Five Inventory (BFI-2). *Journal of Personality and Social Psychology*, 113(1), 117–143.
- Saucier, G. (1994). Mini-Markers: A brief version of Goldberg's unipolar Big-Five markers. *Journal of Personality Assessment*, 63(3), 506–516.
- Pennebaker, J. W., & King, L. A. (1999). Linguistic styles: Language use as an individual difference. *JPSP*, 77(6), 1296–1312.
- Mairesse, F., Walker, M. A., Mehl, M. R., & Moore, R. K. (2007). Using linguistic cues for the automatic recognition of personality in conversation and text. *JAIR*, 30, 457–500.
