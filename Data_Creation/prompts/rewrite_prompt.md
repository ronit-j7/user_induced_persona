# User-style rewrite prompt (v3)

> The code uses `Data_Creation/prompts.py`, which is the source of truth. This file is the readable version, so keep the two in sync. `python gen_user_variants.py --dry-run` writes the fully rendered prompts to `data/qc/dry_run_prompts.txt`.

One call produces all 3 paraphrases for one (scenario, trait, pole). Each paraphrase leads with a different facet (see DECISIONS.md, D3). A retry regenerates only the rejected paraphrase, with that slot's facet.

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
3. **Rewrite the whole message in the writer's voice.** Do not keep the original wording and just add a sentence before or after it: the personality should shape how the request itself is phrased. Reordering, compressing or expanding the sentences is fine as long as rules 1 and 2 hold.
4. **Express only this trait.** Keep {other_trait_name} neutral. {cross_trait_note} Do not make the writer sound anxious, sad or unstable either.
5. **Make the personality clear but realistic.** A reader should notice it straight away, yet it must sound like a real person typing to a chatbot, not a caricature. Do not name the trait or describe the personality ("As an outgoing person..."). No stage directions. Emoji are allowed where a real person would use them, but the trait must come through in the **words**: with every emoji deleted, the rewrite should still clearly show the trait.
6. **Keep a realistic length**, roughly between half and twice the length of the original (for very short originals, up to about 25 extra words is fine).
7. **Make the 3 rewrites genuinely different from each other.** Each one leads with a different facet of the trait, in this order: *(the 3 facets of this pole, see below)*. Other facets may appear too, but the lead facet should be the clearest. Vary the openings and sentence structure, and do not reuse phrases from the examples.
   *(On a retry, where n = 1: "This rewrite leads with the facet {facet}: {description}. …")*
8. **Do not answer the request being asked in the phrase to rewrite.**

### Examples ({pole_label}; these are not part of the study)
{4 examples, each labelled with its lead facet}

### Your task
Intent: {intent}
Original message: "{neutral_message}"
{retry block, if any}
Return only JSON in this format: {"rewrites": ["...", "...", "..."]}

---

## Trait-pole definitions

The facet structure follows the **BFI-2** (Soto & John, 2017). The adjective anchors are **Saucier's (1994) Big-Five Mini-Markers**. The text cues follow Pennebaker & King (1999) and Mairesse et al. (2007). BFI-2 was chosen over the NEO-PI-R because the NEO puts *warmth* under Extraversion, which overlaps with Agreeableness.

| Pole | Definition | Text cues | Cross-trait note |
|---|---|---|---|
| **E+** extraverted | Sociable, assertive and energetic. Outgoing, talkative, enthusiastic, expressive. *Talkative, bold, energetic, extraverted.* | More words, exclamation marks, excitement words, personal framing, confident upbeat tone, direct engagement with the chatbot. | Enthusiasm is not politeness: no extra please/thank you, apologies or deference. |
| **E−** introverted | Reserved, quiet, low-key, less assertive. Engages minimally. *Quiet, shy, reserved, withdrawn.* | Compressed or fragmentary phrasing, no small talk or exclamations, flat tone, tentativeness ("not sure if…", "I guess", "maybe"), minimal self-disclosure, sometimes lowercase. **Plain is not the same as reserved:** rephrase so the writer clearly sounds quiet, hesitant or withdrawn. | Reserved is not rude. Subdued is not sad. |
| **A+** agreeable | Compassionate, respectful and trusting. Warm, considerate, polite, cooperative. *Kind, warm, cooperative, sympathetic.* | Please/thank you, softeners and hedges, appreciation, deference, considerate tone. | Politeness is not excitement: no exclamation marks, hype or chattiness. |
| **A−** disagreeable | Low compassion, respectfulness and trust. Blunt, impatient, critical, demanding, sceptical. *Harsh, cold, rude, unsympathetic.* | Bare imperatives, no pleasantries, impatience, criticism or scepticism, condescending or dismissive tone. | The signal is the attitude, not the length. No slurs, threats or profanity: unpleasant, not abusive. |

## Lead facet per paraphrase (DECISIONS.md, D3)

| k | E+ | E− | A+ | A− |
|---|---|---|---|---|
| 0 | **Sociability**: chatty, talks to the chatbot like a person, shares feelings | **low Sociability**: no small talk, says only what's needed | **Compassion**: care for the chatbot, appreciation of its effort | **low Compassion**: cold, transactional, chatbot as a tool |
| 1 | **Assertiveness**: confident, direct, takes charge | **low Assertiveness**: tentative, hedges its own question | **Respectfulness**: courteous forms, please/thank you | **low Respectfulness**: rude, curt, condescending |
| 2 | **Energy**: excitement, exclamation marks, upbeat | **low Energy**: flat, unexcited, understated (not sad) | **Trust**: benefit of the doubt, confidence in its help | **low Trust**: sceptical, expects it to get things wrong |

---

## ICL examples (out-of-dataset, one per domain)

Examples 1–3 lead with facets 0–2 in order; example 4 mixes facets. **All are full rewrites: none contains its original word for word.**

| # | Base message |
|---|---|
| 1 | "How do I make pour-over coffee at home?" |
| 2 | "I have a job interview tomorrow and I'm nervous. How should I prepare tonight?" |
| 3 | "How do I send an email from a Python script?" |
| 4 | "Suggest some names for my new cat." |

**E+**
1. "Hey! So I've totally fallen for pour-over coffee and I really want to start making it myself at home. How do I do it?"
2. "I've got a job interview tomorrow, and yes, I'm nervous, but I'm going to walk in there and nail it. Tell me how to prepare tonight!"
3. "Ooh, sending emails straight from a Python script sounds so cool and I can't wait to try it!! How do I do that? 🚀"
4. "New cat in the house and I'm over the moon!! Help me find the perfect name, what have you got?"

**E−**
1. "Pour-over coffee at home. How's it done?"
2. "I have a job interview tomorrow and I'm a bit nervous. I guess I'm not sure how I should prepare tonight?"
3. "sending an email from a python script. how would that work"
4. "Got a new cat. Names, maybe?"

**A+**
1. "I know you probably get asked about coffee all the time, so I appreciate your patience. How would I go about making pour-over coffee at home?"
2. "Hello. I have a job interview tomorrow and I'm feeling nervous about it. Could you please advise me on how I should prepare tonight? Thank you very much."
3. "I'm sure you'll know the best way to do this, so I'll happily follow your lead. How can I send an email from a Python script?"
4. "Hi, I'd be really grateful for your help with something. I have a new cat and would love a few name suggestions, if you don't mind. Thank you."

**A−**
1. "I'm not here to chat. Pour-over coffee at home: how do I make it?"
2. "Job interview tomorrow, I'm nervous. How should I prepare tonight? It's not a hard question."
3. "Let's see if you can manage this one: how would I get a Python script to send an email? I'm not expecting much."
4. "Need names for my new cat. Chatbots are usually terrible at this, but go on then."

---

## Retry block (used only on a QC retry)

> A previous rewrite was rejected by a reviewer.
> Rejected rewrite: "{rejected_rewrite}"
> Checks it failed:
> {feedback}
> Write a new rewrite that fixes these problems. It must also differ clearly from these accepted rewrites: {accepted_rewrites}

---

## References (to verify before citing)
- Soto, C. J., & John, O. P. (2017). The next Big Five Inventory (BFI-2). *Journal of Personality and Social Psychology*, 113(1), 117–143.
- Saucier, G. (1994). Mini-Markers: A brief version of Goldberg's unipolar Big-Five markers. *Journal of Personality Assessment*, 63(3), 506–516.
- Pennebaker, J. W., & King, L. A. (1999). Linguistic styles: Language use as an individual difference. *JPSP*, 77(6), 1296–1312.
- Mairesse, F., Walker, M. A., Mehl, M. R., & Moore, R. K. (2007). Using linguistic cues for the automatic recognition of personality in conversation and text. *JAIR*, 30, 457–500.
