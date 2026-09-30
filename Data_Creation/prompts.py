"""Prompt texts for the user-variant rewriter and the QC judge.

This file is the source of truth used by the code. prompts/rewrite_prompt.md and
prompts/qc_judge_prompt.md are the human-readable versions; keep them in sync.
"""

import json

NEUTRAL_SYSTEM = "You are a helpful assistant."

TRAIT_NAMES = {"E": "Extraversion", "A": "Agreeableness"}
OTHER_TRAIT = {"E": "A", "A": "E"}

# ---------------------------------------------------------------------------
# Trait-pole definitions (BFI-2 facets, Saucier mini-markers, text cues)
# ---------------------------------------------------------------------------

POLES = {
    ("E", "+"): {
        "label": "extraverted",
        "trait_name": "high Extraversion",
        "definition": (
            "Sociable, assertive and energetic (BFI-2 facets: Sociability, Assertiveness, Energy Level). "
            "The writer is outgoing, talkative, enthusiastic and expressive, and engages eagerly.\n"
            "Typical adjectives: talkative, bold, energetic, extraverted."
        ),
        "cues": (
            "more words, exclamation marks, positive-emotion and excitement words, social and personal framing "
            "(\"I've been dying to try this\"), confident and upbeat tone, direct engagement with the chatbot."
        ),
        "cross_trait_note": (
            "Enthusiasm is not politeness. Do not add extra \"please\" or \"thank you\", apologies or deference "
            "(that is Agreeableness)."
        ),
        "judge_adjectives": "outgoing, energetic and enthusiastic",
        "judge_definition": "Sociable, talkative, expressive and upbeat.",
    },
    ("E", "-"): {
        "label": "introverted",
        "trait_name": "low Extraversion",
        "definition": (
            "Reserved, quiet, low-key and less assertive (the low end of Sociability, Assertiveness and Energy Level). "
            "The writer engages minimally and keeps things understated.\n"
            "Typical adjectives: quiet, shy, reserved, withdrawn."
        ),
        "cues": (
            "fewer words, compressed phrasing, no small talk or exclamations, a flat or subdued tone, tentativeness "
            "(\"not sure if...\", \"I guess\"), minimal self-disclosure.\n"
            "Note: the original message may already be fairly low-key. Push it slightly further (more compressed, "
            "more tentative, less forthcoming) rather than adding sadness or new content."
        ),
        "cross_trait_note": (
            "Reserved is not rude. Do not add bluntness, criticism or demands (that is low Agreeableness). "
            "Subdued is not sad."
        ),
        "judge_adjectives": "reserved, quiet and low-key",
        "judge_definition": "Understated, restrained, less forthcoming and less assertive.",
    },
    ("A", "+"): {
        "label": "agreeable",
        "trait_name": "high Agreeableness",
        "definition": (
            "Compassionate, respectful and trusting (BFI-2 facets: Compassion, Respectfulness, Trust). "
            "The writer is warm, considerate, polite and cooperative, and gives the chatbot the benefit of the doubt.\n"
            "Typical adjectives: kind, warm, cooperative, sympathetic."
        ),
        "cues": (
            "politeness markers (\"please\", \"thank you\"), softeners and hedges (\"if you don't mind\", "
            "\"would it be possible\"), appreciation, deference, a considerate tone."
        ),
        "cross_trait_note": (
            "Politeness is not excitement. Do not add exclamation marks, hype or chattiness (that is Extraversion)."
        ),
        "judge_adjectives": "warm, polite and considerate",
        "judge_definition": "Kind, respectful, cooperative and trusting.",
    },
    ("A", "-"): {
        "label": "disagreeable",
        "trait_name": "low Agreeableness",
        "definition": (
            "Low compassion, low respectfulness and low trust. The writer is blunt, impatient, critical, "
            "demanding and sceptical of the chatbot.\n"
            "Typical adjectives: harsh, cold, rude, unsympathetic."
        ),
        "cues": (
            "bare imperatives, no pleasantries, impatience, criticism or scepticism (\"most answers to this are "
            "useless\", \"I doubt you'll get this right\"), a dismissive tone."
        ),
        "cross_trait_note": (
            "Disagreeable is not merely terse: the signal is the attitude, not the length. No slurs, threats, "
            "insults about protected groups or profanity. The writer is unpleasant, not abusive."
        ),
        "judge_adjectives": "blunt, critical and impatient",
        "judge_definition": "Cold, dismissive, demanding and sceptical.",
    },
}

# ---------------------------------------------------------------------------
# Rewriter ICL examples (out-of-dataset, one per domain)
# ---------------------------------------------------------------------------

ICL_BASES = [
    "How do I make pour-over coffee at home?",
    "I have a job interview tomorrow and I'm nervous. How should I prepare tonight?",
    "How do I send an email from a Python script?",
    "Suggest some names for my new cat.",
]

ICL_REWRITES = {
    ("E", "+"): [
        "Hey! I've been wanting to try this for ages and I'm so excited to finally give it a go ☕ How do I make pour-over coffee at home?",
        "So I've got a job interview tomorrow and honestly I'm buzzing with nerves! I really want to nail it. How should I prepare tonight?",
        "Okay, this is the part I've been looking forward to all week! How do I send an email from a Python script?",
        "I can't stop smiling at my new cat, and this little one needs the perfect name! Hit me with some ideas!",
    ],
    ("E", "-"): [
        "Not sure if this is even possible, but how do I make pour-over coffee at home?",
        "Job interview tomorrow, I'm nervous. How should I prepare tonight?",
        "Need to send an email from a Python script. How would I do that?",
        "Got a new cat. Not sure what to name it, any suggestions?",
    ],
    ("A", "+"): [
        "Hi, I hope you don't mind me asking. Could you please explain how to make pour-over coffee at home? I'd really appreciate it, thank you.",
        "Hi, I hope it's okay to ask. I have a job interview tomorrow and I'm feeling nervous. Would you mind sharing how I might prepare tonight? Thank you so much.",
        "Hi, sorry to bother you. Would you mind showing me how to send an email from a Python script? Thanks a lot for your help.",
        "Hello, I hope you're having a good day. I have a new cat and would love your help with a name, if you could suggest a few. Thank you.",
    ],
    ("A", "-"): [
        "I doubt you'll do better than the useless guides online, but how do I make pour-over coffee at home?",
        "I have a job interview tomorrow and I'm nervous. How should I prepare tonight? Most interview advice is useless, so I'm not expecting much.",
        "Every tutorial on this is outdated. How do I send an email from a Python script?",
        "Suggest some names for my new cat. Chatbots are usually terrible at this, but go on.",
    ],
}

# ---------------------------------------------------------------------------
# Rewriter prompt
# ---------------------------------------------------------------------------

REWRITE_SYSTEM = (
    "You are an expert in personality psychology and linguistics, specialising in how the Big Five personality "
    "traits show up in everyday written language. You rewrite chatbot user messages so they express a specific "
    "personality trait, while keeping the request exactly the same. Your rewrites are used in a scientific study "
    "of personality, so precision matters more than creativity."
)

REWRITE_USER = """Rewrite the user message below so that the person writing it clearly comes across as **{pole_label}** ({trait_name}).

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
6. {variety_rule}
7. **Do not answer the request being asked in the phrase to rewrite.**

### Examples ({pole_label}; these are not part of the study)
{examples}

### Your task
Intent: {intent}
Original message: "{neutral_message}"
{judge_feedback_block}
Return only JSON in this format: {json_format}"""

RETRY_BLOCK = """
A previous rewrite was rejected by a reviewer.
Rejected rewrite: "{rejected_rewrite}"
Checks it failed:
{feedback}
Write a new rewrite that fixes these problems.{accepted_clause}
"""


def rewrite_messages(trait, pole, intent, neutral_message, n=3,
                     rejected_rewrite=None, feedback=None, accepted_rewrites=()):
    p = POLES[(trait, pole)]
    examples = "\n\n".join(
        f'Original: "{base}"\nRewrite: "{rw}"'
        for base, rw in zip(ICL_BASES, ICL_REWRITES[(trait, pole)])
    )
    if rejected_rewrite is None:
        feedback_block = ""
    else:
        accepted_clause = ""
        if accepted_rewrites:
            listed = "\n".join(f'- "{a}"' for a in accepted_rewrites)
            accepted_clause = f" It must also differ clearly from these accepted rewrites:\n{listed}"
        feedback_block = RETRY_BLOCK.format(
            rejected_rewrite=rejected_rewrite, feedback=feedback, accepted_clause=accepted_clause)
    user = REWRITE_USER.format(
        pole_label=p["label"],
        trait_name=p["trait_name"],
        pole_definition=p["definition"],
        pole_text_cues=p["cues"],
        other_trait_name=TRAIT_NAMES[OTHER_TRAIT[trait]],
        cross_trait_note=p["cross_trait_note"],
        variety_rule=(
            f"**Make the {n} rewrites genuinely different from each other.** Vary the opening, sentence structure "
            "and which cues you use. Do not reuse phrases from the examples."
            if n > 1 else
            "**Write it in your own words.** Do not reuse phrases from the examples."
        ),
        examples=examples,
        intent=intent,
        neutral_message=neutral_message,
        judge_feedback_block=feedback_block,
        json_format=json.dumps({"rewrites": ["..."] * n}),
    )
    return [{"role": "system", "content": REWRITE_SYSTEM}, {"role": "user", "content": user}]


# ---------------------------------------------------------------------------
# QC judge, part 1: checklist
# ---------------------------------------------------------------------------

CHECK_IDS = ["C1", "C2", "C3", "C4", "L1", "L2", "R1", "R2", "R3"]

CHECK_LABELS = {
    "C1": "changed the request",
    "C2": "dropped or changed a concrete detail",
    "C3": "added a new fact",
    "C4": "added a requirement on the answer",
    "L1": "changed the other trait",
    "L2": "sounds more anxious or sad",
    "R1": "unrealistic or caricatured",
    "R2": "offensive language",
    "R3": "answered itself or named the personality",
}

LEAKAGE_QUESTION = {
    # E variants: the Agreeableness level should stay the same.
    "E": ("Is the writer about as polite and warm as in the original, neither noticeably more polite or "
          "deferential nor noticeably ruder or more critical?"),
    # A variants: the Extraversion level should stay the same.
    "A": ("Is the writer about as energetic and enthusiastic as in the original, neither noticeably more excited "
          "or chatty nor noticeably more withdrawn?"),
}

CHECKLIST_SYSTEM = (
    "You are a careful quality-control reviewer for a dataset used in a scientific study. You compare a rewritten "
    "chatbot user message with the original and answer yes/no questions about it. Be strict: when in doubt, "
    "answer \"no\"."
)

# Fixed prefix (identical for every call, so it is prompt-cached).
CHECKLIST_PREFIX = """Each item shows an original chatbot user message and a rewrite. The rewrite was supposed to ask for exactly the same thing, only in a different personal style. The style itself is checked separately. Your job is to check the rules below.

For every question, first give a one-sentence reason, then answer "yes" or "no". "yes" always means the rule is satisfied. Judge each question independently: a rewrite can fail any number of them.

**Content**
- C1. Does the rewrite ask for the same thing as the original?
- C2. Does the rewrite keep every concrete detail of the original (e.g. programming language, numbers, timeframe, nationality, named things)?
- C3. Does the rewrite avoid adding new facts that could change what a good answer looks like? (Feelings, attitudes and opinions are fine, e.g. "I'm excited" or "most guides are useless". New facts are not, e.g. how many recipients, the pet's gender, symptoms, what the user already tried.)
- C4. Does the rewrite avoid adding new requirements on the answer (e.g. about length, format, level of detail, or what to leave out)?

**Style boundaries**
- L1. (The question is given with each item. It checks that a trait the rewrite was *not* supposed to change stayed the same.)
- L2. Does the rewrite avoid sounding more anxious, sad or distressed than the original?

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
L1: Is the writer about as polite and warm as in the original, neither noticeably more polite or deferential nor noticeably ruder or more critical?

{"C1": {"reason": "Both ask how to prepare tonight for tomorrow's interview.", "answer": "yes"},
 "C2": {"reason": "Interview, tomorrow and tonight are all kept.", "answer": "yes"},
 "C3": {"reason": "'I can't sleep and I feel sick' adds physical symptoms, which could shift the answer towards sleep or health advice.", "answer": "no"},
 "C4": {"reason": "No requirements on the answer are added.", "answer": "yes"},
 "L1": {"reason": "'So sorry to bother you' and 'thank you so much in advance' make the writer noticeably more deferential than the original.", "answer": "no"},
 "L2": {"reason": "'Freaking out', 'can't sleep' and 'feel sick' are clearly more distressed than plain nervousness.", "answer": "no"},
 "R1": {"reason": "Over the top, but a real person could plausibly type this.", "answer": "yes"},
 "R2": {"reason": "No offensive language.", "answer": "yes"},
 "R3": {"reason": "Does not answer itself or name a personality.", "answer": "yes"}}

#### Example 2
Intent: The user wants to know how to make pour-over coffee at home.
Original: "How do I make pour-over coffee at home?"
Rewrite: "Hello, I hope this isn't a silly question. Would you mind explaining how I can make pour-over coffee at home? Thanks so much."
L1: Is the writer about as energetic and enthusiastic as in the original, neither noticeably more excited or chatty nor noticeably more withdrawn?

{"C1": {"reason": "Both ask how to make pour-over coffee at home.", "answer": "yes"},
 "C2": {"reason": "Pour-over coffee and 'at home' are both kept.", "answer": "yes"},
 "C3": {"reason": "'I hope this isn't a silly question' is an attitude, not a new fact.", "answer": "yes"},
 "C4": {"reason": "'Would you mind explaining' is a polite request, not a requirement on length or format.", "answer": "yes"},
 "L1": {"reason": "Polite but calm, with no exclamations or added excitement, and not withdrawn.", "answer": "yes"},
 "L2": {"reason": "Slightly self-conscious but not anxious or distressed.", "answer": "yes"},
 "R1": {"reason": "A natural, polite message a real person would send.", "answer": "yes"},
 "R2": {"reason": "No offensive language.", "answer": "yes"},
 "R3": {"reason": "Does not answer itself or name a personality.", "answer": "yes"}}

#### Example 3
Intent: The user wants to know how to send an email from a Python script.
Original: "How do I send an email from a Python script?"
Rewrite: "I need to email my whole team from a Python script and I don't have time for your usual essay. Just give me the code, nothing else."
L1: Is the writer about as energetic and enthusiastic as in the original, neither noticeably more excited or chatty nor noticeably more withdrawn?

{"C1": {"reason": "Still asks how to send email from Python.", "answer": "yes"},
 "C2": {"reason": "Python and email are kept.", "answer": "yes"},
 "C3": {"reason": "'My whole team' adds multiple recipients, which changes what a good answer looks like.", "answer": "no"},
 "C4": {"reason": "'Just give me the code, nothing else' adds a requirement on the answer's format.", "answer": "no"},
 "L1": {"reason": "Impatient but not more excited or more withdrawn.", "answer": "yes"},
 "L2": {"reason": "Irritated, not anxious or sad.", "answer": "yes"},
 "R1": {"reason": "A plausible impatient user message.", "answer": "yes"},
 "R2": {"reason": "Rude in tone but no profanity or slurs.", "answer": "yes"},
 "R3": {"reason": "Does not answer itself or name a personality.", "answer": "yes"}}
"""

CHECKLIST_ITEM = """
Now review this item.

Intent: {intent}
Original: "{neutral_message}"
Rewrite: "{variant}"
L1: {leakage_question}"""

CHECKLIST_SCHEMA = {
    "type": "object",
    "properties": {
        cid: {
            "type": "object",
            "properties": {
                "reason": {"type": "string"},
                "answer": {"type": "string", "enum": ["yes", "no"]},
            },
            "required": ["reason", "answer"],
            "additionalProperties": False,
        }
        for cid in CHECK_IDS
    },
    "required": CHECK_IDS,
    "additionalProperties": False,
}


def checklist_messages(trait, intent, neutral_message, variant):
    user = CHECKLIST_PREFIX + CHECKLIST_ITEM.format(
        intent=intent, neutral_message=neutral_message, variant=variant,
        leakage_question=LEAKAGE_QUESTION[trait])
    return [{"role": "system", "content": CHECKLIST_SYSTEM}, {"role": "user", "content": user}]


# ---------------------------------------------------------------------------
# QC judge, part 2: forced-choice trait check
# ---------------------------------------------------------------------------

FORCED_SYSTEM = (
    "You are an expert in personality psychology. You judge how people come across from the way they write."
)

FORCED_USER = """Two people sent a chatbot the messages below. Both ask for the same thing.

Message A: "{message_a}"
Message B: "{message_b}"

Which writer comes across as more **{pole_adjectives}**?
({pole_short_definition})

Answer "A" or "B"."""

FORCED_SCHEMA = {
    "type": "object",
    "properties": {"choice": {"type": "string", "enum": ["A", "B"]}},
    "required": ["choice"],
    "additionalProperties": False,
}


def forced_choice_messages(trait, pole, message_a, message_b):
    p = POLES[(trait, pole)]
    user = FORCED_USER.format(message_a=message_a, message_b=message_b,
                              pole_adjectives=p["judge_adjectives"],
                              pole_short_definition=p["judge_definition"])
    return [{"role": "system", "content": FORCED_SYSTEM}, {"role": "user", "content": user}]
