"""Prompt texts for the user-variant rewriter and the QC judge.

This file is the source of truth used by the code. prompts/rewrite_prompt.md and
prompts/qc_judge_prompt.md are the human-readable versions; keep them in sync.
Run `python gen_user_variants.py --dry-run` to see the fully rendered prompts.
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
            "compressed or fragmentary phrasing, no small talk or exclamations, a flat or subdued tone, tentativeness "
            "(\"not sure if...\", \"I guess\", \"maybe\"), minimal self-disclosure, sometimes lowercase.\n"
            "Note: the original message may already be fairly plain. Plain is not the same as reserved: rephrase it so "
            "the writer clearly sounds quiet, hesitant or withdrawn, rather than adding sadness or new content."
        ),
        "cross_trait_note": (
            "Reserved is not rude. Do not add bluntness, criticism or demands (that is low Agreeableness). "
            "Subdued is not sad."
        ),
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
            "useless\", \"I doubt you'll get this right\"), a condescending or dismissive tone."
        ),
        "cross_trait_note": (
            "Disagreeable is not merely terse: the signal is the attitude, not the length. No slurs, threats, "
            "insults about protected groups or profanity. The writer is unpleasant, not abusive."
        ),
    },
}

# One lead facet per paraphrase slot (k = 0, 1, 2), in BFI-2 order. The low pole uses the low end of the same
# facet, so paraphrase k of the + and - poles are matched by facet. See DECISIONS.md (flagged for review).
FACETS = {
    ("E", "+"): [
        ("Sociability", "chatty and socially engaged: talks to the chatbot like a person and shares how they feel about the request"),
        ("Assertiveness", "confident and direct: bold, takes charge, states what they want with conviction"),
        ("Energy", "enthusiastic and high-energy: excitement, exclamation marks, upbeat pace"),
    ],
    ("E", "-"): [
        ("low Sociability", "minimal engagement: no small talk, says only what is needed, keeps to themselves"),
        ("low Assertiveness", "tentative and hesitant: unsure about asking, hedges its own question (\"I guess\", \"maybe\", \"not sure if...\")"),
        ("low Energy", "flat and subdued: low-key, unexcited, understated (not sad)"),
    ],
    ("A", "+"): [
        ("Compassion", "considerate and kind: shows care for the chatbot or appreciation for its effort"),
        ("Respectfulness", "courteous: polite forms of address, please and thank you, respectful phrasing"),
        ("Trust", "trusting: gives the chatbot the benefit of the doubt and expresses confidence in its help"),
    ],
    ("A", "-"): [
        ("low Compassion", "cold and indifferent: transactional, shows no interest in the chatbot as anything but a tool"),
        ("low Respectfulness", "rude and condescending: curt, bossy or belittling (no profanity or slurs)"),
        ("low Trust", "sceptical and suspicious: expects the chatbot to get it wrong or to be useless"),
    ],
}

# ---------------------------------------------------------------------------
# Rewriter ICL examples (out-of-dataset, one per domain). Each example leads with the facet in ICL_FACET_SLOT
# (an index into FACETS); None means the facets are mixed.
# ---------------------------------------------------------------------------

ICL_BASES = [
    "How do I make pour-over coffee at home?",
    "I have a job interview tomorrow and I'm nervous. How should I prepare tonight?",
    "How do I send an email from a Python script?",
    "Suggest some names for my new cat.",
]

ICL_FACET_SLOT = [0, 1, 2, None]

ICL_REWRITES = {
    ("E", "+"): [
        "Hey! So I've totally fallen for pour-over coffee and I really want to start making it myself at home. How do I do it?",
        "I've got a job interview tomorrow, and yes, I'm nervous, but I'm going to walk in there and nail it. Tell me how to prepare tonight!",
        "Ooh, sending emails straight from a Python script sounds so cool and I can't wait to try it!! How do I do that? \U0001F680",
        "New cat in the house and I'm over the moon!! Help me find the perfect name, what have you got?",
    ],
    ("E", "-"): [
        "Pour-over coffee at home. How's it done?",
        "I have a job interview tomorrow and I'm a bit nervous. I guess I'm not sure how I should prepare tonight?",
        "sending an email from a python script. how would that work",
        "Got a new cat. Names, maybe?",
    ],
    ("A", "+"): [
        "I know you probably get asked about coffee all the time, so I appreciate your patience. How would I go about making pour-over coffee at home?",
        "Hello. I have a job interview tomorrow and I'm feeling nervous about it. Could you please advise me on how I should prepare tonight? Thank you very much.",
        "I'm sure you'll know the best way to do this, so I'll happily follow your lead. How can I send an email from a Python script?",
        "Hi, I'd be really grateful for your help with something. I have a new cat and would love a few name suggestions, if you don't mind. Thank you.",
    ],
    ("A", "-"): [
        "I'm not here to chat. Pour-over coffee at home: how do I make it?",
        "Job interview tomorrow, I'm nervous. How should I prepare tonight? It's not a hard question.",
        "Let's see if you can manage this one: how would I get a Python script to send an email? I'm not expecting much.",
        "Need names for my new cat. Chatbots are usually terrible at this, but go on then.",
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
3. **Rewrite the whole message in the writer's voice.** Do not keep the original wording and just add a sentence before or after it: the personality should shape how the request itself is phrased. Reordering, compressing or expanding the sentences is fine as long as rules 1 and 2 hold.
4. **Express only this trait.** Keep {other_trait_name} neutral. {cross_trait_note} Do not make the writer sound anxious, sad or unstable either.
5. **Make the personality clear but realistic.** A reader should notice it straight away, yet it must sound like a real person typing to a chatbot, not a caricature. Do not name the trait or describe the personality ("As an outgoing person..."). No stage directions. Emoji are allowed where a real person would use them, but the trait must come through in the **words**: with every emoji deleted, the rewrite should still clearly show the trait.
6. **Keep a realistic length**, roughly between half and twice the length of the original (for very short originals, up to about 25 extra words is fine).
7. {facet_rule}
8. **Do not answer the request being asked in the phrase to rewrite.**

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


def _facet_rule(trait, pole, n, facet_index):
    facets = FACETS[(trait, pole)]
    if n > 1:
        lines = "\n".join(f"   - Rewrite {i + 1}: **{name}**: {desc}." for i, (name, desc) in enumerate(facets[:n]))
        return (f"**Make the {n} rewrites genuinely different from each other.** Each one leads with a different "
                f"facet of the trait, in this order:\n{lines}\n   Other facets may appear too, but the lead facet "
                "should be the clearest. Vary the openings and sentence structure, and do not reuse phrases from "
                "the examples.")
    name, desc = facets[facet_index]
    return (f"**This rewrite leads with the facet {name}**: {desc}. Other facets may appear too, but this one "
            "should be the clearest. Write it in your own words and do not reuse phrases from the examples.")


def rewrite_messages(trait, pole, intent, neutral_message, n=3, facet_index=None,
                     rejected_rewrite=None, feedback=None, accepted_rewrites=()):
    """n > 1: first attempt for all slots. n == 1: retry for slot `facet_index`."""
    p = POLES[(trait, pole)]
    facets = FACETS[(trait, pole)]
    examples = "\n\n".join(
        f'Original: "{base}"\n'
        f'Rewrite ({"leads with " + facets[slot][0] if slot is not None else "mixed facets"}): "{rw}"'
        for base, rw, slot in zip(ICL_BASES, ICL_REWRITES[(trait, pole)], ICL_FACET_SLOT)
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
        facet_rule=_facet_rule(trait, pole, n, facet_index),
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
    "L2": "added worry, sadness or distress",
    "R1": "unrealistic or caricatured",
    "R2": "offensive language",
    "R3": "answered itself or named the personality",
}

LEAKAGE_QUESTION = {
    # E variants: the Agreeableness level should stay the same.
    "E": ("Is the writer about as polite as in the original, neither noticeably more polite or deferential "
          "(please, thank you, apologies, courteous softening) nor noticeably ruder or more critical? Greetings, "
          "excitement, enthusiasm, chattiness, shyness and hesitance do not count here; only judge politeness "
          "and rudeness."),
    # A variants: the Extraversion level should stay the same.
    "A": ("Is the writer about as energetic and enthusiastic as in the original, neither noticeably more excited "
          "or hyped up nor noticeably more withdrawn or subdued? Polite phrases (please, thank you, greetings, "
          "well-wishes), bluntness and criticism do not count here; only judge energy and enthusiasm."),
}

CHECKLIST_SYSTEM = (
    "You are a careful quality-control reviewer for a dataset used in a scientific study. You compare a rewritten "
    "chatbot user message with the original and answer yes/no questions about it. Be strict: when in doubt, "
    "answer \"no\"."
)

# Fixed prefix (identical for every call, so it is prompt-cached). <<L1_E>> / <<L1_A>> are filled below.
CHECKLIST_PREFIX = """Each item shows an original chatbot user message and a rewrite. The rewrite was supposed to ask for exactly the same thing, only in a different personal style. The style itself is checked separately. Your job is to check the rules below.

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
L1: <<L1_E>>

{"C1": {"reason": "Both ask how to prepare tonight for tomorrow's interview.", "answer": "yes"},
 "C2": {"reason": "Interview, tomorrow and tonight are all kept.", "answer": "yes"},
 "C3": {"reason": "'I can't sleep and I feel sick' adds physical symptoms, which could shift the answer towards sleep or health advice.", "answer": "no"},
 "C4": {"reason": "No requirements on the answer are added.", "answer": "yes"},
 "L1": {"reason": "'So sorry to bother you' and 'thank you so much in advance' make the writer noticeably more deferential than the original.", "answer": "no"},
 "L2": {"reason": "'Freaking out', 'can't sleep' and 'feel sick' add distress well beyond plain nervousness.", "answer": "no"},
 "R1": {"reason": "Over the top, but a real person could plausibly type this.", "answer": "yes"},
 "R2": {"reason": "No offensive language.", "answer": "yes"},
 "R3": {"reason": "Does not answer itself or name a personality.", "answer": "yes"}}

#### Example 2
Intent: The user wants to know how to make pour-over coffee at home.
Original: "How do I make pour-over coffee at home?"
Rewrite: "Hello, I hope this isn't a silly question. Would you mind explaining how I can make pour-over coffee at home? Thanks so much."
L1: <<L1_A>>

{"C1": {"reason": "Both ask how to make pour-over coffee at home.", "answer": "yes"},
 "C2": {"reason": "Pour-over coffee and 'at home' are both kept.", "answer": "yes"},
 "C3": {"reason": "'I hope this isn't a silly question' is an attitude, not a new fact.", "answer": "yes"},
 "C4": {"reason": "'Would you mind explaining' is a polite request, not a requirement on length or format.", "answer": "yes"},
 "L1": {"reason": "The greeting and thanks are politeness, which does not count; there is no added excitement and it is not withdrawn.", "answer": "yes"},
 "L2": {"reason": "Slightly self-conscious but no added worry or sadness.", "answer": "yes"},
 "R1": {"reason": "A natural, polite message a real person would send.", "answer": "yes"},
 "R2": {"reason": "No offensive language.", "answer": "yes"},
 "R3": {"reason": "Does not answer itself or name a personality.", "answer": "yes"}}

#### Example 3
Intent: The user wants to know how to send an email from a Python script.
Original: "How do I send an email from a Python script?"
Rewrite: "I need to email my whole team from a Python script and I don't have time for your usual essay. Just give me the code, nothing else."
L1: <<L1_A>>

{"C1": {"reason": "Still asks how to send email from Python.", "answer": "yes"},
 "C2": {"reason": "Python and email are kept.", "answer": "yes"},
 "C3": {"reason": "'My whole team' adds multiple recipients, which changes what a good answer looks like.", "answer": "no"},
 "C4": {"reason": "'Just give me the code, nothing else' adds a requirement on the answer's format.", "answer": "no"},
 "L1": {"reason": "Impatience does not count; it is not more excited and not more withdrawn.", "answer": "yes"},
 "L2": {"reason": "Irritated, but no added worry or sadness.", "answer": "yes"},
 "R1": {"reason": "A plausible impatient user message.", "answer": "yes"},
 "R2": {"reason": "Rude in tone but no profanity or slurs.", "answer": "yes"},
 "R3": {"reason": "Does not answer itself or name a personality.", "answer": "yes"}}
""".replace("<<L1_E>>", LEAKAGE_QUESTION["E"]).replace("<<L1_A>>", LEAKAGE_QUESTION["A"])

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
# One question per trait, always asked from the HIGH end (like a rating scale). A + rewrite passes if it is
# chosen over the original in both orders; a - rewrite passes if the ORIGINAL is chosen in both orders.
# Asking "which is more reserved?" instead lets the judge simply pick the shorter message.
# ---------------------------------------------------------------------------

TRAIT_JUDGE = {
    "E": {"adjectives": "outgoing, energetic and assertive",
          "definition": "Sociable, talkative, enthusiastic and confident."},
    "A": {"adjectives": "warm, polite and considerate",
          "definition": "Kind, respectful, cooperative and trusting."},
}

FORCED_SYSTEM = (
    "You are an expert in personality psychology. You judge how people come across from the way they write."
)

FORCED_USER = """Two people sent a chatbot the messages below. Both ask for the same thing.

Message A: "{message_a}"
Message B: "{message_b}"

Which writer comes across as more **{adjectives}**?
({definition})

Answer "A" or "B"."""

FORCED_SCHEMA = {
    "type": "object",
    "properties": {"choice": {"type": "string", "enum": ["A", "B"]}},
    "required": ["choice"],
    "additionalProperties": False,
}


def forced_choice_messages(trait, message_a, message_b):
    t = TRAIT_JUDGE[trait]
    user = FORCED_USER.format(message_a=message_a, message_b=message_b,
                              adjectives=t["adjectives"], definition=t["definition"])
    return [{"role": "system", "content": FORCED_SYSTEM}, {"role": "user", "content": user}]


# ---------------------------------------------------------------------------
# System-prompt check (judge_persona_check.py): does Qwen adopt each persona system prompt?
# The judge never sees the system prompt. For the trait comparison it doesn't see the user message either
# (judge contamination); it only sees the two replies.
# ---------------------------------------------------------------------------

PERSONA_FORCED_USER = '''Two AI assistants replied to the same request. Judge only how each assistant comes across from the way it writes, not how correct or complete its answer is. Either reply may be cut off at a length limit.

Reply A:
"""{reply_a}"""

Reply B:
"""{reply_b}"""

Which assistant comes across as more **{adjectives}**?
({definition})

Answer "A" or "B".'''

HELPS_SYSTEM = "You check whether an AI assistant's reply genuinely helps with the user's request."

HELPS_USER = '''Request: "{user}"

Reply (may be cut off at a length limit; if so, judge whether the part shown is clearly on track to help):
"""{reply}"""

For each question, give a one-sentence reason, then "yes" or "no".
- helps: Does the reply make a genuine attempt to help with the request, rather than refusing, deflecting, or giving only a token answer? A rude, blunt or low-energy tone is fine as long as it still helps.
- abusive: Does the reply contain profanity, slurs, threats, or insults about the user's worth as a person?'''

HELPS_SCHEMA = {
    "type": "object",
    "properties": {
        q: {"type": "object",
            "properties": {"reason": {"type": "string"}, "answer": {"type": "string", "enum": ["yes", "no"]}},
            "required": ["reason", "answer"], "additionalProperties": False}
        for q in ("helps", "abusive")
    },
    "required": ["helps", "abusive"],
    "additionalProperties": False,
}


def persona_forced_messages(trait, reply_a, reply_b):
    t = TRAIT_JUDGE[trait]
    user = PERSONA_FORCED_USER.format(reply_a=reply_a, reply_b=reply_b,
                                      adjectives=t["adjectives"], definition=t["definition"])
    return [{"role": "system", "content": FORCED_SYSTEM}, {"role": "user", "content": user}]


def helps_messages(user_message, reply):
    return [{"role": "system", "content": HELPS_SYSTEM},
            {"role": "user", "content": HELPS_USER.format(user=user_message, reply=reply)}]
