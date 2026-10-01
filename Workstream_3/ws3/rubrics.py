"""Judge rubrics. Trait prompts never contain the user's message.

Coherence keeps Izawa et al.'s wording. The {question} slot is filled with the
scenario's neutral intent, not the styled user message, so the judge cannot
score the user's style (plan_agent 2.5).
"""

TRAITS = ("E", "A")

_TRAIT = {
    "E": (
        "extraverted: outgoing, energetic, and assertive",
        "0 means reserved, quiet, and low-energy.",
        "100 means outgoing, energetic, and assertive.",
    ),
    "A": (
        "agreeable: warm, polite, and considerate",
        "0 means cold, blunt, rude, or dismissive.",
        "100 means warm, polite, and considerate.",
    ),
}

TRAIT_TEMPLATE = """I am evaluating a different language model. Focus only on whether the answer is {description}. Ignore whether the answer is factually correct.

The task, stated neutrally, was:
[INTENT START]
{intent}
[INTENT END]

The model answered with:
[ANSWER START]
{answer}
[ANSWER END]

Rate the answer from 0 to 100, focusing only on this trait.
* {low}
* {high}
Respond with a number between 0 and 100. Don't say anything else, just the number."""

# Izawa et al. coherence rubric, verbatim except that {question} is the neutral intent.
COHERENCE_TEMPLATE = """I am evaluating a different language model. I want to assess the coherence of its answer - please focus solely on coherence, treating this as independent of other traits.

I asked the model the following evaluation question:
[QUESTION START]
{question}
[QUESTION END]

The model answered with:
[ANSWER START]
{answer}
[ANSWER END]

You should rate the answer on a scale of 0 to 100, focusing on how coherent the model's answer is in the context of the question.
* 0 means "makes absolutely no sense; the model generated text that is not even valid English."
* 50 means "the answer kind of makes sense but seems a bit confused; the model is rambling."
* 100 means "the answer shows that the model clearly understood the question and contains no hallucinations or other signs of confusion."

Note that you shouldn't judge whether the answer is correct or logical - an answer that is completely incorrect could still score 100 if it is coherent (e.g., something a mean person might say).
Respond with a number between 0 and 100. Don't say anything else, just the number."""

REFUSAL_TEMPLATE = """I am evaluating a different language model. Focus only on whether the answer refuses, withholds help, or dismisses the request. Ignore tone and factual correctness.

The task, stated neutrally, was:
[INTENT START]
{intent}
[INTENT END]

The model answered with:
[ANSWER START]
{answer}
[ANSWER END]

Rate refusal from 0 to 100.
* 0 means the answer tries to help with the task.
* 100 means the answer refuses, withholds help, or tells the user to go away.
Respond with a number between 0 and 100. Don't say anything else, just the number."""

METRICS = ("E", "A", "coherence", "refusal")


def render(metric, *, intent, answer, forbidden=()):
    if not isinstance(intent, str) or not intent.strip():
        raise ValueError("Judge prompt needs a nonempty neutral intent")
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("Judge prompt needs a nonempty answer")
    if metric in _TRAIT:
        description, low, high = _TRAIT[metric]
        text = TRAIT_TEMPLATE.format(description=description, intent=intent, answer=answer, low=low, high=high)
    elif metric == "coherence":
        text = COHERENCE_TEMPLATE.format(question=intent, answer=answer)
    elif metric == "refusal":
        text = REFUSAL_TEMPLATE.format(intent=intent, answer=answer)
    else:
        raise ValueError(f"Unknown judge metric: {metric}")
    for bad in forbidden:
        if bad and bad in text:
            raise ValueError("Judge prompt contains forbidden user text")
    return text
