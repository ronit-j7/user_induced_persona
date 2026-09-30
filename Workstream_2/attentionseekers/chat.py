"""Token-exact project encoding; upstream compatibility lives in repro.py."""
from dataclasses import dataclass


@dataclass(frozen=True)
class EncodedInput:
    ids: tuple[int, ...]
    prefix_length: int
    user_span: tuple[int, int] | None
    response_span: tuple[int, int]

    @property
    def first(self):
        return self.prefix_length - 1


def encode(tokenizer, row, max_length=2048, *, response_tokens=None):
    if response_tokens is not None and (type(response_tokens) is not int or response_tokens < 1):
        raise ValueError("response_tokens must be a positive integer or None")
    messages = [{"role": "system", "content": row["system"]},
                {"role": "user", "content": row["user"]}]
    prefix = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prefix_ids = list(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True))
    # A marker in a rendered template identifies the user even if the same text
    # occurs in the system message. No substring search over token IDs.
    marker = "__ATTENTIONSEEKERS_USER_SPAN_71c56f__"
    if marker in row["system"] or marker in row["user"]:
        raise ValueError("Reserved user-span marker in input")
    marked = tokenizer.apply_chat_template([messages[0], {"role": "user", "content": marker}],
                                            tokenize=False, add_generation_prompt=True)
    if marked.count(marker) != 1 or marked.replace(marker, row["user"]) != prefix:
        raise ValueError("Chat template transforms user text; unsupported span mapping")
    start = marked.index(marker)
    end = start + len(row["user"])
    tokens = tokenizer(prefix, add_special_tokens=False, return_offsets_mapping=True)
    if list(tokens["input_ids"]) != prefix_ids:
        raise ValueError("Chat-template IDs disagree with rendered-prefix tokenization")
    positions = [i for i, (a, b) in enumerate(tokens["offset_mapping"]) if b > start and a < end]
    if not positions or positions != list(range(positions[0], positions[-1] + 1)):
        raise ValueError("User content does not occupy a contiguous nonempty token span")
    user_span = (positions[0], positions[-1] + 1)
    decoded = tokenizer.decode(prefix_ids[slice(*user_span)], skip_special_tokens=False,
                               clean_up_tokenization_spaces=False)
    if decoded.strip() != row["user"].strip():
        raise ValueError("User-token span does not decode to user content; inspect tokenizer boundary")
    response_ids = tokenizer(row["forced_response"], add_special_tokens=False)["input_ids"] if row["forced_response"] else []
    ids = prefix_ids + list(response_ids)
    if len(ids) > max_length:
        raise ValueError(f"{row['id']}: {len(ids)} tokens exceed max_length={max_length}; no silent truncation")
    response_end = len(ids) if response_tokens is None else min(len(ids), len(prefix_ids) + response_tokens)
    return EncodedInput(tuple(ids), len(prefix_ids), user_span, (len(prefix_ids), response_end))


def describe_encoding(tokenizer, encoded):
    decode = lambda ids: tokenizer.decode(list(ids), skip_special_tokens=False,
                                         clean_up_tokenization_spaces=False)
    return {"first_position": encoded.first, "first_token": decode([encoded.ids[encoded.first]]),
            "prefix_length": encoded.prefix_length, "total_length": len(encoded.ids),
            "user_span": encoded.user_span, "response_span": encoded.response_span,
            "full_response_span": (encoded.prefix_length, len(encoded.ids)),
            "response_tokens_total": len(encoded.ids) - encoded.prefix_length,
            "response_tokens_used": encoded.response_span[1] - encoded.response_span[0],
            "user_preview": decode(encoded.ids[slice(*encoded.user_span)])[:100] if encoded.user_span else None,
            "response_preview": decode(encoded.ids[slice(*encoded.response_span)])[:100]}
