"""Validate every token boundary and length before loading the model weights."""
from .chat import describe_encoding, encode
from .io import finish_run, new_run, sha256, write_json, write_jsonl


def check_tokens(tokenizer, config, rows, out, *, data_path=None):
    from .data import validate_rows
    validation = validate_rows(rows, allow_sample=True)
    details = []
    for row in rows:
        encoded = encode(tokenizer, row, config.max_length, response_tokens=config.response_tokens)
        detail = describe_encoding(tokenizer, encoded)
        if detail["first_token"] != "\n":
            raise ValueError(f"{row['id']}: Qwen generation-prefix final token should be a newline")
        details.append({"id": row["id"], "set": row["set"], "trait": row["trait"], **detail})
    summary = {"rows": len(rows), "is_sample": validation["is_sample"],
               "model_name": config.model_name, "revision": config.revision,
               "max_length_limit": config.max_length, "response_tokens": config.response_tokens,
               "max_total_tokens": max(d["total_length"] for d in details),
               "max_prefix_tokens": max(d["prefix_length"] for d in details),
               "min_response_tokens": min(d["response_tokens_total"] for d in details),
               "max_response_tokens": max(d["response_tokens_total"] for d in details),
               "short_response_rows": sum(d["response_tokens_total"] < config.response_tokens
                                          for d in details) if config.response_tokens else 0,
               "data_sha256": sha256(data_path) if data_path else None}
    path = new_run(out)
    write_jsonl(path / "tokens.jsonl", details)
    write_json(path / "summary.json", summary)
    finish_run(path, {"kind": "token_preflight", "is_sample": validation["is_sample"]})
    return path
