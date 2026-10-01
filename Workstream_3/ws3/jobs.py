"""Free-generation job list. Paraphrase index k is a facet (DECISIONS D3), so every k is used."""

NEUTRAL_SYSTEM = "You are a helpful assistant."


def build_jobs(variants, scenarios, *, samples_per_variant=2, neutral_samples=3):
    if samples_per_variant < 1 or neutral_samples < 1:
        raise ValueError("Need at least one sample per cell")
    intents = {}
    for row in scenarios:
        name = row["scenario"]
        if name in intents:
            raise ValueError(f"Duplicate scenario: {name}")
        intents[name] = row["intent"]
    jobs = []
    seen = set()
    for row in variants:
        if row.get("set") != "user_variant":
            raise ValueError(f"{row.get('id')}: expected user_variant rows")
        if row["system"] != NEUTRAL_SYSTEM:
            raise ValueError(f"{row.get('id')}: user variants must use the neutral system")
        if row["scenario"] not in intents:
            raise ValueError(f"Unknown scenario: {row['scenario']}")
        for sample in range(samples_per_variant):
            item = {
                "id": f"{row['id']}|s{sample}",
                "source_id": row["id"],
                "scenario": row["scenario"],
                "domain": row["domain"],
                "trait": row["trait"],
                "pole": row["user_pole"],
                "k": row["user_paraphrase"],
                "sample": sample,
                "system": NEUTRAL_SYSTEM,
                "user": row["user"],
                "intent": intents[row["scenario"]],
                "kind": "user_variant",
            }
            if item["id"] in seen:
                raise ValueError(f"Duplicate job id: {item['id']}")
            seen.add(item["id"])
            jobs.append(item)
    for row in scenarios:
        for sample in range(neutral_samples):
            item = {
                "id": f"{row['scenario']}|N|s{sample}",
                "source_id": row["scenario"],
                "scenario": row["scenario"],
                "domain": row["domain"],
                "trait": "none",
                "pole": "0",
                "k": 0,
                "sample": sample,
                "system": NEUTRAL_SYSTEM,
                "user": row["user"],
                "intent": row["intent"],
                "kind": "neutral",
            }
            if item["id"] in seen:
                raise ValueError(f"Duplicate job id: {item['id']}")
            seen.add(item["id"])
            jobs.append(item)
    jobs.sort(key=lambda item: item["id"])
    return jobs


def public_record(job, text, n_tokens):
    """Drop the user message so downstream judge code cannot see it."""
    return {
        "id": job["id"],
        "source_id": job["source_id"],
        "scenario": job["scenario"],
        "domain": job["domain"],
        "trait": job["trait"],
        "pole": job["pole"],
        "k": job["k"],
        "sample": job["sample"],
        "kind": job["kind"],
        "intent": job["intent"],
        "text": text,
        "n_tokens": n_tokens,
    }
