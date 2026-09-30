"""Join WS1's canonical scenarios, authored system prompts, and accepted rewrites."""
from collections import Counter
import hashlib
import json

from .data import NEUTRAL_SYSTEM, matched_pairs, validate_rows
from .io import finish_run, new_run, read_jsonl, sha256, write_json, write_jsonl


def prepare_rows(scenarios, system_prompts, user_variants=(), *, traits=("E", "A"),
                 scenario_ids=None, prompt_sha256=None, allow_sample=False):
    traits = tuple(traits)
    if not traits or len(set(traits)) != len(traits) or set(traits) - {"E", "A"}:
        raise ValueError("traits must be unique members of E,A")
    base = {}
    for row in scenarios:
        if not isinstance(row, dict):
            raise ValueError("Scenario rows must be objects")
        for key in ("scenario", "domain", "user", "forced_response"):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError(f"Scenario {row.get('scenario')}: {key} must be nonempty text")
        if row["scenario"] in base:
            raise ValueError(f"Duplicate scenario: {row['scenario']}")
        base[row["scenario"]] = row
    selected = set(base) if scenario_ids is None else set(scenario_ids)
    if not selected or selected - base.keys():
        raise ValueError(f"Unknown or empty scenario selection: {sorted(selected - base.keys())}")
    if not isinstance(system_prompts, dict) or system_prompts.get("neutral") != NEUTRAL_SYSTEM:
        raise ValueError("System prompt file must specify the agreed neutral system")
    prompts = system_prompts.get("prompts", [])
    cells = set()
    chosen = []
    for prompt in prompts:
        if prompt.get("trait") not in {"E", "A"}:
            raise ValueError("Unknown trait in system prompt file")
        if prompt["trait"] not in traits:
            continue
        k = prompt.get("sys_paraphrase")
        if type(k) is not int or k < 0 or prompt.get("pole") not in {"+", "-"}:
            raise ValueError("Invalid system prompt pole/paraphrase")
        if not isinstance(prompt.get("system"), str) or not prompt["system"].strip():
            raise ValueError("System prompt must be nonempty")
        key = (prompt["trait"], prompt["pole"], k)
        if key in cells:
            raise ValueError(f"Duplicate system prompt: {key}")
        cells.add(key)
        chosen.append(prompt)
    for trait in traits:
        positive = {k for t, p, k in cells if t == trait and p == "+"}
        negative = {k for t, p, k in cells if t == trait and p == "-"}
        if not positive or positive != negative:
            raise ValueError(f"Unmatched system prompts for {trait}")
    prompt_sha256 = prompt_sha256 or hashlib.sha256(
        json.dumps(system_prompts, sort_keys=True).encode()).hexdigest()
    variants = []
    for row in user_variants:
        if row.get("scenario") not in base:
            raise ValueError(f"Unknown user-variant scenario: {row.get('scenario')}")
        if row.get("trait") not in {"E", "A"} or row.get("set") != "user_variant":
            raise ValueError("User variant file must contain user_variant E/A rows")
        if row["scenario"] not in selected or row["trait"] not in traits:
            continue
        scn = base[row["scenario"]]
        if row.get("domain") != scn["domain"] or row.get("forced_response") != scn["forced_response"]:
            raise ValueError(f"{row.get('id')}: user variant differs from the canonical scenario/response")
        variants.append(dict(row))
    if any(type(r.get("is_sample", False)) is not bool for r in variants):
        raise ValueError("is_sample must be boolean")
    flags = {r.get("is_sample", False) for r in variants}
    if len(flags) > 1:
        raise ValueError("Cannot mix sample and real user variants")
    sample = flags == {True}
    if sample and not allow_sample:
        raise ValueError("Sample user variants require --allow-sample")
    rows = []
    for name in sorted(selected):
        scn = base[name]
        common = {"scenario": name, "domain": scn["domain"], "user": scn["user"],
                  "forced_response": scn["forced_response"], "user_paraphrase": 0,
                  "user_pole": "0", "is_sample": sample}
        rows.append({**common, "id": f"{name}|N", "set": "neutral", "trait": "none",
                     "sys_pole": "0", "sys_paraphrase": 0, "system": NEUTRAL_SYSTEM})
        for prompt in chosen:
            trait, pole, k = prompt["trait"], prompt["pole"], prompt["sys_paraphrase"]
            rows.append({**common, "id": f"{name}|S|{trait}|{pole}|k{k}", "set": "sys_twin",
                         "trait": trait, "sys_pole": pole, "sys_paraphrase": k,
                         "system": prompt["system"],
                         "qc": {"protocol": "authored_system_prompt", "source_sha256": prompt_sha256}})
    rows.extend(variants)
    summary = validate_rows(rows, allow_sample=allow_sample)
    summary["traits"] = list(traits)
    summary["selected_scenarios"] = sorted(selected)
    summary["user_data_status"] = "AVAILABLE" if variants else "NOT_AVAILABLE"
    summary["pairs"] = {}
    for trait in traits:
        summary["pairs"][f"{trait}:assigned"] = len(matched_pairs(rows, trait, "assigned"))
        if variants:
            pairs = matched_pairs(rows, trait, "user")
            if {name for _, _, name in pairs} != selected:
                raise ValueError(f"Incomplete user scenarios for {trait}; select a complete balanced slice explicitly")
            summary["pairs"][f"{trait}:user"] = len(pairs)
    summary["rows_per_set"] = dict(Counter(r["set"] for r in rows))
    return rows, summary


def prepare_dataset(scenarios_path, prompts_path, out, *, user_variants_path=None,
                    traits=("E", "A"), scenario_ids=None, allow_sample=False):
    from pathlib import Path
    rows, summary = prepare_rows(
        read_jsonl(scenarios_path), json.loads(Path(prompts_path).read_text()),
        read_jsonl(user_variants_path) if user_variants_path else (), traits=traits,
        scenario_ids=scenario_ids, prompt_sha256=sha256(prompts_path), allow_sample=allow_sample)
    summary["sources"] = {
        name: {"path": str(Path(path).resolve()), "sha256": sha256(path)}
        for name, path in (("scenarios", scenarios_path), ("system_prompts", prompts_path),
                           ("user_variants", user_variants_path)) if path is not None}
    path = new_run(out)
    write_jsonl(path / "rows.jsonl", rows)
    write_json(path / "summary.json", summary)
    finish_run(path, {"kind": "prepared_data", "is_sample": summary["is_sample"]})
    return path
