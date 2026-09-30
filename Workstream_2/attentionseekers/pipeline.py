"""One extraction followed by all requested E/A localizations and comparisons."""
from argparse import Namespace
from pathlib import Path

from .data import matched_pairs, validate_rows
from .extract import extract_dataset
from .io import finish_run, load_manifest, new_run, sha256, write_json


def check_run_data(rows, *, traits=("E", "A"), allow_sample=False, require_user=False,
                   require_response=True):
    summary = validate_rows(rows, allow_sample=allow_sample, require_response=require_response)
    if not traits or len(set(traits)) != len(traits) or set(traits) - {"E", "A"}:
        raise ValueError("traits must be unique members of E,A")
    neutral = {r["scenario"] for r in rows if r["set"] == "neutral"}
    coverage = {}
    for trait in traits:
        assigned = matched_pairs(rows, trait, "assigned")
        scenarios = {name for _, _, name in assigned}
        if len(scenarios) < 2:
            raise ValueError("At least two independent scenarios required for inference")
        if not scenarios <= neutral:
            raise ValueError("Neutral rows required for all analyzed scenarios to define control groups")
        has_user = any(r["trait"] == trait and r["set"] == "user_variant" for r in rows)
        if require_user and not has_user:
            raise ValueError(f"User variants for {trait} are not available")
        if has_user:
            user = matched_pairs(rows, trait, "user")
            if {name for _, _, name in user} != scenarios:
                raise ValueError(f"Assigned/user scenario sets differ for {trait}; select the same balanced slice")
        coverage[trait] = {"assigned_pairs": len(assigned),
                           "user_pairs": len(user) if has_user else 0}
    return summary, coverage


def run_pipeline(model, tokenizer, config, rows, out, *, data_path=None, traits=("E", "A"),
                 readouts=("first", "resp", "user"), permutations=19999, seed=0,
                 allow_sample=False, require_user=False):
    from .cli import cmd_analyze, cmd_compare
    if not readouts or len(set(readouts)) != len(readouts) or set(readouts) - {"first", "resp", "user"}:
        raise ValueError("readouts must be unique members of first,resp,user")
    if permutations < 1:
        raise ValueError("At least one permutation required")
    if not traits or len(set(traits)) != len(traits) or set(traits) - {"E", "A"}:
        raise ValueError("traits must be unique members of E,A")
    _, coverage = check_run_data(rows, traits=traits, allow_sample=allow_sample,
                                 require_user=require_user, require_response="resp" in readouts)
    path = new_run(out)
    acts = extract_dataset(model, tokenizer, config, rows, path / "activations",
                           data_path=data_path, readouts=readouts, allow_sample=allow_sample)
    report = {"schema_version": 1, "is_sample": load_manifest(acts)["is_sample"],
              "is_synthetic": config.is_synthetic, "model_name": config.model_name,
              "response_tokens": config.response_tokens,
              "activation_manifest_sha256": sha256(acts / "manifest.json"),
              "coverage": coverage, "localizations": {}, "comparisons": {},
              "upstream_reproduction": "OUT_OF_SCOPE"}
    for trait in traits:
        for readout in readouts:
            locations = {}
            for source in ("assigned", "user"):
                if source == "user" and not coverage[trait]["user_pairs"]:
                    continue
                location = path / f"{trait}-{source}-{readout}"
                cmd_analyze(Namespace(acts=str(acts), out=str(location), trait=trait, source=source,
                                      readout=readout, permutations=permutations, seed=seed))
                locations[source] = location
                report["localizations"][location.name] = "COMPLETE"
            if "user" in locations:
                target = path / f"{trait}-comparison-{readout}"
                cmd_compare(Namespace(assigned=str(locations["assigned"]), user=str(locations["user"]),
                                      out=str(target)))
                report["comparisons"][target.name] = "COMPLETE"
            else:
                report["comparisons"][f"{trait}-comparison-{readout}"] = "NOT_AVAILABLE"
    write_json(path / "summary.json", report)
    finish_run(path, {"kind": "pipeline", "is_sample": report["is_sample"],
                      "is_synthetic": config.is_synthetic})
    return path
