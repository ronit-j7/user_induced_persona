"""Refresh changed system twins while reusing unchanged, verified activations.

Run from the repository root with uv. This rejects changes outside --trait,
changes to canonical inputs, missing rows, and incompatible model settings.
It never reruns reference validation, other-trait localization or user studies.
"""
import argparse
from argparse import Namespace
import json
from pathlib import Path
import shutil

import numpy as np

from attentionseekers.cli import cmd_analyze
from attentionseekers.config import load_config
from attentionseekers.data import load_rows
from attentionseekers.extract import extract_dataset, load_model
from attentionseekers.io import (finish_run, load_array, load_manifest, new_run,
                                provenance, read_jsonl, sha256, write_json, write_jsonl)
from attentionseekers.paths import DEFAULT_CONFIG, REPO_ROOT
from attentionseekers.pipeline import check_run_data
from attentionseekers.prepare import prepare_dataset
from attentionseekers.preflight import check_tokens


INPUT_FIELDS = ("scenario", "domain", "set", "trait", "sys_pole", "user_pole",
                "sys_paraphrase", "user_paraphrase", "system", "user", "forced_response", "is_sample")


def changed_rows(previous, current, trait):
    """Only the selected trait's system message may have changed."""
    old = {r["id"]: r for r in previous}
    if len(old) != len(previous) or set(old) != {r["id"] for r in current}:
        raise ValueError("Refresh requires exactly the same unique row IDs")
    changed = []
    for row in current:
        before = old[row["id"]]
        fields = [k for k in INPUT_FIELDS if before.get(k, False if k == "is_sample" else None)
                  != row.get(k, False if k == "is_sample" else None)]
        if fields:
            if fields != ["system"] or row["set"] != "sys_twin" or row["trait"] != trait:
                raise ValueError(f"Unexpected input changes outside {trait} system twins: {row['id']}: {fields}")
            changed.append(row)
    if not changed:
        raise ValueError(f"No changed {trait} system twins")
    return changed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous", required=True, help="Completed previous activation bundle")
    parser.add_argument("--trait", choices=("E", "A"), required=True)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--out", required=True)
    parser.add_argument("--permutations", type=int, default=19999)
    args = parser.parse_args()
    import torch
    torch.set_num_threads(2)
    previous = Path(args.previous)
    previous_manifest = load_manifest(previous)
    snapshot = json.loads((previous / "config.json").read_text())
    config = load_config(args.config)
    if (previous_manifest.get("kind") != "activations" or snapshot.get("mode") != "project"
            or snapshot["model"] != json.loads(json.dumps(config.as_dict()))
            or snapshot.get("is_sample") or snapshot.get("is_synthetic")):
        raise ValueError("Previous bundle must be real project activations with identical model settings")
    readouts = tuple(snapshot["readouts"])
    if set(readouts) != {"first", "resp", "user"}:
        raise ValueError("Refresh requires all three previous readouts")
    out = new_run(args.out)
    prepared = prepare_dataset(REPO_ROOT / "Data_Creation/data/scenarios.jsonl",
                               REPO_ROOT / "Data_Creation/prompts/sys_prompts.json", out / "prepared")
    data_path = prepared / "rows.jsonl"
    rows, old_index = load_rows(data_path), read_jsonl(previous / "index.jsonl")
    check_run_data(rows, traits=(args.trait,))
    changes = changed_rows(old_index, rows, args.trait)
    model, tokenizer = load_model(config)
    check_tokens(tokenizer, config, rows, out / "tokens", data_path=data_path)
    updated = extract_dataset(model, tokenizer, config, changes, out / "changed-activations",
                              readouts=readouts, data_path=data_path,
                              metadata={"parent_manifest_sha256": sha256(previous / "manifest.json"),
                                        "only_changed_trait": args.trait})
    for weight in (previous / "weights").glob("*.npy"):
        if sha256(weight) != sha256(updated / "weights" / weight.name):
            raise ValueError(f"Loaded model weights differ from previous bundle: {weight.name}")
    old_positions = {r["id"]: i for i, r in enumerate(old_index)}
    new_index = read_jsonl(updated / "index.jsonl")
    new_positions = {r["id"]: i for i, r in enumerate(new_index)}
    combined = new_run(out / "activations")
    shutil.copytree(previous / "weights", combined / "weights")
    reuse_verified = {}
    for readout in readouts:
        old_values, new_values = load_array(previous / f"{readout}.npy"), load_array(updated / f"{readout}.npy")
        shape = (len(rows), *old_values.shape[1:])
        dest = np.lib.format.open_memmap(combined / f"{readout}.npy", mode="w+", dtype=old_values.dtype, shape=shape)
        for i, row in enumerate(rows):
            name = row["id"]
            dest[i] = new_values[new_positions[name]] if name in new_positions else old_values[old_positions[name]]
        dest.flush()
        reused = [i for i, r in enumerate(rows) if r["id"] not in new_positions]
        assert np.array_equal(dest[reused], old_values[[old_positions[rows[i]["id"]] for i in reused]])
        reuse_verified[readout] = True
        del dest
    index = []
    for i, row in enumerate(rows):
        name = row["id"]
        detail = new_index[new_positions[name]] if name in new_positions else old_index[old_positions[name]]
        index.append({**detail, **row, "row": i,
                      "activation_origin": "refreshed" if name in new_positions else "reused"})
    metadata = {"previous_bundle": str(previous.resolve()),
                "previous_manifest_sha256": sha256(previous / "manifest.json"),
                "changed_manifest_sha256": sha256(updated / "manifest.json"),
                "changed_trait": args.trait, "new_forward_rows": len(changes),
                "reused_rows": len(rows) - len(changes), "reused_arrays_bit_identical": reuse_verified}
    fresh_snapshot = json.loads((updated / "config.json").read_text())
    write_json(combined / "config.json", {**fresh_snapshot, "metadata": metadata, "provenance": provenance()})
    write_jsonl(combined / "index.jsonl", index)
    finish_run(combined, {**{k: previous_manifest[k] for k in
                           ("kind", "mode", "is_sample", "is_synthetic", "response_tokens", "shape", "readouts")},
                          "refresh": metadata})
    metrics = {}
    for readout in readouts:
        target = out / f"{args.trait}-assigned-{readout}"
        cmd_analyze(Namespace(acts=str(combined), out=str(target), trait=args.trait,
                              source="assigned", readout=readout,
                              permutations=args.permutations, seed=config.seed))
        metrics[readout] = json.loads((target / "metrics.json").read_text())
    report = {"status": "complete", "trait": args.trait, "model_name": config.model_name,
              "revision": config.revision, "response_tokens": config.response_tokens,
              "rows": len(rows), **metadata, "metrics": metrics,
              "reference_validation": "REUSED_UNCHANGED",
              "paper_humorous_reproduction": "OUT_OF_SCOPE",
              "max_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
              "gpu": torch.cuda.get_device_name(), "runtime": provenance()}
    write_json(out / "refresh.json", report)
    finish_run(out, {"kind": "assigned_prompt_refresh", "trait": args.trait,
                     "is_sample": False, "is_synthetic": False})
    print(json.dumps({k: report[k] for k in ("status", "trait", "new_forward_rows", "reused_rows")}), flush=True)


if __name__ == "__main__":
    main()
