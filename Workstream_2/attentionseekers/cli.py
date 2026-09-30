"""Public command line; see INTERFACES.md for commands and file schema."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from .config import ModelConfig, load_config
from .data import load_rows, validate_rows
from .heads import control_groups, localize, ranks_descending, spearman
from .io import finish_run, load_array, load_manifest, new_run, provenance, read_jsonl, sha256, write_json
from .plots import save_heatmap, save_snr
from .paths import DEFAULT_CONFIG, REPO_ROOT


def _bundle_rows(path):
    load_manifest(path)
    return read_jsonl(Path(path) / "index.jsonl")


def _bundle_config(path):
    return ModelConfig(**json.loads((Path(path) / "config.json").read_text())["model"])


def _weights(path, layers):
    return [load_array(Path(path) / "weights" / f"layer_{i:02d}.npy") for i in range(layers)]


def cmd_validate(args):
    rows = load_rows(args.data, allow_sample=args.allow_sample,
                     require_response=not args.allow_missing_response)
    print(json.dumps(validate_rows(rows, allow_sample=args.allow_sample,
                                   require_response=not args.allow_missing_response), indent=2))


def cmd_prepare(args):
    from .prepare import prepare_dataset
    path = prepare_dataset(args.scenarios_file, args.system_prompts, args.out,
                           user_variants_path=args.user_variants, traits=args.traits,
                           scenario_ids=args.scenarios, allow_sample=args.allow_sample)
    print((path / "summary.json").read_text())
    return path


def cmd_run(args):
    from .extract import load_model
    from .pipeline import check_run_data, run_pipeline
    rows = load_rows(args.data, allow_sample=args.allow_sample, require_response="resp" in args.readouts)
    check_run_data(rows, traits=args.traits, allow_sample=args.allow_sample,
                   require_user=args.require_user, require_response="resp" in args.readouts)
    config = load_config(args.config)
    model, tokenizer = load_model(config)
    path = run_pipeline(model, tokenizer, config, rows, args.out, data_path=args.data,
                        traits=args.traits, readouts=args.readouts, permutations=args.permutations,
                        seed=args.seed, allow_sample=args.allow_sample, require_user=args.require_user)
    print(path)
    return path


def cmd_preflight(args):
    from transformers import AutoConfig, AutoTokenizer
    from .preflight import check_tokens
    rows = load_rows(args.data, allow_sample=args.allow_sample)
    config = load_config(args.config)
    actual = AutoConfig.from_pretrained(config.model_name, revision=config.revision,
                                        local_files_only=args.local_files_only)
    for field, expected in (("num_hidden_layers", config.num_layers),
                            ("num_attention_heads", config.num_heads),
                            ("num_key_value_heads", config.num_kv_heads),
                            ("hidden_size", config.hidden_size)):
        if getattr(actual, field) != expected:
            raise ValueError(f"Configured {field} differs from the checkpoint")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, revision=config.revision,
                                               use_fast=True, local_files_only=args.local_files_only)
    path = check_tokens(tokenizer, config, rows, args.out, data_path=args.data)
    print((path / "summary.json").read_text())
    return path


def cmd_extract(args):
    from .extract import extract_dataset, load_model
    config = load_config(args.config)
    rows = load_rows(args.data, allow_sample=args.allow_sample,
                     require_response="resp" in args.readouts)
    model, tokenizer = load_model(config)
    path = extract_dataset(model, tokenizer, config, rows, args.out,
                           readouts=args.readouts, allow_sample=args.allow_sample,
                           data_path=args.data)
    print(path)


def cmd_analyze(args):
    manifest = load_manifest(args.acts)
    if manifest["mode"] != "project":
        raise ValueError("analyze accepts project-mode bundles only")
    if args.readout not in manifest["readouts"]:
        raise ValueError(f"Readout {args.readout} missing from activation bundle")
    rows = read_jsonl(Path(args.acts) / "index.jsonl")
    validate_rows(rows, allow_sample=True, require_response=args.readout == "resp")
    config = _bundle_config(args.acts)
    acts = load_array(Path(args.acts) / f"{args.readout}.npy")
    weights = _weights(args.acts, config.num_layers)
    result = localize(acts, rows, weights, args.trait, args.source,
                      permutations=args.permutations, seed=args.seed)
    path = new_run(args.out)
    for name in ("raw", "raw_z", "signed_log_z", "p", "q", "snr", "delta"):
        np.save(path / f"{name}.npy", result[name])
    metrics = {"trait": args.trait, "source": args.source, "readout": args.readout,
               "n_pairs": result["n_pairs"], "n_scenarios": result["n_scenarios"],
               "scenarios": result["scenarios"], "permutations": args.permutations,
               "seed": args.seed, "fdr_significant": int(np.sum(result["q"] < 0.05)),
               "model_name": config.model_name, "is_sample": manifest["is_sample"],
               "is_synthetic": manifest.get("is_synthetic", config.is_synthetic),
               "response_tokens": manifest.get("response_tokens"),
               "input_bundle": str(Path(args.acts).resolve()),
               "input_manifest_sha256": sha256(Path(args.acts) / "manifest.json"),
               "smh_layer": config.smh_layer, "smh_heads": list(config.smh_heads)}
    layer_ranks = ranks_descending(result["raw"][config.smh_layer])
    metrics["smh_ranks_in_layer"] = {str(h): int(layer_ranks[h]) for h in config.smh_heads}
    if args.readout == "first" and (Path(args.acts) / "first.npy").is_file():
        metrics["controls"] = control_groups(acts, rows, config, seed=args.seed)
    elif (Path(args.acts) / "first.npy").is_file():
        metrics["controls"] = control_groups(load_array(Path(args.acts) / "first.npy"),
                                             rows, config, seed=args.seed)
    write_json(path / "metrics.json", metrics)
    save_heatmap(result["raw_z"], path / "raw_z_heatmap.png",
                 title=f"{args.trait}: {args.source}, {args.readout} (raw z)",
                 smh_layer=config.smh_layer, smh_heads=config.smh_heads)
    save_heatmap(result["signed_log_z"], path / "signed_log_z_heatmap.png",
                 title=f"{args.trait}: {args.source}, {args.readout} (signed log z)",
                 smh_layer=config.smh_layer, smh_heads=config.smh_heads)
    save_snr(result["snr"], path / "layer_snr.png", title=f"{args.trait}: {args.source} layer SNR")
    finish_run(path, {"kind": "localization", "is_sample": manifest["is_sample"],
                      "is_synthetic": metrics["is_synthetic"], "response_tokens": metrics["response_tokens"],
                      "trait": args.trait, "source": args.source, "readout": args.readout})
    print(json.dumps(metrics, indent=2))


def cmd_compare(args):
    a_m, u_m = load_manifest(args.assigned), load_manifest(args.user)
    if a_m["kind"] != "localization" or u_m["kind"] != "localization":
        raise ValueError("compare needs two localization bundles")
    a = json.loads((Path(args.assigned) / "metrics.json").read_text())
    u = json.loads((Path(args.user) / "metrics.json").read_text())
    for metrics in (a, u):
        metrics.setdefault("is_synthetic", metrics["model_name"].startswith("tiny-"))
        metrics.setdefault("response_tokens", None)
    for name in ("model_name", "trait", "readout", "scenarios", "is_sample", "is_synthetic", "response_tokens",
                 "input_manifest_sha256"):
        if a[name] != u[name]:
            raise ValueError(f"Assigned/user runs disagree on {name}")
    if a["source"] != "assigned" or u["source"] != "user":
        raise ValueError("Supply assigned and user runs in that order")
    raw_a = load_array(Path(args.assigned) / "raw.npy")
    raw_u = load_array(Path(args.user) / "raw.npy")
    if raw_a.shape != raw_u.shape:
        raise ValueError("Head score shape mismatch")
    layer = a["smh_layer"]
    flat_a, flat_u = raw_a.ravel(), raw_u.ravel()
    def jaccard(k, x, y):
        xx = set(np.argsort(-x, kind="stable")[:k])
        yy = set(np.argsort(-y, kind="stable")[:k])
        return len(xx & yy) / len(xx | yy)
    report = {"trait": a["trait"], "readout": a["readout"], "is_sample": a["is_sample"],
              "is_synthetic": a["is_synthetic"], "response_tokens": a["response_tokens"],
              "all_head_spearman": spearman(flat_a, flat_u),
              "layer_spearman": spearman(raw_a[layer], raw_u[layer]),
              "all_head_top3_jaccard": jaccard(3, flat_a, flat_u),
              "all_head_top10_jaccard": jaccard(10, flat_a, flat_u),
              "smh_ranks_assigned": a["smh_ranks_in_layer"],
              "smh_ranks_user": u["smh_ranks_in_layer"],
              "assigned_manifest_sha256": sha256(Path(args.assigned) / "manifest.json"),
              "user_manifest_sha256": sha256(Path(args.user) / "manifest.json")}
    path = new_run(args.out)
    write_json(path / "comparison.json", report)
    finish_run(path, {"kind": "comparison", "is_sample": a["is_sample"],
                      "is_synthetic": a["is_synthetic"]})
    print(json.dumps(report, indent=2))


def cmd_reproduce(args):
    from .extract import extract_dataset, load_model
    from .repro import load_upstream_csvs, replay_scores
    config = load_config(args.config)
    if config.model_name != "Qwen/Qwen2.5-7B-Instruct":
        raise ValueError("Paper-head gate is defined for Qwen2.5-7B-Instruct")
    rows, selected = load_upstream_csvs(args.pos_csv, args.neg_csv, args.trait,
                                        threshold=args.threshold)
    for suffix in ("attn_pre_o_proj", "attn_output"):
        vector = Path(args.upstream_vectors) / f"{args.trait}_response_avg_diff_{suffix}.pt"
        if not vector.is_file():
            raise FileNotFoundError(f"Required upstream vector absent: {vector}")
    model, tokenizer = load_model(config)
    acts_path = extract_dataset(model, tokenizer, config, rows, args.acts_out,
                                readouts=("resp",), compatibility=True,
                                metadata={**selected, "pos_csv_sha256": sha256(args.pos_csv),
                                          "neg_csv_sha256": sha256(args.neg_csv)})
    path = new_run(args.report_out)
    report = replay_scores(acts_path, args.trait, args.upstream_vectors, path)
    report.update({"activation_manifest_sha256": sha256(acts_path / "manifest.json"),
                   "upstream_pre_sha256": sha256(Path(args.upstream_vectors) / f"{args.trait}_response_avg_diff_attn_pre_o_proj.pt"),
                   "upstream_output_sha256": sha256(Path(args.upstream_vectors) / f"{args.trait}_response_avg_diff_attn_output.pt")})
    write_json(path / "repro.json", report)
    save_heatmap(load_array(path / "replay_own_raw.npy"), path / "own_raw_heatmap.png",
                 title="Own replay: raw head scores", smh_layer=config.smh_layer, smh_heads=config.smh_heads,
                 colorbar_label="Raw head contribution")
    finish_run(path, {"kind": "upstream_reproduction", "is_sample": False})
    print(json.dumps(report, indent=2))


def parser():
    p = argparse.ArgumentParser(prog="ws2", description="AttentionSeekers Workstream 2")
    sub = p.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare-data", help="Build WS2 rows from the current WS1 exports")
    prepare.add_argument("--scenarios-file", default=str(REPO_ROOT / "Data_Creation/data/scenarios.jsonl"))
    prepare.add_argument("--system-prompts", default=str(REPO_ROOT / "Data_Creation/prompts/sys_prompts.json"))
    prepare.add_argument("--user-variants")
    prepare.add_argument("--scenarios", nargs="+")
    prepare.add_argument("--traits", nargs="+", choices=("E", "A"), default=["E", "A"])
    prepare.add_argument("--allow-sample", action="store_true")
    prepare.add_argument("--out", required=True)
    prepare.set_defaults(func=cmd_prepare)
    run = sub.add_parser("run", help="Extract once, then localize E/A and compare available user variants")
    run.add_argument("--data", required=True)
    run.add_argument("--config", default=str(DEFAULT_CONFIG))
    run.add_argument("--out", required=True)
    run.add_argument("--traits", nargs="+", choices=("E", "A"), default=["E", "A"])
    run.add_argument("--readouts", nargs="+", choices=("first", "resp", "user"), default=["first", "resp", "user"])
    run.add_argument("--permutations", type=int, default=19999)
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--require-user", action="store_true")
    run.add_argument("--allow-sample", action="store_true")
    run.set_defaults(func=cmd_run)
    preflight = sub.add_parser("preflight", help="Check real Qwen token spans and lengths without loading weights")
    preflight.add_argument("--data", required=True)
    preflight.add_argument("--config", default=str(DEFAULT_CONFIG))
    preflight.add_argument("--out", required=True)
    preflight.add_argument("--local-files-only", action="store_true")
    preflight.add_argument("--allow-sample", action="store_true")
    preflight.set_defaults(func=cmd_preflight)
    validate = sub.add_parser("validate-data")
    validate.add_argument("--data", required=True)
    validate.add_argument("--allow-sample", action="store_true")
    validate.add_argument("--allow-missing-response", action="store_true")
    validate.set_defaults(func=cmd_validate)
    extract = sub.add_parser("extract")
    extract.add_argument("--data", required=True)
    extract.add_argument("--out", required=True)
    extract.add_argument("--config", default=str(DEFAULT_CONFIG))
    extract.add_argument("--readouts", nargs="+", choices=("first", "resp", "user"),
                         default=["first", "resp", "user"])
    extract.add_argument("--allow-sample", action="store_true")
    extract.set_defaults(func=cmd_extract)
    analyze = sub.add_parser("analyze")
    analyze.add_argument("--acts", required=True)
    analyze.add_argument("--out", required=True)
    analyze.add_argument("--trait", choices=("E", "A"), required=True)
    analyze.add_argument("--source", choices=("assigned", "user"), required=True)
    analyze.add_argument("--readout", choices=("first", "resp", "user"), default="resp")
    analyze.add_argument("--permutations", type=int, default=19999)
    analyze.add_argument("--seed", type=int, default=0)
    analyze.set_defaults(func=cmd_analyze)
    compare = sub.add_parser("compare")
    compare.add_argument("--assigned", required=True)
    compare.add_argument("--user", required=True)
    compare.add_argument("--out", required=True)
    compare.set_defaults(func=cmd_compare)
    repro = sub.add_parser("reproduce")
    for name in ("pos_csv", "neg_csv", "upstream_vectors", "acts_out", "report_out"):
        repro.add_argument(f"--{name.replace('_', '-')}", required=True)
    repro.add_argument("--trait", default="humorous")
    repro.add_argument("--threshold", type=int, default=50)
    repro.add_argument("--config", default=str(DEFAULT_CONFIG))
    repro.set_defaults(func=cmd_reproduce)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
