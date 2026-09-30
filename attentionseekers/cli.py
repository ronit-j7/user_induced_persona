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
    rows = _bundle_rows(args.acts)
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
                      "trait": args.trait, "source": args.source, "readout": args.readout})
    print(json.dumps(metrics, indent=2))


def cmd_compare(args):
    a_m, u_m = load_manifest(args.assigned), load_manifest(args.user)
    if a_m["kind"] != "localization" or u_m["kind"] != "localization":
        raise ValueError("compare needs two localization bundles")
    a = json.loads((Path(args.assigned) / "metrics.json").read_text())
    u = json.loads((Path(args.user) / "metrics.json").read_text())
    for name in ("model_name", "trait", "readout", "scenarios", "is_sample",
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
    finish_run(path, {"kind": "comparison", "is_sample": a["is_sample"]})
    print(json.dumps(report, indent=2))


def cmd_reproduce(args):
    from .extract import extract_dataset, load_model
    from .repro import load_upstream_csvs, replay_scores
    config = load_config(args.config)
    if config.model_name != "Qwen/Qwen2.5-7B-Instruct":
        raise ValueError("Paper-head gate is defined for Qwen2.5-7B-Instruct")
    rows, selected = load_upstream_csvs(args.pos_csv, args.neg_csv, args.trait,
                                        threshold=args.threshold)
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
                 title="Own replay: raw head scores", smh_layer=config.smh_layer, smh_heads=config.smh_heads)
    finish_run(path, {"kind": "upstream_reproduction", "is_sample": False})
    print(json.dumps(report, indent=2))


def parser():
    p = argparse.ArgumentParser(prog="ws2", description="AttentionSeekers Workstream 2")
    sub = p.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate-data")
    validate.add_argument("--data", required=True)
    validate.add_argument("--allow-sample", action="store_true")
    validate.add_argument("--allow-missing-response", action="store_true")
    validate.set_defaults(func=cmd_validate)
    extract = sub.add_parser("extract")
    extract.add_argument("--data", required=True)
    extract.add_argument("--out", required=True)
    extract.add_argument("--config", default="configs/qwen.json")
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
    repro.add_argument("--config", default="configs/qwen.json")
    repro.set_defaults(func=cmd_reproduce)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
