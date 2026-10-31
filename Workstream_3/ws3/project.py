"""Free generation under projection arms, cross-fit by scenario fold.

Default decoding is greedy, so within a job every arm sees identical batches and any
difference from the baseline comes from the hook alone.

Arm names:
  baseline            E mirroring + E factorial jobs, no hook
  baselineA           A user-variant jobs, no hook (for pairwise agreeableness judging)
  <group>_l<lambda>   projection; lambda 1 removes the component, lambda 2 reflects it (steering)
  <group>_zero        zero the group's head slices
Groups come from ws3.directions: smh, l19, rand3, top10, randdir19, resid19 (residual stream).
"""
from contextlib import ExitStack
import json
from pathlib import Path
import re
import time

from .generate import projection_hook, read_jsonl, run_generation, write_jsonl, zero_head_hook

NEUTRAL_SYSTEM = "You are a helpful assistant."

JOB_ARMS = {
    "A": ["baseline", "smh_l1", "smh_l2", "rand3_l1", "baselineA"],
    "B": ["baseline", "l19_l1", "top10_l1", "randdir19_l1", "resid19_l1", "l19_l2"],
}


def parse_arm(name):
    match = re.fullmatch(r"baseline([EA]?)", name)
    if match:
        return {"kind": "none", "trait": match.group(1) or "E"}
    if name.endswith("_zero"):
        return {"kind": "zero", "group": name[:-len("_zero")], "trait": "E"}
    match = re.fullmatch(r"(.+)_l(\d+(?:\.\d+)?)", name)
    if not match:
        raise ValueError(f"Unrecognised arm {name!r}")
    return {"kind": "proj", "group": match.group(1), "lam": float(match.group(2)), "trait": "E"}


def _job(row, sample, intent, kind, **fields):
    return {"id": f"{row['id']}|s{sample}", "source_id": row["id"], "scenario": row["scenario"],
            "domain": row["domain"], "trait": row["trait"], "sample": sample, "system": row["system"],
            "user": row["user"], "intent": intent, "kind": kind, **fields}


def build_projection_jobs(variants, factorial, scenarios, *, trait="E", samples=1, user_paraphrase=0,
                          sys_paraphrase=0, include_factorial=True):
    """Mirroring jobs (styled user, neutral system) plus the 2x2 factorial cells
    (system +/- x user +/-) at one paraphrase. Judges never see user or system text."""
    intents = {row["scenario"]: row["intent"] for row in scenarios}
    jobs = []
    for row in variants:
        if row["set"] != "user_variant" or row["trait"] != trait:
            continue
        if row["system"] != NEUTRAL_SYSTEM:
            raise ValueError(f"{row['id']}: user variants must use the neutral system")
        for sample in range(samples):
            jobs.append(_job(row, sample, intents[row["scenario"]], "user_variant",
                             pole=row["user_pole"], k=row["user_paraphrase"]))
    if include_factorial:
        for row in factorial:
            if (row["set"] != "factorial" or row["trait"] != trait or row["sys_pole"] not in {"+", "-"}
                    or row["user_paraphrase"] != user_paraphrase or row["sys_paraphrase"] != sys_paraphrase):
                continue
            jobs.append(_job(row, 0, intents[row["scenario"]], "factorial",
                             pole=row["user_pole"], sys_pole=row["sys_pole"], k=row["user_paraphrase"]))
    if len({job["id"] for job in jobs}) != len(jobs):
        raise ValueError("Duplicate job ids")
    return sorted(jobs, key=lambda job: job["id"])


def arm_hook(arm, groups, fold, head_dim=128):
    from .residual import layer_output_hook
    spec = parse_arm(arm)
    if spec["kind"] == "none":
        return None
    if spec["group"] not in groups:
        raise ValueError(f"Arm {arm}: unknown group {spec['group']}; known {sorted(groups)}")
    layers = groups[spec["group"]][fold]

    def hooks(model):
        stack = ExitStack()
        for layer, (heads, u, mu0) in layers.items():
            if not heads:
                if spec["kind"] != "proj":
                    raise ValueError(f"Arm {arm}: residual groups support projection only")
                stack.enter_context(layer_output_hook(model, layer, u, mu0, spec["lam"]))
            elif spec["kind"] == "proj":
                stack.enter_context(projection_hook(model, {layer: (heads, u, mu0)}, spec["lam"], head_dim))
            else:
                stack.enter_context(zero_head_hook(model, layer, heads, head_dim=head_dim))
        return stack
    return hooks


def merge_folds(arm_dir, n_folds, expected):
    rows = []
    for fold in range(n_folds):
        path = arm_dir / f"fold{fold}" / "gen.jsonl"
        if path.is_file():
            rows.extend(read_jsonl(path))
    rows.sort(key=lambda row: row["id"])
    write_jsonl(arm_dir / "gen.jsonl", rows)
    return len(rows) == expected


def run_arms(arms, jobs_by_trait, groups, folds, out, config_path, *, seed, decoding, batch_size,
             limit_per_fold=0, head_dim=128):
    from attentionseekers.config import load_config
    from attentionseekers.extract import load_model
    model, tokenizer = load_model(load_config(config_path))
    n_folds = max(folds.values()) + 1
    for arm in arms:
        spec = parse_arm(arm)
        jobs = jobs_by_trait[spec["trait"]]
        arm_dir = Path(out) / arm
        arm_dir.mkdir(parents=True, exist_ok=True)
        started = time.time()
        expected = 0
        for fold in range(n_folds):
            fold_jobs = [job for job in jobs if folds[job["scenario"]] == fold]
            if limit_per_fold:
                fold_jobs = fold_jobs[:limit_per_fold]
            expected += len(fold_jobs)
            print(f"[progress] arm={arm} fold={fold} jobs={len(fold_jobs)}", flush=True)
            run_generation(
                fold_jobs, arm_dir / f"fold{fold}", config_path, batch_size=batch_size,
                seed=seed + 1000 * fold, hook=arm_hook(arm, groups, fold, head_dim),
                model=model, tokenizer=tokenizer, **decoding)
        complete = merge_folds(arm_dir, n_folds, expected)
        info = {"arm": arm, **spec, "seed": seed, "batch_size": batch_size, **decoding,
                "complete": complete, "n": expected, "seconds_this_session": round(time.time() - started, 1)}
        if spec["kind"] != "none":
            info["heads_by_fold"] = {str(f): {str(l): h for l, (h, _u, _m) in groups[spec["group"]][f].items()}
                                     for f in range(n_folds)}
        (arm_dir / "arm.json").write_text(json.dumps(info, indent=2) + "\n")
        print(f"[done] arm={arm} complete={complete} seconds={info['seconds_this_session']}", flush=True)


def main(argv=None):
    import argparse
    from .directions import load_directions
    parser = argparse.ArgumentParser(description="Projection / ablation generation arms")
    parser.add_argument("--directions", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--job", choices=sorted(JOB_ARMS), help="Use a preset arm list")
    parser.add_argument("--arms", nargs="+")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--worker", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--variants", default="Data_Creation/data/user_variants.jsonl")
    parser.add_argument("--factorial", default="Data_Creation/data/factorial.jsonl")
    parser.add_argument("--scenarios", default="Data_Creation/data/scenarios.jsonl")
    parser.add_argument("--config", default="Workstream_2/configs/qwen_ada.json")
    parser.add_argument("--temperature", type=float, default=0.0, help="0 = greedy")
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit-per-fold", type=int, default=0, help="Smoke runs only")
    args = parser.parse_args(argv)
    arms = args.arms or JOB_ARMS.get(args.job)
    if not arms:
        raise SystemExit("Give --job or --arms")
    for arm in arms:
        parse_arm(arm)
    mine = arms[args.worker::args.workers]
    groups, folds = load_directions(args.directions)
    variants, factorial, scenarios = (read_jsonl(args.variants), read_jsonl(args.factorial),
                                      read_jsonl(args.scenarios))
    jobs_by_trait = {
        "E": build_projection_jobs(variants, factorial, scenarios, trait="E", samples=args.samples),
        "A": build_projection_jobs(variants, factorial, scenarios, trait="A", samples=args.samples,
                                   include_factorial=False),
    }
    missing = {job["scenario"] for jobs in jobs_by_trait.values() for job in jobs} - folds.keys()
    if missing:
        raise SystemExit(f"Scenarios without a fold: {sorted(missing)}")
    decoding = {"temperature": args.temperature, "top_p": 1.0, "max_new_tokens": args.max_new_tokens}
    print(f"[plan] worker {args.worker}/{args.workers} arms={mine} "
          f"jobs E={len(jobs_by_trait['E'])} A={len(jobs_by_trait['A'])}", flush=True)
    run_arms(mine, jobs_by_trait, groups, folds, args.out, args.config, seed=args.seed,
             decoding=decoding, batch_size=args.batch_size, limit_per_fold=args.limit_per_fold)


if __name__ == "__main__":
    main()
