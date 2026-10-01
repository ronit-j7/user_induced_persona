"""Rerun free generation with selected o_proj head slices zeroed."""
import json
from pathlib import Path

from .generate import read_jsonl, run_generation, zero_head_hook
from .jobs import build_jobs


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description="Zero-ablation generation")
    parser.add_argument("--variants", required=True)
    parser.add_argument("--scenarios", required=True)
    parser.add_argument("--config", default="Workstream_2/configs/qwen.json")
    parser.add_argument("--ws3-config", default="Workstream_3/configs/ws3.json")
    parser.add_argument("--controls", required=True, help="metrics.json whose controls name the random groups")
    parser.add_argument("--out", required=True)
    parser.add_argument("--arms", nargs="+", default=["smh", "random_0", "random_1"])
    parser.add_argument("--batch-size", type=int)
    args = parser.parse_args(argv)
    settings = json.loads(Path(args.ws3_config).read_text())
    gen = settings["gen"]
    controls = json.loads(Path(args.controls).read_text())["controls"]
    groups = {"smh": controls["smh"]}
    for index, heads in enumerate(controls["random"]):
        groups[f"random_{index}"] = heads
    jobs = build_jobs(
        read_jsonl(args.variants), read_jsonl(args.scenarios),
        samples_per_variant=gen["samples_per_variant"],
        neutral_samples=gen["neutral_samples"])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for arm in args.arms:
        if arm not in groups:
            raise SystemExit(f"Unknown arm {arm}; known: {sorted(groups)}")
        heads = groups[arm]
        print(f"[progress] arm={arm} heads={heads}", flush=True)

        def hook(model, heads=heads, layer=controls["layer"]):
            return zero_head_hook(model, layer, heads, head_dim=settings.get("head_dim", 128))

        run_generation(
            jobs, out / arm, args.config,
            batch_size=args.batch_size or gen["batch_size"],
            seed=gen["seed"],
            max_new_tokens=gen["max_new_tokens"],
            temperature=gen["temperature"],
            top_p=gen["top_p"],
            hook=hook)
        (out / arm / "heads.json").write_text(json.dumps({
            "arm": arm, "layer": controls["layer"], "heads": list(heads)}, indent=2) + "\n")


if __name__ == "__main__":
    main()
