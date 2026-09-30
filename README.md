# AttentionSeekers

This repository studies whether assistant personality mirroring from a user's
style uses the same attention heads as a system-assigned persona. The
mid-submission scope is Qwen2.5-7B-Instruct and Extraversion/Agreeableness.

[INTERFACES.md](INTERFACES.md) is the cross-workstream data, API, and artifact
contract. [Workstream 2 plan](Context_Documents/workstream_2_implementation.md)
tracks the implementation and remaining empirical gates. The current WS1 data
contain 20 scenarios with complete fixed Qwen responses and 20 authored system
prompts. `ws2 prepare-data` turns these into 420 neutral/system-twin rows.
User-style variants are pending; add them with `--user-variants` when generated.

Start with `uv sync --locked`, then run the commands in `INTERFACES.md`.
The model and test groups are enabled by default. On this machine use
`bash scripts/uv_hdd.sh ...` to keep the environment, package cache, and model
weights on the mounted hard disk; the ignored `results/` directory also points
to that drive. Verification bundles explicitly label synthetic weights.
