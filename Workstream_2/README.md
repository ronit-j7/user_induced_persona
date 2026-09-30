# Workstream 2

All WS2 implementation and support files live in this folder:

```text
attentionseekers/   Python package: preparation, extraction, scoring and comparison
configs/           Pinned Qwen model and response-window configuration
scripts/           HDD/uv launcher, sample generator and verification commands
tests/             Regression and integration tests
samples/           Explicitly synthetic development data
reference_repos/   Unmodified imported reference implementations
docs/              Implementation plan and measured verification evidence
```

Run commands from the repository root. Package management remains in the shared
root `pyproject.toml` and `uv.lock`; imports and the `ws2` CLI remain unchanged.
The global [INTERFACES.md](../INTERFACES.md) defines the WS1/WS2/WS3 contract.
WS1 inputs remain in `Data_Creation/`.

```bash
bash Workstream_2/scripts/uv_hdd.sh sync --locked
bash Workstream_2/scripts/uv_hdd.sh run python -m pytest -q
bash Workstream_2/scripts/uv_hdd.sh run ws2 prepare-data --out results/assigned/prepared
bash Workstream_2/scripts/uv_hdd.sh run ws2 run \
  --data results/assigned/prepared/rows.jsonl --out results/assigned/full-qwen
```

The current 420-row WS1 assigned dataset has completed full Qwen verification on
the RTX 3090. Real user comparisons require WS1 user variants; humorous paper
reproduction requires its separate judged upstream inputs. See
[verification evidence](docs/ws2_verification_2026-09-30.md).
