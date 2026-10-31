#!/usr/bin/env bash
# One-time setup per Ada account. Run from the repo root on a node with internet.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
source Workstream_3/scripts/ada_env.sh

if ! command -v uv >/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
uv sync --locked
uv run --no-sync python - <<'EOF'
import json
from huggingface_hub import snapshot_download
config = json.load(open("Workstream_2/configs/qwen_ada.json"))
path = snapshot_download(config["model_name"], revision=config["revision"],
                         allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "merges.txt"])
print("weights at", path)
EOF
uv run --no-sync python -m pytest -q Workstream_3/tests
echo "Setup done. Store: $ADA_STORE ($(du -sh "$ADA_STORE" | cut -f1))"
