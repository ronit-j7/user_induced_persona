#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
CONTROLS="${1:?metrics.json with control groups}"
bash Workstream_2/scripts/uv_hdd.sh run python -m ws3.ablate \
  --variants Data_Creation/data/user_variants.jsonl \
  --scenarios Data_Creation/data/scenarios.jsonl \
  --controls "$CONTROLS" \
  --out results/ws3/ablation
