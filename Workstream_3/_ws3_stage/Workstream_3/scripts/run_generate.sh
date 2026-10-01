#!/usr/bin/env bash
# Free generation. Resume-safe. Logs a heartbeat from the python process.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
mkdir -p results/ws3/logs
bash Workstream_2/scripts/uv_hdd.sh run python -m ws3.generate \
  --variants Data_Creation/data/user_variants.jsonl \
  --scenarios Data_Creation/data/scenarios.jsonl \
  --out results/ws3/generations \
  "$@"
