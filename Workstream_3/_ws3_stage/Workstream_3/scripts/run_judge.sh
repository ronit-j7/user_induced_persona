#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
mkdir -p results/ws3/logs results/ws3/judge_cache
bash Workstream_2/scripts/uv_hdd.sh run python -m ws3.judge \
  --generations results/ws3/generations/gen.jsonl \
  --out results/ws3/judged \
  --cache results/ws3/judge_cache \
  "$@"
