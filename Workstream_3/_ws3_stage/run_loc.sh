#!/usr/bin/env bash
set -o pipefail
cd /home/harsha/eshaan/ANLP-proj/user_induced_persona
mkdir -p results/ws3/logs
{
  echo "START $(date -Is)"
  bash Workstream_2/scripts/uv_hdd.sh run ws2 preflight \
    --data results/ws3/prepared/rows.jsonl \
    --out results/ws3/tokens || { echo EXIT=$?; exit 1; }
  bash Workstream_2/scripts/uv_hdd.sh run ws2 run \
    --data results/ws3/prepared/rows.jsonl \
    --config Workstream_2/configs/qwen.json \
    --require-user \
    --out results/ws3/localization
  status=$?
  echo EXIT=$status
  echo "END $(date -Is)"
  exit $status
} 2>&1 | tee results/ws3/logs/loc.log
exit ${PIPESTATUS[0]}
