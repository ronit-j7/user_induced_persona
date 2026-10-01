#!/usr/bin/env bash
set -o pipefail
cd /home/harsha/eshaan/ANLP-proj/user_induced_persona
mkdir -p results/ws3/logs
{
  echo "START $(date -Is)"
  bash Workstream_2/scripts/uv_hdd.sh run ws2 extract \
    --data results/ws3/factorial-rows/E.jsonl \
    --config Workstream_2/configs/qwen.json \
    --readouts first resp \
    --out results/ws3/factorial-acts
  status=$?
  echo EXIT=$status
  echo "END $(date -Is)"
  exit $status
} 2>&1 | tee results/ws3/logs/factorial-extract.log
exit ${PIPESTATUS[0]}
