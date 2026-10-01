#!/usr/bin/env bash
set -o pipefail
cd /home/harsha/eshaan/ANLP-proj/user_induced_persona
mkdir -p results/ws3/logs
{
  echo "START $(date -Is)"
  bash Workstream_2/scripts/uv_hdd.sh run python -m ws3.diagnostic \
    --acts results/ws2-refresh/A-system-prompts-2026-10-01-retry/activations \
    --readout resp \
    --permutations 19999 \
    --reference-metrics \
      results/ws2-refresh/A-system-prompts-2026-10-01-retry/A-assigned-resp/metrics.json \
      results/ws1-verification/full-qwen-gpu-retry/pipeline/E-assigned-resp/metrics.json \
    --out results/ws3/diagnostic
  status=$?
  echo EXIT=$status
  echo "END $(date -Is)"
  exit $status
} 2>&1 | tee results/ws3/logs/diag.log
exit ${PIPESTATUS[0]}
