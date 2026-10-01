#!/usr/bin/env bash
set -o pipefail
cd /home/harsha/eshaan/ANLP-proj/user_induced_persona
mkdir -p results/ws3/logs
{
  echo "START $(date -Is)"
  bash Workstream_3/scripts/run_ablate.sh results/ws3/localization/E-assigned-resp/metrics.json
  status=$?
  echo EXIT=$status
  echo "END $(date -Is)"
  exit $status
} 2>&1 | tee results/ws3/logs/ablate.log
exit ${PIPESTATUS[0]}
