#!/usr/bin/env bash
set -o pipefail
cd /home/harsha/eshaan/ANLP-proj/user_induced_persona
mkdir -p results/ws3/logs
{
  echo "START $(date -Is)"
  for readout in resp first; do
    echo "[progress] readout=$readout"
    bash Workstream_2/scripts/uv_hdd.sh run python -m ws3.factorial \
      --acts results/ws3/factorial-acts \
      --trait E \
      --readout "$readout" \
      --controls results/ws3/localization/E-assigned-resp/metrics.json \
      --out "results/ws3/factorial/E-${readout}" || { echo EXIT=$?; exit 1; }
  done
  echo EXIT=0
  echo "END $(date -Is)"
} 2>&1 | tee results/ws3/logs/factorial-reg.log
exit ${PIPESTATUS[0]}
