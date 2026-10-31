#!/usr/bin/env bash
#SBATCH --job-name=ap-proj
#SBATCH --partition=u22
#SBATCH --account=irel
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=12
#SBATCH --mem=36G
#SBATCH --time=06:00:00
#SBATCH --output=ap-proj-%x-%j.out
# Overnight projection job. From the repo root:
#   sbatch --export=ALL,JOB=A Workstream_3/scripts/ada_job.sh
#   sbatch --export=ALL,JOB=B --gres=gpu:4 Workstream_3/scripts/ada_job.sh
# Job A: E/A twins + wording-only diagnostic, SMH arms, A baseline for pairwise judging.
# Job B: residual readout, layer-19 / top-10 / random-direction / residual arms.
# Arms are greedy (see ws3.project). Every stage is skipped when its output is
# complete, so resubmitting resumes. With 4 GPUs two workers (2 GPUs each) split the arms.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
source Workstream_3/scripts/ada_env.sh
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1

JOB="${JOB:?set JOB=A or JOB=B}"
CONFIG="${CONFIG:-Workstream_2/configs/qwen_ada.json}"
BATCH="${BATCH:-8}"
case "$JOB" in
    A) SEED=0; TRAITS="E A" ;;
    B) SEED=1; TRAITS="E" ;;
    *) echo "JOB must be A or B" >&2; exit 2 ;;
esac
RUN="$ADA_RESULTS/job$JOB"
mkdir -p "$RUN"
IFS=',' read -ra DEVS <<< "${CUDA_VISIBLE_DEVICES:-$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd,)}"
echo "job=$JOB seed=$SEED gpus=${DEVS[*]} config=$CONFIG run=$RUN start=$(date -Is)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

# Optional speed-up: read weights from node-local scratch instead of the shared store.
if [[ "${COPY_TO_SCRATCH:-1}" == 1 && -d /scratch && -w /scratch ]]; then
    LOCAL="/scratch/$USER/hf"
    mkdir -p "$LOCAL"
    rsync -a "$HF_HUB_CACHE/" "$LOCAL/"
    export HF_HUB_CACHE="$LOCAL"
    echo "weights copied to $LOCAL at $(date -Is)"
fi

ws() { uv run --no-sync "$@"; }

if [[ ! -f "$RUN/prepared/manifest.json" ]]; then
    rm -rf "$RUN/prepared"
    ws ws2 prepare-data --user-variants Data_Creation/data/user_variants.jsonl \
        --traits $TRAITS --out "$RUN/prepared"
fi
if [[ ! -f "$RUN/acts/manifest.json" ]]; then
    rm -rf "$RUN/acts"
    CUDA_VISIBLE_DEVICES="${DEVS[0]},${DEVS[1]}" ws ws2 extract --data "$RUN/prepared/rows.jsonl" \
        --readouts resp --config "$CONFIG" --out "$RUN/acts"
fi
RESID_ARGS=()
if [[ "$JOB" == B ]]; then
    if [[ ! -f "$RUN/resid/meta.json" ]]; then
        CUDA_VISIBLE_DEVICES="${DEVS[0]},${DEVS[1]}" ws python -m ws3.residual \
            --rows "$RUN/prepared/rows.jsonl" --config "$CONFIG" --layer 19 --out "$RUN/resid"
    fi
    RESID_ARGS=(--resid "$RUN/resid")
fi
if [[ ! -f "$RUN/directions/directions.json" ]]; then
    # The ${x[@]+...} form keeps bash < 4.4 happy with an empty array under set -u.
    ws python -m ws3.directions --acts "$RUN/acts" ${RESID_ARGS[@]+"${RESID_ARGS[@]}"} --out "$RUN/directions"
fi
echo "inputs ready at $(date -Is)"

ANALYSIS_PID=""
if [[ "$JOB" == A && ! -f "$RUN/analysis/DONE" ]]; then
    # CPU-only localization in fp16 for the report: assigned E/A, user E, same-pole null.
    (
        set -e
        rm -rf "$RUN/analysis"; mkdir -p "$RUN/analysis"
        ws ws2 analyze --acts "$RUN/acts" --trait E --source assigned --permutations 999 --out "$RUN/analysis/E-assigned"
        ws ws2 analyze --acts "$RUN/acts" --trait A --source assigned --permutations 999 --out "$RUN/analysis/A-assigned"
        ws ws2 analyze --acts "$RUN/acts" --trait E --source user --permutations 999 --out "$RUN/analysis/E-user"
        ws python -m ws3.diagnostic --acts "$RUN/acts" --traits E A --permutations 999 \
            --reference-metrics "$RUN/analysis/E-assigned/metrics.json" "$RUN/analysis/A-assigned/metrics.json" \
            --out "$RUN/analysis/same-pole"
        touch "$RUN/analysis/DONE"
    ) > "$RUN/analysis.log" 2>&1 &
    ANALYSIS_PID=$!
fi

PROJECT=(python -m ws3.project --job "$JOB" --seed "$SEED" --directions "$RUN/directions"
         --out "$RUN/arms" --config "$CONFIG" --batch-size "$BATCH"
         --max-new-tokens "${MAX_NEW_TOKENS:-150}")
if (( ${#DEVS[@]} >= 4 )); then
    CUDA_VISIBLE_DEVICES="${DEVS[0]},${DEVS[1]}" ws "${PROJECT[@]}" --worker 0 --workers 2 > "$RUN/worker0.log" 2>&1 &
    W0=$!
    CUDA_VISIBLE_DEVICES="${DEVS[2]},${DEVS[3]}" ws "${PROJECT[@]}" --worker 1 --workers 2 > "$RUN/worker1.log" 2>&1 &
    W1=$!
    STATUS=0
    wait $W0 || STATUS=1
    wait $W1 || STATUS=1
    tail -n 5 "$RUN/worker0.log" "$RUN/worker1.log"
else
    STATUS=0
    CUDA_VISIBLE_DEVICES="${DEVS[0]},${DEVS[1]}" ws "${PROJECT[@]}" 2>&1 | tee "$RUN/worker0.log" || STATUS=1
fi
if [[ -n "$ANALYSIS_PID" ]]; then
    wait "$ANALYSIS_PID" || echo "analysis failed; see $RUN/analysis.log"
fi

# Small, portable bundle to copy back for judging (no activations or weights).
tar -czf "$RUN/export_job$JOB.tar.gz" -C "$RUN" \
    --exclude='*.npy' --exclude='acts' --exclude='prepared' --exclude='resid' \
    arms directions $( [[ -d "$RUN/analysis" ]] && echo analysis ) $(ls "$RUN" | grep -E '\.log$' || true)
echo "status=$STATUS end=$(date -Is) export=$RUN/export_job$JOB.tar.gz"
exit $STATUS
