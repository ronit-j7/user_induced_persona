#!/usr/bin/env bash
#SBATCH --job-name=ap-smoke
#SBATCH --partition=u22
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=00:45:00
#SBATCH --output=ap-smoke-%j.out
# Pre-flight on 2 GPUs: sbatch Workstream_3/scripts/ada_smoke.sh (from the repo root),
# or run it directly inside an interactive GPU session.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
source Workstream_3/scripts/ada_env.sh
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
nvidia-smi
CONFIG="${CONFIG:-Workstream_2/configs/qwen_ada.json}"
uv run --no-sync python -m ws3.ada_smoke --config "$CONFIG" --out "$ADA_RESULTS/smoke_$(date +%Y%m%d_%H%M%S).json"
