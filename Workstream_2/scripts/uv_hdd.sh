#!/usr/bin/env bash
# Route uv packages and Hugging Face weights to the mounted hard disk.
set -euo pipefail
WS2_STORAGE_ROOT="${WS2_STORAGE_ROOT:-/media/gaurav/Data21/eshaan}"
if [[ ! -d "${WS2_STORAGE_ROOT}" ]]; then
    echo "Storage root is unavailable: ${WS2_STORAGE_ROOT}. Set WS2_STORAGE_ROOT to a mounted data drive." >&2
    exit 1
fi
export UV_CACHE_DIR="${WS2_STORAGE_ROOT}/cache/uv"
export UV_PROJECT_ENVIRONMENT="${WS2_STORAGE_ROOT}/envs/user_induced_persona"
export HF_HOME="${WS2_STORAGE_ROOT}/cache/huggingface"
export HF_HUB_CACHE="${WS2_STORAGE_ROOT}/models/huggingface"
export HF_XET_CACHE="${HF_HOME}/xet"
export WS2_RESULTS_ROOT="${WS2_STORAGE_ROOT}/results/user_induced_persona"
mkdir -p "${UV_CACHE_DIR}" "${HF_HUB_CACHE}" "${WS2_RESULTS_ROOT}"
exec uv "$@"
