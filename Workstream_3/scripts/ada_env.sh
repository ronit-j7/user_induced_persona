# Source me. Keeps the uv env, uv cache, HF weights and results under ADA_STORE.
# ADA_STORE must be on a filesystem visible from both the login and compute nodes.
export ADA_STORE="${ADA_STORE:-$HOME/ap_store}"
export UV_CACHE_DIR="${ADA_STORE}/cache/uv"
export UV_PROJECT_ENVIRONMENT="${ADA_STORE}/envs/user_induced_persona"
export HF_HOME="${ADA_STORE}/cache/huggingface"
export HF_HUB_CACHE="${ADA_STORE}/models/huggingface"
export HF_XET_CACHE="${HF_HOME}/xet"
export ADA_RESULTS="${ADA_STORE}/results"
export PATH="$HOME/.local/bin:$PATH"
mkdir -p "$UV_CACHE_DIR" "$ADA_RESULTS" "$(dirname "$HF_HUB_CACHE")"
# A symlink (weights on a larger filesystem) must not be recreated.
if [[ ! -e "$HF_HUB_CACHE" ]]; then
    mkdir -p "$HF_HUB_CACHE"
fi
