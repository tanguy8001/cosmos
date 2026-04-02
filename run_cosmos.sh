#!/bin/bash
# Launch an interactive cosmos-policy container session via Singularity.
# Equivalent to the docker run command in SETUP.md.
#
# Usage (from any directory):
#   /path/to/cosmos-policy/run_cosmos.sh
#
# Inside the container the repo is mounted at /workspace and your
# ~/.cache is accessible (huggingface, uv, etc.).
#
# For Slurm jobs, request a GPU node first:
#   srun --gres=gpu:1 --pty bash
#   ./run_cosmos.sh
# Or embed singularity exec directly in your sbatch script (see bottom).

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
SIF_FILE="${REPO_DIR}/cosmos-policy.sif"

if [[ ! -f "${SIF_FILE}" ]]; then
    echo "ERROR: Image not found: ${SIF_FILE}"
    echo "Build it first with:  sbatch ${REPO_DIR}/build_singularity.sh"
    exit 1
fi

# Pass any extra arguments straight to bash (e.g. a script to run non-interactively)
CMD="${@:-bash}"

# Read HF token from file if not already set (store token in ~/.hf_token, never commit it)
if [[ -z "${HF_TOKEN:-}" && -f "${HOME}/.hf_token" ]]; then
    HF_TOKEN="$(cat "${HOME}/.hf_token")"
fi

exec singularity exec \
    --nv \
    --env SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
    --env REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
    --env CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
    --env UV_LINK_MODE=copy \
    ${HF_TOKEN:+--env HF_TOKEN="${HF_TOKEN}"} \
    --bind "${HOME}/.cache:${HOME}/.cache" \
    --bind "${REPO_DIR}:/workspace" \
    --pwd /workspace \
    "${SIF_FILE}" \
    ${CMD}

# ---------------------------------------------------------------------------
# Example sbatch snippet for non-interactive jobs:
#
# #!/bin/bash
# #SBATCH --gres=gpu:1
# #SBATCH --time=04:00:00
# #SBATCH --mem=32G
# REPO=/n/netscratch/hankyang_lab/Lab/tdieudonne/cosmos-policy
# singularity exec --nv \
#     --bind "${HOME}/.cache:${HOME}/.cache" \
#     --bind "${REPO}:/workspace" \
#     --pwd /workspace \
#     "${REPO}/cosmos-policy.sif" \
#     bash -c "just eval-libero"
# ---------------------------------------------------------------------------
