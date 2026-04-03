#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -t 48:00:00
#SBATCH -J cosmos_robocasa_v2_eval
#SBATCH -o out.%j
#SBATCH -e err.%j
#SBATCH -p seas_gpu,gpu
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1
#SBATCH --mem=256000

echo "=== Job info ==="
echo "JobID: ${SLURM_JOB_ID:-}"
echo "Node:  $(hostname)"
echo "PWD:   $(pwd)"
echo "================"

# ---- Configuration ----
REPO_DIR="/n/netscratch/hankyang_lab/Lab/tdieudonne/cosmos-policy"
SIF_FILE="${REPO_DIR}/cosmos-policy.sif"
PORT=8777

# Cosmos policy checkpoint (downloads from HF on first run)
CONFIG="cosmos_predict2_2b_480p_robocasa_50_demos_per_task__inference"
CKPT="nvidia/Cosmos-Policy-RoboCasa-Predict2-2B"
STATS="nvidia/Cosmos-Policy-RoboCasa-Predict2-2B/robocasa_dataset_statistics.json"
T5_EMB="nvidia/Cosmos-Policy-RoboCasa-Predict2-2B/robocasa_t5_embeddings.pkl"

# Eval settings
TASKS="CloseFridge"  # space-separated task names, or "all_atomic"
SPLIT="pretrain"
NUM_TRIALS=50
LOG_DIR="${REPO_DIR}/eval_logs_v2"

# Python interpreters
CONDA_BASE="/n/home08/tdieudonne/miniconda3"
PYTHON_ROBOCASA="${CONDA_BASE}/envs/robocasa/bin/python"

# ---- Environment Setup ----
source ~/.bashrc

# MuJoCo EGL rendering
export MUJOCO_GL=egl
if [[ "${CUDA_VISIBLE_DEVICES:-}" =~ ^MIG- ]]; then
    export MUJOCO_EGL_DEVICE_ID=4
    echo "MIG GPU detected - using EGL device 4"
else
    export MUJOCO_EGL_DEVICE_ID=0
    echo "Regular GPU detected - using EGL device 0"
fi

# HF token
if [[ -z "${HF_TOKEN:-}" && -f "${HOME}/.hf_token" ]]; then
    export HF_TOKEN="$(cat "${HOME}/.hf_token")"
fi

nvidia-smi || true

# ==============================================================================
# STEP 1: Start Cosmos Policy server (inside Singularity)
# ==============================================================================
echo "=== Launching Cosmos Policy Server ==="

singularity exec --nv \
    --env SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
    --env REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
    --env CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
    --env UV_LINK_MODE=copy \
    ${HF_TOKEN:+--env HF_TOKEN="${HF_TOKEN}"} \
    --bind "${HOME}/.cache:${HOME}/.cache" \
    --bind "${REPO_DIR}:/workspace" \
    --pwd /workspace \
    "${SIF_FILE}" \
    bash -c "
        cd /workspace && \
        uv run --extra cu128 --group robocasa --python 3.10 \
            python -m cosmos_policy.experiments.robot.robocasa.serve_cosmos_policy \
            --config ${CONFIG} \
            --ckpt_path ${CKPT} \
            --dataset_stats_path ${STATS} \
            --t5_text_embeddings_path ${T5_EMB} \
            --port ${PORT}
    " &

SERVER_PID=$!

# Wait for server to load model (usually ~60-120s)
echo "Waiting for server to be ready on port ${PORT}..."
MAX_WAIT=300
ELAPSED=0
while ! curl -s "http://localhost:${PORT}/health" > /dev/null 2>&1; do
    sleep 10
    ELAPSED=$((ELAPSED + 10))
    if [ $ELAPSED -ge $MAX_WAIT ]; then
        echo "ERROR: Server did not become ready within ${MAX_WAIT}s"
        kill ${SERVER_PID} 2>/dev/null || true
        exit 1
    fi
    echo "  Still waiting... (${ELAPSED}s)"
done
echo "Server is ready!"

# ==============================================================================
# STEP 2: Run RoboCasa evaluation (in robocasa conda env)
# ==============================================================================
echo "=== Starting RoboCasa Evaluation ==="

$PYTHON_ROBOCASA \
    ${REPO_DIR}/cosmos_policy/experiments/robot/robocasa/run_robocasa_eval_v2.py \
    --server_url "http://localhost:${PORT}" \
    --tasks ${TASKS} \
    --split ${SPLIT} \
    --num_trials ${NUM_TRIALS} \
    --log_dir "${LOG_DIR}"

# ==============================================================================
# Cleanup
# ==============================================================================
echo "Cleaning up server..."
kill ${SERVER_PID} 2>/dev/null || true
wait ${SERVER_PID} 2>/dev/null || true

echo "=== Done ==="
