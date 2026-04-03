#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -t 48:00:00
#SBATCH -J cosmos_robocasa_eval
#SBATCH -o cosmos_robocasa_eval_out.%j
#SBATCH -e cosmos_robocasa_eval_err.%j
#SBATCH -p seas_gpu,gpu
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1
#SBATCH --mem=256000

echo "=== Job info ==="
echo "JobID: ${SLURM_JOB_ID:-}"
echo "Node:   $(hostname)"
echo "PWD:    $(pwd)"
echo "================"

source ~/.bashrc

cd /n/netscratch/hankyang_lab/Lab/tdieudonne/cosmos-policy
REPO_DIR="/n/netscratch/hankyang_lab/Lab/tdieudonne/cosmos-policy"
SIF_FILE="${REPO_DIR}/cosmos-policy.sif"

# Auto-detect EGL device ID based on GPU configuration
# MIG GPUs typically have CUDA_VISIBLE_DEVICES set to a UUID and need device 4
# Regular GPUs have numeric CUDA_VISIBLE_DEVICES and use device 0
export MUJOCO_GL=egl
if [[ "${CUDA_VISIBLE_DEVICES:-}" =~ ^MIG- ]]; then
    # MIG configuration detected
    export MUJOCO_EGL_DEVICE_ID=4
    echo "MIG GPU detected - using EGL device 4"
else
    # Regular GPU configuration
    export MUJOCO_EGL_DEVICE_ID=0
    echo "Regular GPU detected - using EGL device 0"
fi
export XLA_PYTHON_CLIENT_MEM_FRACTION=1.0

nvidia-smi || true

TASKS=(
    PnPCounterToCab PnPCabToCounter PnPCounterToSink PnPSinkToCounter
    PnPCounterToMicrowave PnPMicrowaveToCounter PnPCounterToStove PnPStoveToCounter
    OpenSingleDoor CloseSingleDoor OpenDoubleDoor CloseDoubleDoor
    OpenDrawer CloseDrawer
    TurnOnSinkFaucet TurnOffSinkFaucet TurnSinkSpout
    TurnOnStove TurnOffStove
    CoffeeSetupMug CoffeeServeMug CoffeePressButton
    TurnOnMicrowave TurnOffMicrowave
)

for TASK in "${TASKS[@]}"; do
    echo "=== Launching eval for task: ${TASK} ==="
    singularity exec --nv \
        --env SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
        --env REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
        --env CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
        --env UV_LINK_MODE=copy \
        ${HF_TOKEN:+--env HF_TOKEN="${HF_TOKEN}"} \
        --bind "${HOME}/.cache:${HOME}/.cache" \
        --bind "${REPO_DIR}:/workspace" \
        --bind /usr/share/glvnd/egl_vendor.d:/usr/share/glvnd/egl_vendor.d \
        --pwd /workspace \
        "${SIF_FILE}" \
        bash -c "
            cd /workspace && \
            uv run --extra cu128 --group robocasa --python 3.10 \
                python -m cosmos_policy.experiments.robot.robocasa.run_robocasa_eval \
                --config cosmos_predict2_2b_480p_robocasa_50_demos_per_task__inference \
                --ckpt_path nvidia/Cosmos-Policy-RoboCasa-Predict2-2B \
                --config_file cosmos_policy/config/config.py \
                --use_wrist_image True \
                --num_wrist_images 1 \
                --use_proprio True \
                --normalize_proprio True \
                --unnormalize_actions True \
                --dataset_stats_path nvidia/Cosmos-Policy-RoboCasa-Predict2-2B/robocasa_dataset_statistics.json \
                --t5_text_embeddings_path nvidia/Cosmos-Policy-RoboCasa-Predict2-2B/robocasa_t5_embeddings.pkl \
                --trained_with_image_aug True \
                --chunk_size 32 \
                --num_open_loop_steps 16 \
                --task_name ${TASK} \
                --num_trials_per_task 50 \
                --run_id_note chkpt45000--5stepAct--seed195--deterministic \
                --local_log_dir cosmos_policy/experiments/robot/robocasa/logs/ \
                --seed 195 \
                --randomize_seed False \
                --deterministic True \
                --use_variance_scale False \
                --use_jpeg_compression True \
                --flip_images True \
                --num_denoising_steps_action 5 \
                --num_denoising_steps_future_state 1 \
                --num_denoising_steps_value 1 \
                --data_collection False
        "
    echo "=== Done: ${TASK} ==="
done