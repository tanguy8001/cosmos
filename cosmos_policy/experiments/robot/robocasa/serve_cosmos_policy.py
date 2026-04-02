"""
Cosmos Policy inference server for client-server evaluation.

Runs inside the Singularity container. Loads the model and serves
action predictions over HTTP so that a separate robocasa process
(in its own conda env) can query it.

Usage (inside singularity):
    uv run --extra cu128 --group robocasa --python 3.10 \
        python -m cosmos_policy.experiments.robot.robocasa.serve_cosmos_policy \
        --config cosmos_predict2_2b_480p_robocasa_50_demos_per_task__inference \
        --ckpt_path nvidia/Cosmos-Policy-RoboCasa-Predict2-2B \
        --dataset_stats_path nvidia/Cosmos-Policy-RoboCasa-Predict2-2B/robocasa_dataset_statistics.json \
        --t5_text_embeddings_path nvidia/Cosmos-Policy-RoboCasa-Predict2-2B/robocasa_t5_embeddings.pkl
"""

import argparse
import base64
import io
import json
import sys

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

# Cosmos policy imports
from cosmos_policy.experiments.robot.cosmos_utils import (
    get_action,
    get_model,
    init_t5_text_embeddings_cache,
    load_dataset_stats,
    unnormalize_actions,
)
from cosmos_policy.utils.utils import set_seed_everywhere


# ---- Request / Response schemas ----

class InferenceRequest(BaseModel):
    # Images as base64-encoded uint8 numpy arrays (H, W, 3)
    primary_image: str       # left agentview
    secondary_image: str     # right agentview
    wrist_image: str         # eye-in-hand
    proprio: list[float]     # 9D proprio vector
    task_description: str    # language instruction


class InferenceResponse(BaseModel):
    actions: list[list[float]]   # (chunk_size, action_dim)
    value: float


# ---- Helpers ----

def decode_image(b64: str) -> np.ndarray:
    """Decode base64 string to uint8 numpy image."""
    raw = base64.b64decode(b64)
    return np.load(io.BytesIO(raw), allow_pickle=False)


def encode_image(img: np.ndarray) -> str:
    """Encode uint8 numpy image to base64 string."""
    buf = io.BytesIO()
    np.save(buf, img, allow_pickle=False)
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ---- Argument parsing ----

def parse_args():
    p = argparse.ArgumentParser(description="Cosmos Policy inference server")
    p.add_argument("--config", type=str, required=True,
                   help="Inference config name (e.g. cosmos_predict2_2b_480p_robocasa_50_demos_per_task__inference)")
    p.add_argument("--ckpt_path", type=str, required=True,
                   help="Checkpoint path (local or HF repo id)")
    p.add_argument("--config_file", type=str, default="cosmos_policy/config/config.py")
    p.add_argument("--dataset_stats_path", type=str, required=True)
    p.add_argument("--t5_text_embeddings_path", type=str, default="")
    p.add_argument("--chunk_size", type=int, default=32)
    p.add_argument("--num_open_loop_steps", type=int, default=16)
    p.add_argument("--num_denoising_steps_action", type=int, default=5)
    p.add_argument("--seed", type=int, default=195)
    p.add_argument("--port", type=int, default=8777)
    p.add_argument("--host", type=str, default="0.0.0.0")
    return p.parse_args()


# ---- Build app ----

args = parse_args()

# Create a lightweight config namespace that get_action expects
class Cfg:
    pass

cfg = Cfg()
cfg.config = args.config
cfg.ckpt_path = args.ckpt_path
cfg.config_file = args.config_file
cfg.dataset_stats_path = args.dataset_stats_path
cfg.t5_text_embeddings_path = args.t5_text_embeddings_path
cfg.chunk_size = args.chunk_size
cfg.num_open_loop_steps = args.num_open_loop_steps
cfg.num_denoising_steps_action = args.num_denoising_steps_action
cfg.seed = args.seed
cfg.suite = "robocasa"
cfg.use_wrist_image = True
cfg.num_wrist_images = 1
cfg.use_third_person_image = True
cfg.num_third_person_images = 2
cfg.use_proprio = True
cfg.normalize_proprio = True
cfg.unnormalize_actions = True
cfg.trained_with_image_aug = True
cfg.use_jpeg_compression = True
cfg.flip_images = True
cfg.use_variance_scale = False
cfg.randomize_seed = False
cfg.planning_model_config_name = ""
cfg.planning_model_ckpt_path = ""
cfg.ar_future_prediction = False
cfg.ar_value_prediction = False
cfg.ar_qvalue_prediction = False

print("Loading model...")
model, config = get_model(cfg)
print("Loading dataset statistics...")
dataset_stats = load_dataset_stats(args.dataset_stats_path)
print("Loading T5 text embeddings...")
if args.t5_text_embeddings_path:
    init_t5_text_embeddings_cache(args.t5_text_embeddings_path)

set_seed_everywhere(args.seed)

app = FastAPI()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/act", response_model=InferenceResponse)
def act(req: InferenceRequest):
    # Decode images
    primary_image = decode_image(req.primary_image)
    secondary_image = decode_image(req.secondary_image)
    wrist_image = decode_image(req.wrist_image)

    # Build observation dict (same format as old run_robocasa_eval.prepare_observation)
    obs = {
        "primary_image": primary_image,
        "secondary_image": secondary_image,
        "wrist_image": wrist_image,
        "proprio": np.array(req.proprio, dtype=np.float32),
    }

    # Call the model
    result = get_action(
        cfg,
        model,
        dataset_stats,
        obs,
        req.task_description,
        seed=cfg.seed,
        randomize_seed=cfg.randomize_seed,
        num_denoising_steps_action=cfg.num_denoising_steps_action,
        generate_future_state_and_value_in_parallel=True,
    )

    actions = result["actions"]  # list of numpy arrays
    value = float(result.get("value_prediction", 0.0))

    # Only return the first num_open_loop_steps actions
    actions = actions[: cfg.num_open_loop_steps]
    actions_list = [a.tolist() if isinstance(a, np.ndarray) else a for a in actions]

    return InferenceResponse(actions=actions_list, value=value)


if __name__ == "__main__":
    print(f"Starting server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)
