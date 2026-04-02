"""
Evaluate a Cosmos Policy checkpoint on the new RoboCasa (v1.0) benchmark
using a client-server architecture.

This script is the CLIENT: it runs the new robocasa environment (gymnasium)
and queries the cosmos-policy server for actions.

Runs in the robocasa conda environment (not inside Singularity).

Usage:
    python run_robocasa_eval_v2.py \
        --server_url http://localhost:8777 \
        --tasks CloseFridge CloseCabinet \
        --split pretrain \
        --num_trials 50 \
        --log_dir ./eval_logs
"""

import argparse
import base64
import collections
import io
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path

import imageio
import numpy as np
import requests
import tqdm

# New robocasa imports
import robocasa.utils.env_utils as env_utils
from robocasa.utils.dataset_registry import ATOMIC_TASK_DATASETS, COMPOSITE_TASK_DATASETS
from robocasa.utils.dataset_registry_utils import get_task_horizon

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---- Helpers ----

def encode_image(img: np.ndarray) -> str:
    """Encode uint8 numpy image to base64 string (numpy format)."""
    buf = io.BytesIO()
    np.save(buf, img, allow_pickle=False)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def query_server(server_url: str, obs: dict) -> dict:
    """Send observation to cosmos-policy server and get actions back."""
    payload = {
        "primary_image": encode_image(obs["primary_image"]),
        "secondary_image": encode_image(obs["secondary_image"]),
        "wrist_image": encode_image(obs["wrist_image"]),
        "proprio": obs["proprio"].tolist(),
        "task_description": obs["task_description"],
    }
    response = requests.post(f"{server_url}/act", json=payload, timeout=120)
    response.raise_for_status()
    return response.json()


def wait_for_server(server_url: str, timeout: int = 300, interval: int = 5):
    """Wait for the server to become ready."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(f"{server_url}/health", timeout=5)
            if r.status_code == 200:
                logger.info("Server is ready.")
                return
        except requests.ConnectionError:
            pass
        logger.info(f"Waiting for server at {server_url} ...")
        time.sleep(interval)
    raise TimeoutError(f"Server at {server_url} did not become ready within {timeout}s")


def prepare_observation(raw_obs, flip_images: bool = True) -> dict:
    """
    Extract images and proprio from raw robosuite observations.

    Builds the 9D proprio that the old cosmos-policy checkpoint expects:
        gripper_qpos(2) + eef_pos(3) + eef_quat(4)

    Images come from:
        robot0_agentview_left_image, robot0_agentview_right_image, robot0_eye_in_hand_image
    """
    # Extract images
    primary_img = raw_obs["robot0_agentview_left_image"]
    secondary_img = raw_obs["robot0_agentview_right_image"]
    wrist_img = raw_obs["robot0_eye_in_hand_image"]

    if flip_images:
        primary_img = np.flipud(primary_img).copy()
        secondary_img = np.flipud(secondary_img).copy()
        wrist_img = np.flipud(wrist_img).copy()

    # Build 9D proprio matching the old training format:
    #   gripper_qpos(2) + eef_pos(3) + eef_quat(4)
    proprio = np.concatenate([
        raw_obs["robot0_gripper_qpos"],
        raw_obs["robot0_eef_pos"],
        raw_obs["robot0_eef_quat"],
    ]).astype(np.float32)

    return {
        "primary_image": primary_img,
        "secondary_image": secondary_img,
        "wrist_image": wrist_img,
        "proprio": proprio,
    }


def pad_action_for_env(action_7d: np.ndarray) -> np.ndarray:
    """
    Pad 7D action (xyz+rpy+gripper) to 12D for PandaMobile env.
    Appends [0, 0, 0, 0, -1] for mobile base (zero motion, arm control mode).
    Same logic as old run_robocasa_eval.py line 731.
    """
    mobile_base_action = np.array([0.0, 0.0, 0.0, 0.0, -1.0])
    return np.concatenate([action_7d, mobile_base_action])


# ---- Main evaluation ----

def eval_task(
    env_name: str,
    split: str,
    server_url: str,
    num_trials: int,
    log_dir: str,
    seed: int = 7,
    num_open_loop_steps: int = 16,
    env_img_res: int = 224,
):
    """Run evaluation for a single task."""
    task_horizon = get_task_horizon(env_name)
    max_steps = int(task_horizon * 1.5)  # extra time for the policy

    now = datetime.now().strftime("%Y-%m-%d-%H-%M")
    task_log_dir = os.path.join(log_dir, "evals", split, env_name, now)

    # Skip if already evaluated
    for root, dirs, files in os.walk(os.path.dirname(task_log_dir)):
        if "stats.json" in files:
            logger.info(f"{env_name}/{split} already evaluated, skipping.")
            return

    os.makedirs(task_log_dir, exist_ok=True)

    # Create environment using robosuite directly (not gym wrapper)
    # This gives us raw observations including robot0_eef_pos
    env = env_utils.create_env(
        env_name=env_name,
        robots="PandaOmron",
        camera_names=["robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand"],
        camera_widths=env_img_res,
        camera_heights=env_img_res,
        seed=seed,
        render_onscreen=False,
        split=split,
    )

    total_successes = 0

    for episode_idx in tqdm.tqdm(range(num_trials), desc=f"{env_name}"):
        raw_obs = env.reset()

        # Get language instruction
        ep_meta = env.get_ep_meta()
        task_description = ep_meta.get("lang", env_name)

        # Wait for objects to settle
        for _ in range(10):
            dummy_action = np.zeros(env.action_dim)
            raw_obs, _, _, _ = env.step(dummy_action)

        action_queue = collections.deque()
        success = False
        replay_images = []

        for t in range(max_steps):
            obs = prepare_observation(raw_obs, flip_images=True)
            obs["task_description"] = task_description

            # Save frame for replay
            replay_images.append(obs["primary_image"].copy())

            # Query server for new action chunk when queue is empty
            if not action_queue:
                result = query_server(server_url, obs)
                action_chunk = result["actions"]
                for a in action_chunk[:num_open_loop_steps]:
                    action_queue.append(np.array(a, dtype=np.float32))

            # Execute action
            action_7d = action_queue.popleft()
            action_12d = pad_action_for_env(action_7d)
            raw_obs, reward, done, info = env.step(action_12d)

            # Check success
            if env._check_success():
                success = True
                logger.info(f"  Episode {episode_idx}: SUCCESS at step {t}")
                break

        if success:
            total_successes += 1

        # Save replay video
        if replay_images:
            video_path = os.path.join(
                task_log_dir,
                f"ep{episode_idx:03d}_{'success' if success else 'fail'}.mp4",
            )
            with imageio.get_writer(video_path, fps=20) as writer:
                for img in replay_images:
                    writer.append_data(img)

        logger.info(
            f"  Episode {episode_idx}: {'SUCCESS' if success else 'FAILURE'} "
            f"(running: {total_successes}/{episode_idx + 1})"
        )

    # Save stats
    success_rate = total_successes / num_trials if num_trials > 0 else 0.0
    stats = {
        "task": env_name,
        "split": split,
        "num_trials": num_trials,
        "successes": total_successes,
        "success_rate": success_rate,
    }
    stats_path = os.path.join(task_log_dir, "stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"{env_name}: {success_rate:.1%} ({total_successes}/{num_trials})")

    env.close()
    return stats


def main():
    parser = argparse.ArgumentParser(description="Evaluate Cosmos Policy on new RoboCasa")
    parser.add_argument("--server_url", type=str, default="http://localhost:8777")
    parser.add_argument("--tasks", nargs="+", required=True,
                        help="Task names (e.g. CloseFridge CloseCabinet) or 'all_atomic'")
    parser.add_argument("--split", type=str, default="pretrain", choices=["pretrain", "target"])
    parser.add_argument("--num_trials", type=int, default=50)
    parser.add_argument("--num_open_loop_steps", type=int, default=16)
    parser.add_argument("--env_img_res", type=int, default=224)
    parser.add_argument("--log_dir", type=str, default="./eval_logs")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    # Resolve task list
    task_names = []
    for t in args.tasks:
        if t == "all_atomic":
            task_names.extend(ATOMIC_TASK_DATASETS.keys())
        elif t == "all_composite":
            task_names.extend(COMPOSITE_TASK_DATASETS.keys())
        else:
            task_names.append(t)

    logger.info(f"Evaluating {len(task_names)} tasks: {task_names}")

    # Wait for server
    wait_for_server(args.server_url)

    # Run evaluations
    all_stats = []
    for task_name in task_names:
        stats = eval_task(
            env_name=task_name,
            split=args.split,
            server_url=args.server_url,
            num_trials=args.num_trials,
            log_dir=args.log_dir,
            seed=args.seed,
            num_open_loop_steps=args.num_open_loop_steps,
            env_img_res=args.env_img_res,
        )
        if stats:
            all_stats.append(stats)

    # Print summary
    if all_stats:
        print("\n=== Evaluation Summary ===")
        total_s, total_n = 0, 0
        for s in all_stats:
            print(f"  {s['task']:30s}  {s['success_rate']:.1%}  ({s['successes']}/{s['num_trials']})")
            total_s += s["successes"]
            total_n += s["num_trials"]
        print(f"  {'OVERALL':30s}  {total_s/total_n:.1%}  ({total_s}/{total_n})")

        # Save aggregate stats
        summary_path = os.path.join(args.log_dir, "summary.json")
        with open(summary_path, "w") as f:
            json.dump(all_stats, f, indent=2)


if __name__ == "__main__":
    main()
