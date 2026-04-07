"""
run_robocasa_eval_gemma.py

Evaluates Cosmos Policy with a Gemma 4 VLM as a high-level error-detection planner.

Gemma observes the scene every `gemma_replan_every` action chunks and generates a
refined instruction — primarily to detect failure states (e.g. missed grasp) and
redirect the low-level policy.  At 20 Hz with 16 open-loop steps per chunk, calling
every 3 chunks means Gemma intervenes every ~2.4 seconds.

All Cosmos Policy flags are identical to eval_cosmos.sh for fair comparison.

Usage (inside Singularity via uv run):
    python -m cosmos_policy.experiments.robot.robocasa.run_robocasa_eval_gemma \
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
        --task_name PnPCabToCounter \
        --num_trials_per_task 50 \
        --seed 195 \
        --randomize_seed False \
        --deterministic True \
        --use_variance_scale False \
        --use_jpeg_compression True \
        --flip_images True \
        --num_denoising_steps_action 5 \
        --num_denoising_steps_future_state 1 \
        --num_denoising_steps_value 1 \
        --gemma_model_id Qwen/Qwen3.5-35B-A3B \
        --gemma_replan_every 3 \
        --gemma_device cuda:1 \
        --local_log_dir cosmos_policy/experiments/robot/robocasa/logs_gemma/
"""

import ast
import os
import pickle
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import draccus
import numpy as np
import robocasa
import robosuite
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor
from qwen_vl_utils import process_vision_info

from cosmos_policy.experiments.robot.cosmos_utils import (
    get_action,
    get_model,
    init_t5_text_embeddings_cache,
    load_dataset_stats,
)
from cosmos_policy.experiments.robot.robocasa.robocasa_utils import save_rollout_video
from cosmos_policy.experiments.robot.robot_utils import DATE_TIME, log_message, setup_logging
from cosmos_policy.utils.utils import set_seed_everywhere

CONTROLLER_CONFIGS_PATH = "cosmos_policy/experiments/robot/robocasa/robocasa_controller_configs.pkl"

TASK_MAX_STEPS = {
    "PnPCounterToCab": 500, "PnPCabToCounter": 500,
    "PnPCounterToSink": 700, "PnPSinkToCounter": 500,
    "PnPCounterToMicrowave": 600, "PnPMicrowaveToCounter": 500,
    "PnPCounterToStove": 500, "PnPStoveToCounter": 500,
    "OpenSingleDoor": 500, "CloseSingleDoor": 500,
    "OpenDoubleDoor": 1000, "CloseDoubleDoor": 700,
    "OpenDrawer": 500, "CloseDrawer": 500,
    "TurnOnStove": 500, "TurnOffStove": 500,
    "TurnOnSinkFaucet": 500, "TurnOffSinkFaucet": 500, "TurnSinkSpout": 500,
    "CoffeeSetupMug": 600, "CoffeeServeMug": 600, "CoffeePressButton": 300,
    "TurnOnMicrowave": 500, "TurnOffMicrowave": 500,
}

GEMMA_SYSTEM_PROMPT = (
    "You are monitoring a robot arm performing kitchen manipulation tasks. "
    "Respond with exactly one word: CONTINUE or RETRY.\n"
    "- CONTINUE: the robot is making visible progress toward its goal\n"
    "- RETRY: the robot has clearly failed (missed grasp, dropped object, visibly stuck) "
    "and should replan from its current position\n"
    "Output ONLY the single word, nothing else."
)


@dataclass
class PolicyEvalConfig:
    # fmt: off
    suite: str = "robocasa"

    # --- Cosmos Policy (must match eval_cosmos.sh exactly) ---
    config: str = ""
    ckpt_path: str = ""
    config_file: str = "cosmos_policy/config/config.py"
    planning_model_config_name: str = ""
    planning_model_ckpt_path: str = ""

    use_third_person_image: bool = True
    num_third_person_images: int = 2
    use_wrist_image: bool = True
    num_wrist_images: int = 1
    use_proprio: bool = True
    flip_images: bool = True
    use_variance_scale: bool = False
    use_jpeg_compression: bool = True
    ar_future_prediction: bool = False
    ar_value_prediction: bool = False
    ar_qvalue_prediction: bool = False
    num_denoising_steps_action: int = 5
    num_denoising_steps_future_state: int = 1
    num_denoising_steps_value: int = 1
    unnormalize_actions: bool = True
    normalize_proprio: bool = True
    dataset_stats_path: str = ""
    t5_text_embeddings_path: str = ""
    trained_with_image_aug: bool = True
    chunk_size: int = 32
    num_open_loop_steps: int = 16

    deterministic: bool = True
    seed: int = 195
    randomize_seed: bool = False

    # --- RoboCasa env ---
    task_name: str = "PnPCabToCounter"
    num_trials_per_task: int = 50
    env_img_res: int = 224
    robots: str = "PandaMobile"
    controllers: str = "OSC_POSE"
    obj_instance_split: str = "B"
    layout_and_style_ids: str = "((1,1),(2,2),(4,4),(6,9),(7,10))"
    randomize_cameras: bool = False

    # --- Qwen3.5 VLM planner ---
    gemma_model_id: str = "Qwen/Qwen3.5-35B-A3B"
    gemma_replan_every: int = 3    # call Gemma every N action chunks (3 chunks = ~2.4s at 20Hz)
    gemma_device: str = "cuda:1"   # device for Gemma (use cuda:0 if single GPU + 4bit)
    gemma_load_in_4bit: bool = False  # 4-bit quant: ~16GB VRAM instead of ~62GB

    # --- Logging ---
    local_log_dir: str = "./experiments/logs_gemma"
    run_id_note: Optional[str] = None

    data_collection: bool = False
    # fmt: on


# ---- VLM planner ----

class GemmaPlanner:
    def __init__(self, model_id: str, device: str, load_in_4bit: bool):
        print(f"Loading VLM processor: {model_id}")
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

        print(f"Loading VLM model (4bit={load_in_4bit}) ...")
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            bnb = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_id, quantization_config=bnb, device_map=device, trust_remote_code=True
            )
        else:
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_id, torch_dtype=torch.bfloat16, device_map=device, trust_remote_code=True
            )
        self.model.eval()
        print("Gemma planner ready.")

    @torch.inference_mode()
    def plan(
        self,
        frame_buffer: list,  # list of PIL Images (primary camera), subsampled at FRAME_SUBSAMPLE
        task_description: str,
        step: int,
    ) -> tuple:
        """
        Returns (raw_response: str, should_retry: bool).
        Uses process_vision_info (qwen_vl_utils) so video frames are encoded correctly.
        frame_buffer is sampled at 20Hz / FRAME_SUBSAMPLE = 5fps.
        """
        if step < 100:
            stage = "Early stage: robot should be approaching and opening the cabinet."
        elif step < 250:
            stage = "Mid stage: robot should be reaching in and grasping the object."
        else:
            stage = "Late stage: robot should have the object and be placing it on the counter."

        user_text = (
            f"Robot task: {task_description}\n"
            f"Current state: {stage}\n"
            "The video shows the robot's trajectory from the start of the episode.\n"
            "Did the robot fail (missed grasp, dropped object, or visibly stuck)?\n"
            "Answer with a single letter: A for CONTINUE, B for RETRY."
        )

        # Video content block: no "type" key — qwen_vl_utils requires this exact format
        messages = [
            {"role": "user", "content": [
                {
                    "video": frame_buffer,   # list of PIL Images
                    "sample_fps": 5,         # 20Hz env / 4 subsample = 5fps
                    "max_frames": 128,
                    "total_pixels": 20480 * 28 * 28,
                    "min_pixels": 16 * 28 * 28,
                },
                {"type": "text", "text": user_text},
            ]},
        ]

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        image_inputs, video_inputs, video_kwargs = process_vision_info(
            [messages],
            return_video_kwargs=True,
            image_patch_size=16,
            return_video_metadata=True,
        )
        if video_inputs is not None:
            video_inputs, video_metadatas = zip(*video_inputs)
            video_inputs, video_metadatas = list(video_inputs), list(video_metadatas)
        else:
            video_metadatas = None

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            video_metadata=video_metadatas,
            **video_kwargs,
            do_resize=False,
            return_tensors="pt",
        ).to(self.model.device)

        out_ids = self.model.generate(
            **inputs,
            max_new_tokens=5,
            do_sample=False,
        )
        new_tokens = out_ids[0, inputs["input_ids"].shape[1]:]
        raw_response = self.processor.decode(new_tokens, skip_special_tokens=True).strip()

        is_retry = raw_response.upper().startswith("B") or "RETRY" in raw_response.upper()
        return raw_response, is_retry


# ---- Environment ----

def create_robocasa_env(cfg: PolicyEvalConfig, seed=None, episode_idx=None):
    all_layout_style_ids = ast.literal_eval(cfg.layout_and_style_ids)
    if episode_idx is not None:
        scene_index = (episode_idx // 10) % len(all_layout_style_ids)
        layout_and_style_ids = (all_layout_style_ids[scene_index],)
    else:
        layout_and_style_ids = all_layout_style_ids

    with open(CONTROLLER_CONFIGS_PATH, "rb") as f:
        controller_configs = pickle.load(f)

    env = robosuite.make(
        env_name=cfg.task_name,
        robots=cfg.robots,
        controller_configs=controller_configs,
        camera_names=["robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand"],
        camera_widths=cfg.env_img_res,
        camera_heights=cfg.env_img_res,
        has_renderer=False,
        has_offscreen_renderer=True,
        ignore_done=True,
        use_object_obs=True,
        use_camera_obs=True,
        camera_depths=False,
        seed=seed,
        obj_instance_split=cfg.obj_instance_split,
        generative_textures=None,
        randomize_cameras=cfg.randomize_cameras,
        layout_and_style_ids=layout_and_style_ids,
        translucent_robot=False,
    )
    return env


def prepare_observation(obs, flip_images: bool):
    img_left = obs["robot0_agentview_left_image"]
    img_right = obs["robot0_agentview_right_image"]
    img_wrist = obs["robot0_eye_in_hand_image"]
    if flip_images:
        img_left = np.flipud(img_left)
        img_right = np.flipud(img_right)
        img_wrist = np.flipud(img_wrist)
    proprio = np.concatenate((obs["robot0_gripper_qpos"], obs["robot0_eef_pos"], obs["robot0_eef_quat"]))
    return {
        "primary_image": img_left,
        "secondary_image": img_right,
        "wrist_image": img_wrist,
        "proprio": proprio,
    }


# ---- Eval loop ----

def run_eval(cfg: PolicyEvalConfig):
    set_seed_everywhere(cfg.seed)

    # Load Cosmos Policy
    print("Loading Cosmos Policy model...")
    model, config = get_model(cfg)
    dataset_stats = load_dataset_stats(cfg.dataset_stats_path)
    if cfg.t5_text_embeddings_path:
        init_t5_text_embeddings_cache(cfg.t5_text_embeddings_path)

    # Load VLM planner
    vlm = GemmaPlanner(cfg.gemma_model_id, cfg.gemma_device, cfg.gemma_load_in_4bit)

    # Setup logging
    log_file, log_path, run_id = setup_logging(
        cfg, cfg.task_name, cfg.local_log_dir, run_id_note=cfg.run_id_note, use_wandb=False
    )

    log_message(f"Task: {cfg.task_name} | Trials: {cfg.num_trials_per_task} | "
                f"VLM replan every {cfg.gemma_replan_every} chunks", log_file)

    max_steps = TASK_MAX_STEPS.get(cfg.task_name, 500)
    total_successes = 0

    for episode_idx in range(cfg.num_trials_per_task):
        env = create_robocasa_env(cfg, seed=cfg.seed + episode_idx, episode_idx=episode_idx)
        obs = env.reset()

        # Get language instruction from environment
        task_description = env.get_ep_meta().get("lang", cfg.task_name)

        # Wait for objects to settle
        for _ in range(10):
            obs, _, _, _ = env.step(np.zeros(env.action_spec[0].shape))

        action_queue = deque()
        chunk_count = 0
        success = False
        retry_count = 0
        replay_primary_images = []
        replay_secondary_images = []
        replay_wrist_images = []
        # VLM is called every gemma_replan_every * num_open_loop_steps timesteps.
        # This is independent of the action queue so RETRY can truncate mid-chunk.
        vlm_call_interval = cfg.gemma_replan_every * cfg.num_open_loop_steps
        # Frame buffer: collect primary-camera frames between VLM calls.
        # Subsample every 4 steps to keep token count reasonable (~12 frames per call).
        frame_buffer = []
        FRAME_SUBSAMPLE = 4

        for t in range(max_steps):
            observation = prepare_observation(obs, cfg.flip_images)

            # Collect full-res frames for replay video
            replay_primary_images.append(observation["primary_image"])
            replay_secondary_images.append(observation["secondary_image"])
            replay_wrist_images.append(observation["wrist_image"])

            # Accumulate subsampled frames for VLM video context
            if t % FRAME_SUBSAMPLE == 0:
                frame_buffer.append(Image.fromarray(observation["primary_image"]))

            # Call VLM on a fixed timestep cadence (every ~2.4s at 20Hz with defaults)
            if t > 0 and t % vlm_call_interval == 0:
                t0 = time.time()
                raw_response, should_retry = vlm.plan(
                    frame_buffer,
                    task_description,
                    step=t,
                )
                vlm_ms = (time.time() - t0) * 1000
                if should_retry:
                    retry_count += 1
                    action_queue.clear()  # truncate current chunk, force Cosmos to replan
                    log_message(
                        f"  [ep{episode_idx} t={t}] VLM='{raw_response}' → RETRY #{retry_count} "
                        f"({vlm_ms:.0f}ms) — queue cleared, replanning",
                        log_file,
                    )
                else:
                    log_message(
                        f"  [ep{episode_idx} t={t}] VLM='{raw_response}' → CONTINUE ({vlm_ms:.0f}ms)",
                        log_file,
                    )

            if len(action_queue) == 0:
                # Query Cosmos Policy with the original cached instruction (no new T5 embedding)
                result = get_action(
                    cfg,
                    model,
                    dataset_stats,
                    observation,
                    task_description,
                    seed=cfg.seed,
                    randomize_seed=cfg.randomize_seed,
                    num_denoising_steps_action=cfg.num_denoising_steps_action,
                    generate_future_state_and_value_in_parallel=True,
                )
                actions = result["actions"][:cfg.num_open_loop_steps]
                action_queue.extend(actions)
                chunk_count += 1

            action = action_queue.popleft()
            # Pad 7D -> 12D for PandaMobile
            if action.shape[-1] == 7 and env.action_dim == 12:
                action = np.concatenate([action, [0.0, 0.0, 0.0, 0.0, -1.0]])

            obs, _, _, _ = env.step(action)

            if env._check_success():
                success = True
                log_message(f"  Episode {episode_idx}: SUCCESS at t={t}", log_file)
                break

        if success:
            total_successes += 1

        # Save rollout video (primary | secondary | wrist concatenated horizontally)
        rollout_data_dir = os.path.join(cfg.local_log_dir, "rollout_data", f"{cfg.task_name}--{DATE_TIME}")
        os.makedirs(rollout_data_dir, exist_ok=True)
        save_rollout_video(
            replay_primary_images,
            replay_secondary_images,
            replay_wrist_images,
            episode_idx,
            success=success,
            task_description=task_description,
            rollout_data_dir=rollout_data_dir,
            log_file=log_file,
        )

        log_message(
            f"Episode {episode_idx}: {'SUCCESS' if success else 'FAILURE'} "
            f"({total_successes}/{episode_idx + 1}) | VLM retries: {retry_count}",
            log_file,
        )
        env.close()

    success_rate = total_successes / cfg.num_trials_per_task
    log_message(
        f"\n=== {cfg.task_name}: {success_rate:.1%} ({total_successes}/{cfg.num_trials_per_task}) ===",
        log_file,
    )
    return success_rate


@draccus.wrap()
def main(cfg: PolicyEvalConfig):
    run_eval(cfg)


if __name__ == "__main__":
    main()
