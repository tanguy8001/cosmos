#!/usr/bin/env bash
# Run with:
# singularity exec --nv --bind "${HOME}/.cache:${HOME}/.cache" --bind "/n/netscratch/hankyang_lab/Lab/tdieudonne/cosmos-policy:/workspace" --pwd /workspace cosmos-policy.sif bash -c "uv run --extra cu128 --group robocasa --python 3.10 --with 'git+https://github.com/huggingface/transformers.git' --with 'qwen-vl-utils' python test_qwen_video.py --video_path cosmos_policy/wrist_view.mp4 --task_description 'pick the lemon from the cabinet and place it on the counter' --model_id Qwen/Qwen3.5-4B --device cuda:1 --fraction 0.5"
# fmt: off
"""
Standalone test: feed a rollout video (up to halfway) to Qwen and check
whether it outputs CONTINUE or RETRY — using the exact same format as
run_robocasa_eval_gemma.py.
"""

import argparse
import os

# SSL_CERT_FILE may be set to a Singularity-only path that doesn't exist on the host.
# Clear it so httpx/requests fall back to system certs. Also force offline mode since
# the model is already cached.
_cert = os.environ.get("SSL_CERT_FILE", "")
if _cert and not os.path.exists(_cert):
    del os.environ["SSL_CERT_FILE"]
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import imageio
import numpy as np
import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoModelForImageTextToText, AutoProcessor

FRAME_SUBSAMPLE = 4  # match eval script: every 4th frame → 5fps at 20Hz


def load_frames_to_halfway(video_path: str, fraction: float = 0.5) -> list:
    """Load frames up to `fraction` of the video, subsampled by FRAME_SUBSAMPLE."""
    reader = imageio.get_reader(video_path)
    meta = reader.get_meta_data()
    fps = meta.get("fps", 20)
    total_frames = reader.count_frames()
    cutoff = int(total_frames * fraction)

    frames = []
    for i, frame in enumerate(reader):
        if i >= cutoff:
            break
        #frames.append(Image.fromarray(frame))
        if i % FRAME_SUBSAMPLE == 0:
            frames.append(Image.fromarray(frame))
    reader.close()

    duration_s = total_frames / fps
    print(f"Video: {total_frames} frames @ {fps:.1f}fps = {duration_s:.1f}s total")
    print(f"Using first {fraction*100:.0f}% = {cutoff} frames → {len(frames)} PIL images after subsampling")
    return frames, duration_s


def run_qwen(model, processor, frame_buffer, task_description):

    user_text = (
        f"Robot task: {task_description}\n"
        "Analyze the video for failures (missed grasp, dropped object, or visibly stuck).\n"
        "After your reasoning, provide your final decision by exactly following this format:\n"
        "DECISION: CONTINUE\n"
        "or\n"
        "DECISION: RETRY"
    )

    messages = [
        {"role": "user", "content": [
            {
                "video": frame_buffer,
                "sample_fps": 5,
                "max_frames": 128,
                "total_pixels": 20480 * 28 * 28,
                "min_pixels": 16 * 28 * 28,
            },
            {"type": "text", "text": user_text},
        ]},
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
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

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        video_metadata=video_metadatas,
        **video_kwargs,
        do_resize=False,
        return_tensors="pt",
    ).to(model.device)

    print(f"Input token count: {inputs['input_ids'].shape[1]}")

    with torch.inference_mode():
        out_ids = model.generate(**inputs, max_new_tokens=4096, do_sample=False, temperature=1.0, top_p=0.95, top_k=20, min_p=0.0)

    new_tokens = out_ids[0, inputs["input_ids"].shape[1]:]
    raw = processor.decode(new_tokens, skip_special_tokens=True).strip()
    return raw


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--task_description", type=str,
                        default="pick the lemon from the cabinet and place it on the counter")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3.5-4B")
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--fraction", type=float, default=0.5,
                        help="Fraction of video to use (0.5 = first half)")
    args = parser.parse_args()

    print(f"\nLoading frames from: {args.video_path}")
    frames, duration_s = load_frames_to_halfway(args.video_path, args.fraction)

    print(f"\nLoading Qwen: {args.model_id} on {args.device}")
    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
        trust_remote_code=True,
    )
    model.eval()
    print("Model ready.\n")

    print(f"Task: {args.task_description}")

    response = run_qwen(model, processor, frames, args.task_description)

    print(f"\n{'='*50}")
    print(f"Raw Qwen output: '{response}'")
    is_retry = response.upper().startswith("B") or "RETRY" in response.upper()
    print(f"Decision: {'RETRY ✓' if is_retry else 'CONTINUE'}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()
