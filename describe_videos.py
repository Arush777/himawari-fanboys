"""Local-only helper: print/save a long, detailed factual description for each task's
video. Uses its own richer prompt (separate from pipeline.py's compact one used to
seed the 4 graded style captions). Not part of the graded pipeline."""
import json
import os
import sys
import tempfile

import config
from llm_client import FireworksClient
from video_utils import (
    choose_num_frames,
    download_video,
    extract_frames,
    frame_to_b64,
    get_duration_seconds,
)

INPUT_PATH = os.environ.get("INPUT_PATH", "/input/tasks.json")
OUTPUT_PATH = os.environ.get("DESCRIBE_OUTPUT_PATH", "descriptions.json")

DETAILED_DESCRIBE_PROMPT = (
    "You are shown {n} frames sampled evenly, in chronological order, from one video clip. "
    "Write a long, factual, neutral, highly detailed description of everything visible across "
    "the frames. Cover: the setting/location, all subjects (people, animals, objects) and their "
    "appearance, what actions or changes happen over the course of the clip, colors and lighting, "
    "camera framing/angle, and any distinctive or notable visual details. Aim for a thorough "
    "paragraph of at least 8-10 sentences. Do not speculate beyond what is visible in the frames."
)


def describe_video_detailed(video_url: str, client: FireworksClient) -> str:
    with tempfile.TemporaryDirectory() as tmp_dir:
        video_path = os.path.join(tmp_dir, "clip.mp4")
        download_video(video_url, video_path)

        duration = get_duration_seconds(video_path)
        num_frames = choose_num_frames(
            duration, config.SECONDS_PER_FRAME, config.MIN_FRAMES, config.MAX_FRAMES,
        )

        frames_dir = os.path.join(tmp_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)
        frame_paths = extract_frames(
            video_path, frames_dir,
            num_frames=num_frames, max_width=config.FRAME_MAX_WIDTH,
        )
        frames_b64 = [frame_to_b64(p) for p in frame_paths]

        return client.describe_frames(
            frames_b64, DETAILED_DESCRIBE_PROMPT.format(n=len(frames_b64)),
            max_tokens=2048,
        )


def main() -> int:
    with open(INPUT_PATH, "r") as f:
        tasks = json.load(f)

    client = FireworksClient(config.FIREWORKS_API_KEY, config.FIREWORKS_MODEL_ID, config.FIREWORKS_BASE_URL)

    results = []
    for task in tasks:
        task_id = task["task_id"]
        video_url = task["video_url"]
        print(f"[{task_id}] describing...", file=sys.stderr)
        description = describe_video_detailed(video_url, client)
        print(f"[{task_id}] {description}\n", file=sys.stderr)
        results.append({"task_id": task_id, "description": description})

    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    return 0


if __name__ == "__main__":
    sys.exit(main())
