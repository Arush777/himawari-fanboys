"""Local-only helper: print/save a long, detailed factual description for each task's
video. Uses its own richer prompt (separate from pipeline.py's compact one used to
seed the 4 graded style captions). Not part of the graded pipeline."""
import json
import os
import sys
import tempfile

import config
from llm_client import RitsClient
from video_utils import download_video, extract_frames, frame_to_data_uri

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


def describe_video_detailed(video_url: str, vision_client: RitsClient) -> str:
    with tempfile.TemporaryDirectory() as tmp_dir:
        video_path = os.path.join(tmp_dir, "clip.mp4")
        download_video(video_url, video_path)

        frames_dir = os.path.join(tmp_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)
        frame_paths = extract_frames(
            video_path, frames_dir,
            num_frames=config.NUM_FRAMES, max_width=config.FRAME_MAX_WIDTH,
        )
        frame_uris = [frame_to_data_uri(p) for p in frame_paths]

        return vision_client.describe_frames(
            frame_uris, DETAILED_DESCRIBE_PROMPT.format(n=len(frame_uris)),
            max_tokens=900,
        )


def main() -> int:
    with open(INPUT_PATH, "r") as f:
        tasks = json.load(f)

    vision_client = RitsClient(config.VISION_API_KEY, config.VISION_API_ENDPOINT, config.VISION_MODEL_ID)

    results = []
    for task in tasks:
        task_id = task["task_id"]
        video_url = task["video_url"]
        print(f"[{task_id}] describing...", file=sys.stderr)
        description = describe_video_detailed(video_url, vision_client)
        print(f"[{task_id}] {description}\n", file=sys.stderr)
        results.append({"task_id": task_id, "description": description})

    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    return 0


if __name__ == "__main__":
    sys.exit(main())
