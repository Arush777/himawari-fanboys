"""LLM-as-a-judge: score a results.json's captions against the source videos.

Standalone eval tool (like describe_videos.py) — not part of the graded main.py contract.
For each task, re-downloads the clip and samples frames directly (the same ground truth
the captioning model saw), then asks the judge model to score each style's caption on:

  - accuracy   (1-5): does the caption reflect what's actually in the video?
  - tone_fit   (1-5): does it match the requested style's tone?

Scoring against the frames themselves (not the factual description used to generate the
captions) avoids just checking self-consistency with that description.

Usage:
    export INPUT_PATH="$(pwd)/sample_input/tasks.json"        # tasks: video_url + styles
    export RESULTS_PATH="$(pwd)/sample_output/results.json"   # captions to grade
    export JUDGE_OUTPUT_PATH="$(pwd)/sample_output/judged_results.json"
    python3 judge.py
"""
import json
import os
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

import config
from llm_client import FireworksClient
from pipeline import STYLE_GUIDE
from video_utils import (
    choose_num_frames,
    download_video,
    extract_frames,
    frame_to_b64,
    get_duration_seconds,
)

INPUT_PATH = os.environ.get("INPUT_PATH", "/input/tasks.json")
RESULTS_PATH = os.environ.get("RESULTS_PATH", "/output/results.json")
JUDGE_OUTPUT_PATH = os.environ.get("JUDGE_OUTPUT_PATH", "/output/judged_results.json")
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", "4"))

JUDGE_PROMPT = (
    "You are shown {n} frames sampled evenly, in chronological order, from a video clip. "
    "You are grading captions written for this clip by another AI. For EACH style below, "
    "score the caption on two 1-5 integer scales:\n\n"
    "- accuracy: 1 = describes something not in the video at all, 5 = clearly and "
    "specifically reflects the real subject, setting, and action visible in the frames.\n"
    "- tone_fit: 1 = does not match the requested tone at all, 5 = strongly and "
    "unmistakably matches the requested tone.\n\n"
    "An empty caption always scores 1 on both. For `notes`, cite specific visual details "
    "from THIS clip's frames that support or undercut the score — do not restate the style "
    "definition.\n\n"
    "Captions to grade:\n{captions_block}\n\n"
    "Style definitions:\n{style_block}\n\n"
    "Respond with only a single JSON object matching the requested schema, no other text."
)


def _judge_schema(styles: list[str]) -> dict:
    per_style = {
        "type": "object",
        "properties": {
            "accuracy": {"type": "integer", "minimum": 1, "maximum": 5},
            "tone_fit": {"type": "integer", "minimum": 1, "maximum": 5},
            "notes": {"type": "string"},
        },
        "required": ["accuracy", "tone_fit", "notes"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {s: per_style for s in styles},
        "required": list(styles),
        "additionalProperties": False,
    }


def _judge_prompt(captions: dict, styles: list[str], num_frames: int) -> str:
    captions_block = "\n".join(f'- "{s}": {captions.get(s, "") or "(empty)"}' for s in styles)
    style_block = "\n".join(f'- "{s}": {STYLE_GUIDE[s]}' for s in styles)
    return JUDGE_PROMPT.format(n=num_frames, captions_block=captions_block, style_block=style_block)


def judge_task(video_url: str, captions: dict, styles: list[str], client: FireworksClient) -> dict:
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

        prompt = _judge_prompt(captions, styles, len(frames_b64))
        return client.generate_json(prompt, _judge_schema(styles), frames_b64=frames_b64, max_tokens=3072)


def main() -> int:
    with open(INPUT_PATH, "r") as f:
        tasks = {t["task_id"]: t for t in json.load(f)}
    with open(RESULTS_PATH, "r") as f:
        results = json.load(f)

    client = FireworksClient(config.FIREWORKS_API_KEY, config.JUDGE_MODEL_ID, config.FIREWORKS_BASE_URL)

    def run(result: dict) -> dict:
        task_id = result["task_id"]
        task = tasks[task_id]
        captions = result["captions"]
        styles = task["styles"]
        try:
            scores = judge_task(task["video_url"], captions, styles, client)
        except Exception:
            import traceback
            print(f"[{task_id}] JUDGE FAILED: {traceback.format_exc()}", file=sys.stderr)
            scores = {s: {"accuracy": 1, "tone_fit": 1, "notes": "judge failed"} for s in styles}
        return {"task_id": task_id, "scores": scores}

    with ThreadPoolExecutor(max_workers=max(1, min(MAX_WORKERS, len(results)))) as pool:
        judged = list(pool.map(run, results))

    all_scores = [s for j in judged for s in j["scores"].values()]
    summary = {
        "num_tasks": len(judged),
        "avg_accuracy": round(sum(s["accuracy"] for s in all_scores) / len(all_scores), 2),
        "avg_tone_fit": round(sum(s["tone_fit"] for s in all_scores) / len(all_scores), 2),
    }

    os.makedirs(os.path.dirname(JUDGE_OUTPUT_PATH), exist_ok=True)
    with open(JUDGE_OUTPUT_PATH, "w") as f:
        json.dump({"summary": summary, "results": judged}, f, indent=2)

    print(f"Judged {summary['num_tasks']} tasks — "
          f"avg accuracy {summary['avg_accuracy']}/5, avg tone_fit {summary['avg_tone_fit']}/5",
          file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
