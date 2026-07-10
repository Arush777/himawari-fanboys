"""Entry point: read /input/tasks.json, caption each video, write /output/results.json."""
import json
import os
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor

import config
from llm_client import ClaudeClient
from pipeline import caption_video

INPUT_PATH = os.environ.get("INPUT_PATH", "/input/tasks.json")
OUTPUT_PATH = os.environ.get("OUTPUT_PATH", "/output/results.json")
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", "4"))


def main() -> int:
    with open(INPUT_PATH, "r") as f:
        tasks = json.load(f)

    client = ClaudeClient(config.ANTHROPIC_API_KEY, config.CLAUDE_MODEL_ID)

    def run_task(task: dict) -> dict:
        task_id = task["task_id"]
        styles = task["styles"]
        try:
            captions = caption_video(task["video_url"], styles, client)
        except Exception:
            print(f"[{task_id}] FAILED: {traceback.format_exc()}", file=sys.stderr)
            captions = {s: "" for s in styles}
        return {"task_id": task_id, "captions": captions}

    # Clips are independent; run them in parallel to stay inside the 10-minute cap.
    with ThreadPoolExecutor(max_workers=max(1, min(MAX_WORKERS, len(tasks)))) as pool:
        results = list(pool.map(run_task, tasks))

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    return 0


if __name__ == "__main__":
    sys.exit(main())
