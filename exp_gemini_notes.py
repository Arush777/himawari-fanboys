"""Temporal-notes hybrid arm.

Exactly the shipped specialist pipeline with ONE change: before Claude writes the
factual description from sampled frames, Gemini 3.1 Pro watches the FULL VIDEO and
emits terse "temporal notes" (events in chronological order, arrivals, state changes,
camera motion — explicitly NO colours/appearance, where native-video describing
hallucinated in the row-6 A/B). The notes are appended to the champion DESCRIBE_PROMPT
as auxiliary input; Claude keeps its own voice and remains the sole author. Styling and
selection are byte-identical to the champion, so any A/B delta is attributable to the
notes. If the Gemini call fails, the arm degrades to exactly the champion.

Usage: INPUT_PATH=... OUTPUT_PATH=... python exp_gemini_notes.py
"""
import json
import os
import sys
import tempfile
import traceback
from concurrent.futures import ThreadPoolExecutor

from google import genai
from google.genai import types as gtypes

import config
import pipeline
from llm_client import ClaudeClient
from video_utils import download_video

GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.1-pro-preview")
INLINE_LIMIT = 18 * 1024 * 1024

TEMPORAL_NOTES_PROMPT = (
    "Watch this video clip. List, in strict chronological order, only the EVENTS and "
    "CHANGES that happen over time: subject movement and direction, things that start "
    "or stop, arrivals or departures (state explicitly if something arrives but does "
    "NOT depart, or vice versa), state changes (e.g. a light changing), and camera "
    "motion. 3-6 short bullet points, each under 15 words. Do NOT describe colours, "
    "appearance, clothing, species, brands, or the setting - events and motion only. "
    "Only include events that actually occur within the clip; never speculate about "
    "what happens before or after it. If essentially nothing changes, reply with the "
    "single bullet: '- static scene, no notable changes'."
)

NOTES_BLOCK = (
    "\n\nAdditionally, here are temporal notes from watching the full continuous video "
    "(the frames above are only samples, so they can miss or misorder events):\n"
    "{notes}\n"
    "Use these notes ONLY to get the sequence of events right - what happens, in what "
    "order, and what does NOT happen. For appearance, colours, setting and every other "
    "visual detail, trust the frames. Write the description in your own words; do not "
    "copy the notes' wording. If a note contradicts what the frames clearly show, "
    "trust the frames."
)


def gemini_temporal_notes(video_path: str, client: genai.Client) -> str:
    size = os.path.getsize(video_path)
    if size <= INLINE_LIMIT:
        with open(video_path, "rb") as f:
            video_part = gtypes.Part.from_bytes(data=f.read(), mime_type="video/mp4")
        contents = [video_part, TEMPORAL_NOTES_PROMPT]
    else:
        uploaded = client.files.upload(file=video_path)
        import time
        while uploaded.state and uploaded.state.name == "PROCESSING":
            time.sleep(3)
            uploaded = client.files.get(name=uploaded.name)
        if uploaded.state and uploaded.state.name != "ACTIVE":
            raise RuntimeError(f"file upload state: {uploaded.state.name}")
        contents = [uploaded, TEMPORAL_NOTES_PROMPT]
    resp = client.models.generate_content(
        model=GEMINI_MODEL, contents=contents,
        config=gtypes.GenerateContentConfig(temperature=0.2, max_output_tokens=500),
    )
    text = (resp.text or "").strip()
    if not text:
        raise RuntimeError("empty gemini notes")
    return text


def caption_video_notes(video_url: str, styles: list[str],
                        claude: ClaudeClient, gem: genai.Client) -> dict:
    with tempfile.TemporaryDirectory() as tmp_dir:
        video_path = os.path.join(tmp_dir, "clip.mp4")
        download_video(video_url, video_path)
        try:
            notes = gemini_temporal_notes(video_path, gem)
        except Exception:
            print(f"[gemini notes failed, plain describe] {traceback.format_exc()}",
                  file=sys.stderr)
            notes = None

    frames_b64 = pipeline._extract_frames_b64(video_url)
    describe_prompt = pipeline.DESCRIBE_PROMPT.format(n=len(frames_b64))
    if notes:
        describe_prompt += NOTES_BLOCK.format(notes=notes)
    description = claude.describe_frames(frames_b64, describe_prompt)
    print(f"[notes={'gemini' if notes else 'none'}]", file=sys.stderr)

    candidates = {}
    for s in styles:
        if s not in pipeline.STYLE_GUIDE:
            continue
        try:
            candidates[s] = claude.generate_json(
                pipeline._specialist_prompt(s, description),
                pipeline.CANDIDATE_SCHEMA, max_tokens=800)
        except Exception:
            print(f"[specialist {s} failed]", file=sys.stderr)

    result = {}
    if candidates:
        sel_schema = {"type": "object",
                      "properties": {s: {"type": "string"} for s in candidates},
                      "required": list(candidates), "additionalProperties": False}
        for attempt in (1, 2):
            try:
                chosen = claude.generate_json(
                    pipeline._selection_prompt(description, candidates), sel_schema,
                    frames_b64=frames_b64, max_tokens=2000)
                for s in candidates:
                    result[s] = str(chosen.get(s, "")).strip()
                break
            except Exception:
                print(f"[selection attempt {attempt} failed]", file=sys.stderr)
        for s in candidates:
            if not result.get(s, "").strip():
                result[s] = str(candidates[s].get("a", "")).strip()

    missing = [s for s in styles if not result.get(s, "").strip()]
    if missing:
        fb = pipeline.captions_from_description(description, missing, claude)
        for s in missing:
            result[s] = fb.get(s, "")
    return {s: result.get(s, "") for s in styles}


def main() -> int:
    with open(os.environ["INPUT_PATH"]) as f:
        tasks = json.load(f)
    claude = ClaudeClient(config.ANTHROPIC_API_KEY, config.CLAUDE_MODEL_ID)
    gem = genai.Client(api_key=os.environ["GEMINI_API_KEY"].strip("'\""))

    def run_task(task):
        try:
            captions = caption_video_notes(task["video_url"], task["styles"], claude, gem)
        except Exception:
            print(f"[{task['task_id']}] FAILED: {traceback.format_exc()}", file=sys.stderr)
            captions = {s: "" for s in task["styles"]}
        return {"task_id": task["task_id"], "captions": captions}

    workers = int(os.environ.get("MAX_WORKERS", "3"))
    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(tasks)))) as pool:
        results = list(pool.map(run_task, tasks))

    out = os.environ["OUTPUT_PATH"]
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
