"""Faithful runtime reproduction of the post-hoc style portfolio.

The 56.25% post-hoc arm selected one globally fixed source pipeline per style:

* formal             -> Sonnet description, Kimi audit, Kimi formal writer;
* sarcastic          -> shipped Sonnet specialist pipeline;
* humorous_tech      -> Gemini temporal notes injected into Sonnet's describe prompt,
                         then the shipped Sonnet specialist pipeline;
* humorous_non_tech  -> Patch-v1's exact prompts, selector, and regex backstop.

This module reproduces those four complete source arms. It deliberately does not share
descriptions or selectors across arms: the source artifacts came from independent runs,
and sharing those stages was the material error in the earlier runtime approximation.
Only deterministic preprocessing is shared (one video download and one extraction of
the same evenly sampled frames). All four arms generate all four style candidates so
their selectors retain the cross-style context present in the source runs; the router
then keeps the single globally assigned style from each arm.

This is an untracked local experiment. It does not modify the graded pipeline or image.
"""

from __future__ import annotations

import json
import os
import re
import sys
import tempfile
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

from google import genai

import config
import exp_gemini_notes
import pipeline
from exp_kimi_hybrid import KimiClient
from llm_client import ClaudeClient
from video_utils import (
    choose_num_frames,
    download_video,
    extract_frames,
    frame_to_b64,
    get_duration_seconds,
)


DRY_RUN = os.environ.get("DRY_RUN", "0") == "1"
ALL_STYLES = tuple(pipeline.STYLE_GUIDE)

# Exact Patch-v1 deterministic medium-reference backstop. The source was recovered
# from Claude file history for the run that produced patched_gcs12.json.
PATCH_MEDIUM_RE = re.compile(
    r"\b(camera|footage|filmed|filming|cinematog\w*|time-?lapse|montage|editing|"
    r"render(?:ing|ed)?|screenshot|(?<!per )frames?(?! per second)|lens|"
    r"video|clip|shot)\b",
    re.IGNORECASE,
)

PATCH_PERSONAS = dict(pipeline.PERSONAS)
PATCH_PERSONAS["humorous_tech"] = (
    "You are a comedy writer for a developer audience. Your jokes map bugs, "
    "deploys, CPUs, Wi-Fi and AI onto everyday scenes, and they actually land. "
    "The tech metaphor is always about the things and actions IN the scene (the "
    "dog, the waves, the typing hands) - never about the camera or the video."
)


def patch_specialist_prompt(style: str, description: str) -> str:
    """Return Patch-v1's specialist prompt byte-for-byte."""
    exemplars = pipeline.CAPTION_EXEMPLARS.get(style, [])
    ex_block = ""
    if exemplars:
        ex_lines = "\n".join(f'- "{example}"' for example in exemplars)
        ex_block = (
            "\nExamples of the tone sharpness expected, from OTHER videos (a balloon "
            "festival, a blacksmith) - match their quality, never reuse their "
            f"subjects or jokes:\n{ex_lines}\n"
        )
    return (
        f"{PATCH_PERSONAS[style]}\n\n"
        "Here is a factual description of a video clip:\n\n"
        f"{description}\n\n"
        f'Write TWO different candidate captions for this video in the "{style}" '
        f"style: {pipeline.STYLE_GUIDE[style]}\n"
        f"{ex_block}\n"
        "The two candidates must take clearly different angles (different detail "
        "focused on, or a different joke/framing). Each caption will be scored by a "
        "judge who watches the video: 1-5 for accuracy (every claim visibly true) and "
        "1-5 for tone fit (the style must be unmistakable, not mild). Write to earn "
        "5/5 on both. Each candidate must include at least one concrete, specific "
        "visual detail from the description, be 1-2 sentences, in English. "
        "HARD RULES: (1) Never mention or joke about the camera, filming, footage, "
        "frames, shots, rendering, cinematography, editing, time-lapse, or the "
        "video/clip itself - every observation and every joke must be about what is "
        "IN the scene, not about how it was captured. (2) Do not narrate events, "
        "intentions, or outcomes that are not visibly happening (no 'waiting for', "
        "'about to', 'then leaves' unless the description says it visibly occurs). "
        "Avoid named places, brands, or sign text unless the description says they "
        "are unmistakably legible. Respond with only a JSON object with keys \"a\" "
        "and \"b\"."
    )


def patch_selection_prompt(description: str, candidates: dict) -> str:
    """Return Patch-v1's frame-grounded selection prompt byte-for-byte."""
    candidate_block = json.dumps(candidates, indent=2)
    return (
        "Above are frames sampled evenly, in chronological order, from a video clip. "
        "Here is a factual description of the clip:\n\n"
        f"{description}\n\n"
        "For each caption style below there are two candidate captions, \"a\" and "
        f"\"b\":\n\n{candidate_block}\n\n"
        "The frames are ground truth. For EACH style, choose the candidate that (1) is "
        "more accurate - every claim visibly true in the frames - and (2) has the more "
        "unmistakable, sharper execution of its style. Except in the formal style, "
        "treat any candidate that mentions or jokes about the camera, filming, "
        "footage, frames, rendering, or the video itself as defective - prefer the "
        "other candidate. Also prefer candidates that do not narrate events that are "
        "not visible. Return the chosen caption TEXT exactly as written, one per "
        "style, in a JSON object keyed by style name. Do not rewrite, merge, or edit "
        "the captions; copy the winner verbatim."
    )


def extract_shared_inputs(video_url: str, work_dir: str) -> tuple[str, list[str]]:
    """Download once and reproduce the champion's frame sampling exactly."""
    video_path = os.path.join(work_dir, "clip.mp4")
    download_video(video_url, video_path)
    duration = get_duration_seconds(video_path)
    frame_count = choose_num_frames(
        duration,
        config.SECONDS_PER_FRAME,
        config.MIN_FRAMES,
        config.MAX_FRAMES,
    )
    frames_dir = os.path.join(work_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    frame_paths = extract_frames(
        video_path,
        frames_dir,
        num_frames=frame_count,
        max_width=config.FRAME_MAX_WIDTH,
    )
    return video_path, [frame_to_b64(path) for path in frame_paths]


def describe_frames(
    frames_b64: list[str], claude: ClaudeClient, extra_prompt: str = ""
) -> str:
    prompt = pipeline.DESCRIBE_PROMPT.format(n=len(frames_b64)) + extra_prompt
    return claude.describe_frames(frames_b64, prompt)


def specialist_arm(
    frames_b64: list[str],
    description: str,
    claude: ClaudeClient,
    specialist_prompt: Callable[[str, str], str] = pipeline._specialist_prompt,
    selection_prompt: Callable[[str, dict], str] = pipeline._selection_prompt,
) -> dict[str, str]:
    """Run the source arms' full four-style specialist and selector flow."""
    candidates: dict[str, dict] = {}
    for style in ALL_STYLES:
        try:
            candidates[style] = claude.generate_json(
                specialist_prompt(style, description),
                pipeline.CANDIDATE_SCHEMA,
                max_tokens=800,
            )
        except Exception:
            print(
                f"[specialist {style} failed] {traceback.format_exc(limit=1)}",
                file=sys.stderr,
            )

    result: dict[str, str] = {}
    if candidates:
        selection_schema = {
            "type": "object",
            "properties": {style: {"type": "string"} for style in candidates},
            "required": list(candidates),
            "additionalProperties": False,
        }
        for attempt in (1, 2):
            try:
                chosen = claude.generate_json(
                    selection_prompt(description, candidates),
                    selection_schema,
                    frames_b64=frames_b64,
                    max_tokens=2000,
                )
                for style in candidates:
                    result[style] = str(chosen.get(style, "")).strip()
                break
            except Exception:
                print(
                    f"[selection attempt {attempt} failed] "
                    f"{traceback.format_exc(limit=1)}",
                    file=sys.stderr,
                )
        for style, pair in candidates.items():
            if not result.get(style, "").strip():
                result[style] = str(pair.get("a", "")).strip()

    missing = [style for style in ALL_STYLES if not result.get(style, "").strip()]
    if missing:
        fallback = pipeline.captions_from_description(description, missing, claude)
        for style in missing:
            result[style] = fallback.get(style, "")
    return {style: result.get(style, "") for style in ALL_STYLES}


def formal_kimi_route(
    frames_b64: list[str], claude: ClaudeClient, kimi: KimiClient
) -> dict[str, str]:
    """Reproduce the formal source arm, including its original fallbacks."""
    description = describe_frames(frames_b64, claude)
    try:
        verified = kimi.audit_description(frames_b64, description)
    except Exception:
        print(
            f"[formal Kimi audit failed] {traceback.format_exc(limit=1)}",
            file=sys.stderr,
        )
        verified = description
    try:
        caption = kimi.write_caption(verified, "formal", {})
    except Exception:
        print(
            f"[formal Kimi writer failed] {traceback.format_exc(limit=1)}",
            file=sys.stderr,
        )
        caption = pipeline.captions_from_description(
            verified, ["formal"], claude
        ).get("formal", "")
    return {"formal": caption}


def champion_route(frames_b64: list[str], claude: ClaudeClient) -> dict[str, str]:
    description = describe_frames(frames_b64, claude)
    return specialist_arm(frames_b64, description, claude)


def temporal_route(
    video_path: str,
    frames_b64: list[str],
    claude: ClaudeClient,
    gemini: genai.Client,
) -> dict[str, str]:
    """Reproduce exp_gemini_notes: notes affect description generation itself."""
    extra_prompt = ""
    try:
        notes = exp_gemini_notes.gemini_temporal_notes(video_path, gemini)
        extra_prompt = exp_gemini_notes.NOTES_BLOCK.format(notes=notes)
    except Exception:
        print(
            f"[Gemini temporal notes failed] {traceback.format_exc(limit=1)}",
            file=sys.stderr,
        )
    description = describe_frames(frames_b64, claude, extra_prompt)
    return specialist_arm(frames_b64, description, claude)


def patch_v1_route(frames_b64: list[str], claude: ClaudeClient) -> dict[str, str]:
    """Reproduce Patch-v1, including backstop checks on every comedy style."""
    description = describe_frames(frames_b64, claude)
    result = specialist_arm(
        frames_b64,
        description,
        claude,
        specialist_prompt=patch_specialist_prompt,
        selection_prompt=patch_selection_prompt,
    )
    for style in ALL_STYLES:
        if style == "formal" or not PATCH_MEDIUM_RE.search(result.get(style, "")):
            continue
        try:
            redo = claude.generate_json(
                patch_specialist_prompt(style, description)
                + "\nYour previous attempt mentioned the camera/footage/video itself,"
                " which is forbidden. Joke ONLY about what is in the scene.",
                pipeline.CANDIDATE_SCHEMA,
                max_tokens=800,
            )
            for candidate in (
                str(redo.get("a", "")).strip(),
                str(redo.get("b", "")).strip(),
            ):
                if candidate and not PATCH_MEDIUM_RE.search(candidate):
                    result[style] = candidate
                    break
        except Exception:
            print(
                f"[Patch-v1 backstop failed for {style}] "
                f"{traceback.format_exc(limit=1)}",
                file=sys.stderr,
            )
    return result


def caption_video_exact(
    task: dict,
    claude: ClaudeClient,
    kimi: KimiClient,
    gemini: genai.Client,
) -> dict[str, str]:
    requested_styles = task["styles"]
    if DRY_RUN:
        return {
            style: f"Dry-run caption for {style}; no model call was made."
            for style in requested_styles
        }

    with tempfile.TemporaryDirectory() as work_dir:
        video_path, frames_b64 = extract_shared_inputs(task["video_url"], work_dir)
        route_functions = {
            "formal": lambda: formal_kimi_route(frames_b64, claude, kimi),
            "champion": lambda: champion_route(frames_b64, claude),
            "temporal": lambda: temporal_route(
                video_path, frames_b64, claude, gemini
            ),
            "patch_v1": lambda: patch_v1_route(frames_b64, claude),
        }
        route_outputs: dict[str, dict[str, str]] = {}
        route_errors: dict[str, str] = {}
        # The source arms are independent. Parallel execution preserves their prompts
        # and outputs while keeping the critical path inside the 10-minute contract.
        with ThreadPoolExecutor(max_workers=4) as routes:
            future_to_name = {
                routes.submit(function): name
                for name, function in route_functions.items()
            }
            for future in as_completed(future_to_name):
                name = future_to_name[future]
                try:
                    route_outputs[name] = future.result()
                except Exception:
                    route_errors[name] = traceback.format_exc()
                    print(f"[{name} route failed] {route_errors[name]}", file=sys.stderr)

    champion = route_outputs.get("champion", {})
    source_for_style = {
        "formal": ("formal", "formal"),
        "sarcastic": ("champion", "sarcastic"),
        "humorous_tech": ("temporal", "humorous_tech"),
        "humorous_non_tech": ("patch_v1", "humorous_non_tech"),
    }
    captions: dict[str, str] = {}
    for style in requested_styles:
        route_name, source_style = source_for_style.get(
            style, ("champion", style)
        )
        caption = route_outputs.get(route_name, {}).get(source_style, "").strip()
        # A failed specialist branch falls back to the independently generated
        # champion caption rather than risking an empty submission.
        captions[style] = caption or champion.get(style, "").strip()

    empty = [style for style, caption in captions.items() if not caption]
    if empty:
        raise RuntimeError(
            f"empty captions after route and champion fallbacks: {empty}; "
            f"route errors={list(route_errors)}"
        )
    print(
        f"[{task['task_id']}] routes={','.join(sorted(route_outputs))} "
        f"frames={len(frames_b64)}",
        file=sys.stderr,
    )
    return captions


def main() -> int:
    started = time.monotonic()
    input_path = Path(os.environ["INPUT_PATH"])
    output_path = Path(os.environ["OUTPUT_PATH"])
    tasks = json.loads(input_path.read_text())

    if DRY_RUN:
        claude = kimi = gemini = None
    else:
        claude = ClaudeClient(config.ANTHROPIC_API_KEY, config.CLAUDE_MODEL_ID)
        kimi = KimiClient(config.FIREWORKS_API_KEY)
        gemini_key = os.environ["GEMINI_API_KEY"].strip("'\"")
        gemini = genai.Client(api_key=gemini_key)

    def run_task(task: dict) -> dict:
        try:
            captions = caption_video_exact(task, claude, kimi, gemini)
        except Exception:
            print(
                f"[{task['task_id']}] FAILED: {traceback.format_exc()}",
                file=sys.stderr,
            )
            captions = {style: "" for style in task["styles"]}
        return {"task_id": task["task_id"], "captions": captions}

    workers = int(os.environ.get("MAX_WORKERS", "2"))
    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(tasks)))) as pool:
        results = list(pool.map(run_task, tasks))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2) + "\n")
    print(f"wrote {output_path}")
    print(f"elapsed_seconds={time.monotonic() - started:.1f}")
    if kimi is not None:
        print(kimi.usage_summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
