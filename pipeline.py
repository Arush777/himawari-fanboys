"""Core video -> multi-style caption pipeline."""
import json
import os
import re
import tempfile

import config
from llm_client import RitsClient
from video_utils import download_video, extract_frames, frame_to_data_uri

STYLE_GUIDE = {
    "formal": (
        "Professional, objective, factual tone, like a news-agency or stock-footage caption. "
        "One clear declarative sentence stating what the video shows. "
        "No slang, no humour, no exclamation marks, no opinions."
    ),
    "sarcastic": (
        "Dry, deadpan, ironic, lightly mocking. Use understatement, mock admiration, or "
        "faint praise aimed at something actually visible in the clip. "
        "One sentence. Subtle wit, not mean-spirited or absurd."
    ),
    "humorous_tech": (
        "Genuinely funny, built on a technology or programming joke (e.g. bugs, deploys, "
        "merge conflicts, CPUs, Wi-Fi, AI, loading screens) mapped onto what is literally "
        "happening in the clip. One to two sentences. The scene must still be recognisable "
        "from the caption."
    ),
    "humorous_non_tech": (
        "Genuinely funny, warm, relatable everyday humour that anyone would get. "
        "Absolutely NO technology, programming, internet, or science references. "
        "One to two sentences about everyday life, feelings, food, weather, work, etc., "
        "grounded in what the clip shows."
    ),
}

DESCRIBE_PROMPT = (
    "You are shown {n} frames sampled evenly, in chronological order, from a single video clip "
    "(roughly 30 seconds to 2 minutes long). Write a factual, neutral description of the clip: "
    "the setting and time of day, the main subject(s), what they are doing, how the action "
    "progresses across the frames, and any distinctive visual details (colours, weather, "
    "objects, visible text, camera angle or motion). 4-6 sentences. Describe only what is "
    "clearly visible in the frames; do not speculate or invent details."
)


def _style_prompt(description: str, styles: list[str]) -> str:
    style_lines = "\n".join(f'- "{s}": {STYLE_GUIDE[s]}' for s in styles)
    keys = ", ".join(f'"{s}"' for s in styles)
    return (
        "Here is a factual description of a video clip:\n\n"
        f"{description}\n\n"
        "Write one caption for this video in EACH of the following styles:\n"
        f"{style_lines}\n\n"
        "Every caption is judged separately on two things: (1) how accurately it reflects the "
        "actual video content, and (2) how well it matches its requested tone. So each caption "
        "- including the funny ones - must clearly reference the real subject, setting, and "
        "action of the clip, and must stand alone without the other captions. Write in "
        "English. Do not mention frames, images, descriptions, or that this is a video "
        "analysis.\n\n"
        f"Respond with ONLY a JSON object with exactly these keys: {keys}. "
        "No markdown, no extra text, just the raw JSON object."
    )


def _parse_json_object(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise


def _describe_video(video_url: str, vision_client: RitsClient) -> str:
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
            frame_uris, DESCRIBE_PROMPT.format(n=len(frame_uris)),
        )


def describe_video(video_url: str, vision_client: RitsClient) -> str:
    """Return just the factual scene description (no style rewriting)."""
    return _describe_video(video_url, vision_client)


def captions_from_description(description: str, styles: list[str],
                              text_client: RitsClient) -> dict:
    """Rewrite a factual description into one caption per requested style."""
    raw = text_client.generate_text(_style_prompt(description, styles))
    captions = _parse_json_object(raw)
    result = {s: str(captions.get(s, "")).strip() for s in styles}

    # A missing style scores zero for the whole clip — retry just the missing ones once.
    missing = [s for s in styles if not result[s]]
    if missing:
        raw = text_client.generate_text(_style_prompt(description, missing))
        retry = _parse_json_object(raw)
        for s in missing:
            result[s] = str(retry.get(s, "")).strip()

    return result


def caption_video(video_url: str, styles: list[str],
                   vision_client: RitsClient, text_client: RitsClient) -> dict:
    description = _describe_video(video_url, vision_client)
    return captions_from_description(description, styles, text_client)
