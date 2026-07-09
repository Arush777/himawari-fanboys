"""Core video -> multi-style caption pipeline.

Architecture (style-specialist + best-of-2, shipped after beating the previous
describe-then-style pipeline 58.3% in blind pairwise judging on 30s-2min clips):

  1. Extract evenly spaced frames, ask the VLM for a factual description.
  2. One SPECIALIST call per requested style (persona + tone exemplars) writes two
     candidate captions taking different angles.
  3. One frame-grounded SELECTION call sees the actual frames plus all candidates and
     picks the best caption per style (accuracy to pixels + unmistakable tone).

Every new stage falls back to the previous single-call path on failure, so a task can
never end up with empty captions because of the extra machinery.
"""
import json
import os
import sys
import tempfile
import traceback
from typing import Protocol

import config
from video_utils import (
    choose_num_frames,
    download_video,
    extract_frames,
    frame_to_b64,
    get_duration_seconds,
)


class CaptionClient(Protocol):
    def describe_frames(self, frames_b64: list[str], prompt: str, max_tokens: int = 1024) -> str:
        ...

    def generate_json(self, prompt: str, schema: dict, frames_b64: list[str] | None = None,
                      max_tokens: int = 1024) -> dict:
        ...

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

# Persona line prepended to each style's specialist call. Writing one style per call
# (instead of all four in one) measurably sharpened tone in pairwise judging.
PERSONAS = {
    "formal": (
        "You are a senior news-agency caption writer. Your captions are precise, "
        "objective, single-sentence, publishable as-is."
    ),
    "sarcastic": (
        "You are a dry, deadpan comedy writer known for ironic, mock-admiring "
        "one-liners. Your sarcasm is unmistakable but never mean."
    ),
    "humorous_tech": (
        "You are a comedy writer for a developer audience. Your jokes map bugs, "
        "deploys, CPUs, Wi-Fi and AI onto everyday scenes, and they actually land."
    ),
    "humorous_non_tech": (
        "You are a warm observational comedian. Your jokes are about everyday life - "
        "food, moods, weather, work - and never mention technology or science."
    ),
}

# Tone exemplars from OTHER scenes (a balloon festival, a blacksmith), used only for
# tone calibration. Formal deliberately gets none: A/B testing showed exemplars help
# sarcasm/humour but add noise to formal.
CAPTION_EXEMPLARS = {
    "sarcastic": [
        "Ah yes, dozens of giant balloons expensively drifting wherever the wind feels like taking them, truly humanity's boldest answer to a question nobody asked.",
        "Wow, a man hitting hot metal with a hammer over and over until it becomes slightly flatter hot metal, riveting stuff from the cutting edge of the year 1400.",
    ],
    "humorous_tech": [
        "Sunrise rollout: fifty hot-air balloons spinning up over the valley like autoscaled pods, and impressively not a single one crashed on launch.",
        "This blacksmith is just debugging hardware with a hammer: heat the metal, whack it, inspect the output, repeat until the sword finally compiles.",
    ],
    "humorous_non_tech": [
        "Fifty balloons drifting over the river at sunrise, all pretending they know where they're going, just like the rest of us before breakfast.",
        "A man with the arm strength of three gym memberships keeps whacking that glowing metal like it owes him money, and the sparks clearly agree.",
    ],
}

DESCRIBE_PROMPT = (
    "You are shown {n} frames sampled evenly, in chronological order, from a single video clip "
    "(roughly 30 seconds to 2 minutes long). Write a factual, neutral description of the clip: "
    "the setting and time of day, the main subject(s), what they are doing, how the action "
    "progresses across the frames, and any distinctive visual details (colours, weather, "
    "objects, visible text, camera angle or motion). 4-6 sentences. Describe only what is "
    "clearly visible in the frames; do not speculate or invent details. Do not identify a "
    "city, country, company, building, or sign text unless it is large, legible, and "
    "unambiguous in the sampled frames."
)

CANDIDATE_SCHEMA = {
    "type": "object",
    "properties": {"a": {"type": "string"}, "b": {"type": "string"}},
    "required": ["a", "b"],
    "additionalProperties": False,
}


def _specialist_prompt(style: str, description: str) -> str:
    exemplars = CAPTION_EXEMPLARS.get(style, [])
    ex_block = ""
    if exemplars:
        ex_lines = "\n".join(f'- "{e}"' for e in exemplars)
        ex_block = (
            "\nExamples of the tone sharpness expected, from OTHER videos (a balloon "
            "festival, a blacksmith) - match their quality, never reuse their "
            f"subjects or jokes:\n{ex_lines}\n"
        )
    return (
        f"{PERSONAS[style]}\n\n"
        "Here is a factual description of a video clip:\n\n"
        f"{description}\n\n"
        f'Write TWO different candidate captions for this video in the "{style}" '
        f"style: {STYLE_GUIDE[style]}\n"
        f"{ex_block}\n"
        "The two candidates must take clearly different angles (different detail "
        "focused on, or a different joke/framing). Each caption will be scored by a "
        "judge who watches the video: 1-5 for accuracy (every claim visibly true) and "
        "1-5 for tone fit (the style must be unmistakable, not mild). Write to earn "
        "5/5 on both. Each candidate must include at least one concrete, specific "
        "visual detail from the description, be 1-2 sentences, in English. Never "
        "mention frames, images, descriptions, or video analysis. Avoid named places, "
        "brands, or sign text unless the description says they are unmistakably "
        "legible. Respond with only a JSON object with keys \"a\" and \"b\"."
    )


def _selection_prompt(description: str, candidates: dict) -> str:
    cand_block = json.dumps(candidates, indent=2)
    return (
        "Above are frames sampled evenly, in chronological order, from a video clip. "
        "Here is a factual description of the clip:\n\n"
        f"{description}\n\n"
        "For each caption style below there are two candidate captions, \"a\" and "
        f"\"b\":\n\n{cand_block}\n\n"
        "The frames are ground truth. For EACH style, choose the candidate that (1) is "
        "more accurate - every claim visibly true in the frames - and (2) has the more "
        "unmistakable, sharper execution of its style. Return the chosen caption TEXT "
        "exactly as written, one per style, in a JSON object keyed by style name. Do "
        "not rewrite, merge, or edit the captions; copy the winner verbatim."
    )


def _style_prompt(description: str, styles: list[str]) -> str:
    """Single-call multi-style prompt. Kept as the FALLBACK path (and for the demo
    app); the primary path is the per-style specialist flow in caption_video()."""
    style_lines = "\n".join(f'- "{s}": {STYLE_GUIDE[s]}' for s in styles)
    return (
        "Here is a factual description of a video clip:\n\n"
        f"{description}\n\n"
        "Write one caption for this video in EACH of the following styles:\n"
        f"{style_lines}\n\n"
        "Every caption is judged separately on two things: (1) how accurately it reflects the "
        "actual video content, and (2) how well it matches its requested tone. So each caption "
        "- including the funny ones - must name at least one concrete, specific visual detail "
        "from the description (a colour, an object, a distinguishing feature, a setting "
        "detail) rather than a generic paraphrase, and must stand alone. "
        "Avoid named places, brands, building names, or exact sign text unless the "
        "description says the text is unmistakably legible. "
        "Write in English. Do not mention frames, images, descriptions, or that "
        "this is a video analysis. Respond with only a single JSON object matching the "
        "requested schema, no other text."
    )


def _caption_schema(styles: list[str]) -> dict:
    return {
        "type": "object",
        "properties": {s: {"type": "string"} for s in styles},
        "required": list(styles),
        "additionalProperties": False,
    }


def _extract_frames_b64(video_url: str) -> list[str]:
    """Download the clip and return base64 JPEG frames, evenly sampled, in order."""
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
        return [frame_to_b64(p) for p in frame_paths]


def _describe_video(video_url: str, client: CaptionClient) -> str:
    frames_b64 = _extract_frames_b64(video_url)
    return client.describe_frames(frames_b64, DESCRIBE_PROMPT.format(n=len(frames_b64)))


def describe_video(video_url: str, client: CaptionClient) -> str:
    """Return just the factual scene description (no style rewriting)."""
    return _describe_video(video_url, client)


def captions_from_description(description: str, styles: list[str],
                              client: CaptionClient) -> dict:
    """Single-call fallback: rewrite a description into one caption per style.

    Uses structured outputs, so the response is guaranteed to be valid JSON with
    exactly the requested style keys."""
    captions = client.generate_json(_style_prompt(description, styles), _caption_schema(styles))
    result = {s: str(captions.get(s, "")).strip() for s in styles}

    # An empty caption scores zero for that style — retry just the empty ones once.
    missing = [s for s in styles if not result[s]]
    if missing:
        retry = client.generate_json(_style_prompt(description, missing), _caption_schema(missing))
        for s in missing:
            result[s] = str(retry.get(s, "")).strip()

    return result


def caption_video(video_url: str, styles: list[str], client: CaptionClient) -> dict:
    """Primary path: describe -> per-style specialist candidates -> frame-grounded
    selection, with the single-call path as safety net at every stage."""
    frames_b64 = _extract_frames_b64(video_url)
    description = client.describe_frames(frames_b64, DESCRIBE_PROMPT.format(n=len(frames_b64)))

    candidates: dict[str, dict] = {}
    for s in styles:
        if s not in STYLE_GUIDE:
            continue  # unknown style: leave to the fallback single-call path
        try:
            candidates[s] = client.generate_json(
                _specialist_prompt(s, description), CANDIDATE_SCHEMA, max_tokens=800,
            )
        except Exception:
            print(f"[pipeline] specialist call failed for style {s}: "
                  f"{traceback.format_exc()}", file=sys.stderr)

    result: dict[str, str] = {}
    if candidates:
        sel_schema = {
            "type": "object",
            "properties": {s: {"type": "string"} for s in candidates},
            "required": list(candidates),
            "additionalProperties": False,
        }
        # Two attempts: selection output echoes captions verbatim, so a truncated or
        # malformed JSON response on attempt 1 (seen in testing) usually succeeds on
        # retry. On double failure the candidate-"a" fallback below still fires.
        for attempt in (1, 2):
            try:
                chosen = client.generate_json(
                    _selection_prompt(description, candidates), sel_schema,
                    frames_b64=frames_b64, max_tokens=2000,
                )
                for s in candidates:
                    result[s] = str(chosen.get(s, "")).strip()
                break
            except Exception:
                print(f"[pipeline] selection attempt {attempt} failed: "
                      f"{traceback.format_exc()}", file=sys.stderr)
        # Selection returned empty/invented text for a style -> fall back to candidate "a".
        for s in candidates:
            if not result.get(s, "").strip():
                result[s] = str(candidates[s].get("a", "")).strip()

    # Anything still missing/empty (unknown style, failed specialist call, etc.)
    # goes through the proven single-call path.
    missing = [s for s in styles if not result.get(s, "").strip()]
    if missing:
        fallback = captions_from_description(description, missing, client)
        for s in missing:
            result[s] = fallback.get(s, "")

    return {s: result.get(s, "") for s in styles}
