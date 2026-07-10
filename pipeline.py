"""Core video -> multi-style caption pipeline.

Architecture (style-specialist + best-of-2 + facts-grounding + critique/repair, shipped
after beating the previous describe-then-style pipeline 58.3% in blind pairwise judging
on 30s-2min clips, then layering in two more stages):

  1. Extract evenly spaced frames, ask the VLM for a factual description PLUS a list of
     concrete, independently-checkable facts (one structured-output call).
  2. One SPECIALIST call per requested style (persona + tone exemplars + facts) writes
     two candidate captions taking different angles.
  3. One frame-grounded SELECTION call sees the actual frames plus all candidates and
     picks the best caption per style (accuracy to pixels + facts + unmistakable tone).
  4. One CRITIQUE call scores the selected captions 1-5 accuracy/tone_fit against the
     frames (same rubric judge.py uses); any caption scoring below threshold on either
     axis — or tripping the deterministic tech-word guard on humorous_non_tech — gets
     rewritten, once, in a batched REPAIR call fed the critique's notes. Each rewrite
     is re-critiqued and only kept if it scores at least as well as the original.

Every stage falls back to the previous path on failure, so a task can never end up with
empty captions because of the extra machinery, and critique/repair can never make a
result worse than skipping it (any exception there keeps the pre-critique captions).
"""
import json
import os
import re
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

DESCRIBE_FACTS_PROMPT = (
    "You are shown {n} frames sampled evenly, in chronological order, from a single video clip "
    "(roughly 30 seconds to 2 minutes long). Respond with a single JSON object with two fields:\n\n"
    "- \"description\": a factual, neutral description of the clip — the setting and time of "
    "day, the main subject(s), what they are doing, how the action progresses across the "
    "frames, and any distinctive visual details (colours, weather, objects, visible text, "
    "camera angle or motion). 4-6 sentences.\n"
    "- \"facts\": a list of 5-10 short, independently-checkable claims about what is literally "
    "visible in the frames (specific objects, colours, actions, setting details, on-screen "
    "text). Each fact should be concrete enough that someone looking only at the frames could "
    "verify it. Order the facts from most visually prominent and persistent (the main subject "
    "and its central action, visible across many frames) to least (background or single-frame "
    "details), and only state a colour of a background element if it is unambiguous.\n\n"
    "Describe only what is clearly visible in the frames; do not speculate or invent details. "
    "Do not identify a city, country, company, building, or sign text unless it is large, "
    "legible, and unambiguous in the sampled frames."
)

CANDIDATE_SCHEMA = {
    "type": "object",
    "properties": {"a": {"type": "string"}, "b": {"type": "string"}},
    "required": ["a", "b"],
    "additionalProperties": False,
}


def _facts_schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "description": {"type": "string"},
            "facts": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["description", "facts"],
        "additionalProperties": False,
    }


def _specialist_prompt(style: str, description: str, facts: list[str]) -> str:
    exemplars = CAPTION_EXEMPLARS.get(style, [])
    ex_block = ""
    if exemplars:
        ex_lines = "\n".join(f'- "{e}"' for e in exemplars)
        ex_block = (
            "\nExamples of the tone sharpness expected, from OTHER videos (a balloon "
            "festival, a blacksmith) - match their quality, never reuse their "
            f"subjects or jokes:\n{ex_lines}\n"
        )
    facts_block = "\n".join(f"- {f}" for f in facts) if facts else "(none extracted)"
    return (
        f"{PERSONAS[style]}\n\n"
        "Here is a factual description of a video clip:\n\n"
        f"{description}\n\n"
        "Concrete facts noticed in the frames:\n"
        f"{facts_block}\n\n"
        f'Write TWO different candidate captions for this video in the "{style}" '
        f"style: {STYLE_GUIDE[style]}\n"
        f"{ex_block}\n"
        "The two candidates must take clearly different angles (different detail "
        "focused on, or a different joke/framing). Each caption will be scored by a "
        "judge who watches the video: 1-5 for accuracy (every claim visibly true) and "
        "1-5 for tone fit (the style must be unmistakable, not mild). Write to earn "
        "5/5 on both. Anchor the caption on the MAIN subject and its central action — "
        "the first facts in the list — not on background or peripheral details: a "
        "strict judge marks down any claim about background colours, object counts, "
        "small text, or things visible only for a moment, so a joke built on a shaky "
        "peripheral detail loses points that a joke built on the main action keeps. "
        "Each candidate must be consistent with at least one fact from "
        "the list above and must not contradict the description, be 1-2 sentences, "
        "in English. Never mention frames, images, descriptions, or video analysis. "
        "Avoid named places, brands, or sign text unless the description says they "
        "are unmistakably legible. Respond with only a JSON object with keys \"a\" "
        "and \"b\"."
    )


def _selection_prompt(description: str, facts: list[str], candidates: dict) -> str:
    cand_block = json.dumps(candidates, indent=2)
    facts_block = "\n".join(f"- {f}" for f in facts) if facts else "(none extracted)"
    return (
        "Above are frames sampled evenly, in chronological order, from a video clip. "
        "Here is a factual description of the clip:\n\n"
        f"{description}\n\n"
        "Concrete facts noticed in the frames:\n"
        f"{facts_block}\n\n"
        "For each caption style below there are two candidate captions, \"a\" and "
        f"\"b\":\n\n{cand_block}\n\n"
        "The frames are ground truth. For EACH style, choose the candidate that (1) is "
        "more accurate - every claim visibly true in the frames and consistent with "
        "the facts above - and (2) has the more unmistakable, sharper execution of "
        "its style. Accuracy outranks flash: a funnier caption containing even one "
        "claim a strict judge could not verify from the frames (a background colour, "
        "a count, a fleeting detail) loses to a slightly plainer one whose every "
        "claim is checkable. Return the chosen caption TEXT exactly as written, one "
        "per style, in a JSON object keyed by style name. Do not rewrite, merge, or "
        "edit the captions; copy the winner verbatim."
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


CRITIQUE_PROMPT = (
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


def build_critique_schema(styles: list[str]) -> dict:
    # No minimum/maximum on the integers: Fireworks' json_schema mode doesn't fully
    # support them either way, and Anthropic's structured outputs rejects them
    # outright ("For 'integer' type, properties maximum, minimum are not supported").
    # The 1-5 range is enforced by the prompt wording instead.
    per_style = {
        "type": "object",
        "properties": {
            "accuracy": {"type": "integer"},
            "tone_fit": {"type": "integer"},
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


def build_critique_prompt(captions: dict, styles: list[str], num_frames: int) -> str:
    captions_block = "\n".join(f'- "{s}": {captions.get(s, "") or "(empty)"}' for s in styles)
    style_block = "\n".join(f'- "{s}": {STYLE_GUIDE[s]}' for s in styles)
    return CRITIQUE_PROMPT.format(n=num_frames, captions_block=captions_block, style_block=style_block)


# Deterministic guard for humorous_non_tech's hard negative constraint ("absolutely NO
# technology, programming, internet, or science references"). The LLM critique has passed
# captions that leaked tech vocabulary (a tone_fit 2 on the leaderboard run), so this
# lexical check forces such a caption into repair regardless of its critique score.
# High-precision terms only: everyday-ambiguous words (cloud, mouse, bug, web, server,
# charging, loading, devices, ram...) are deliberately excluded — a false trip would
# force a pointless rewrite of a good caption.
_TECH_WORDS_RE = re.compile(
    r"\b("
    r"wi-?fi|internet|online|website|apps?|software|hardware|computers?|laptops?|"
    r"phones?|smartphones?|iphone|android|screens?|monitor|keyboard|robots?|robotic|"
    r"ai|algorithms?|coding|programming|programmer|developer|debug|debugging|deploys?|"
    r"deployed|database|digital|electronic|electronics|tech|technology|browser|email|"
    r"texting|password|pixels?|buffering|downloads?|downloading|uploads?|uploading|"
    r"reboots?|rebooting|battery|batteries|bluetooth|gps|drones?|livestream|podcast|"
    r"selfie|video ?games?|videogames?|glitch|glitching|notifications?|spreadsheets?|"
    r"hashtag|autocorrect|screensaver|cpu|gpu|usb|hdmi|scientists?|scientific|physics|"
    r"chemistry|biology|laboratory|molecules?|atoms?|gravity|dna"
    r")\b",
    re.IGNORECASE,
)


def tech_guard_violations(caption: str) -> list[str]:
    """Tech/science words in a caption that humorous_non_tech is banned from using."""
    return sorted({m.group(0).lower() for m in _TECH_WORDS_RE.finditer(caption or "")})


def _apply_tech_guard(captions: dict, critique: dict) -> dict:
    """Override the critique for humorous_non_tech when the lexical guard trips, so the
    caption is forced into repair with an explicit instruction naming the leaked words."""
    style = "humorous_non_tech"
    if style not in captions:
        return critique
    leaks = tech_guard_violations(captions[style])
    if not leaks:
        return critique
    patched = dict(critique)
    entry = dict(patched.get(style, {}))
    entry["tone_fit"] = 1
    entry["notes"] = (
        f"{entry.get('notes', '')} The caption violates the style's absolute ban on "
        f"technology/science references (found: {', '.join(leaks)}); rewrite it with "
        f"zero such references."
    ).strip()
    patched[style] = entry
    return patched


def critique_captions(captions: dict, styles: list[str], frames_b64: list[str],
                       client: CaptionClient) -> dict:
    """Score each style's caption for accuracy/tone_fit against frames already in memory
    (no re-download — for an independently re-verified score, see judge.py instead)."""
    prompt = build_critique_prompt(captions, styles, len(frames_b64))
    return client.generate_json(
        prompt, build_critique_schema(styles), frames_b64=frames_b64, max_tokens=3072,
    )


def _repair_prompt(weak_styles: list[str], captions: dict, critique: dict,
                    description: str, facts: list[str]) -> str:
    facts_block = "\n".join(f"- {f}" for f in facts) if facts else "(none extracted)"
    items_block = "\n".join(
        f'- "{s}" (style: {STYLE_GUIDE[s]})\n'
        f'  current caption: "{captions.get(s, "")}"\n'
        f'  reviewer feedback: {critique.get(s, {}).get("notes", "")}'
        for s in weak_styles
    )
    return (
        "Here is a factual description of a video clip:\n\n"
        f"{description}\n\n"
        "Concrete facts noticed in the frames:\n"
        f"{facts_block}\n\n"
        "You are also shown the original frames from the clip.\n\n"
        "A reviewer scored the following captions too low on accuracy and/or tone match, and "
        "gave feedback on why. Rewrite EACH of these captions to fix the issue the reviewer "
        "raised, while staying grounded in the facts above and consistent with the attached "
        "frames. Keep each caption's requested style and length.\n\n"
        f"{items_block}\n\n"
        "Respond with only a single JSON object matching the requested schema (one rewritten "
        "caption per style key listed above), no other text."
    )


def _critique_sum(critique: dict, style: str) -> int:
    entry = critique.get(style, {})
    return int(entry.get("accuracy", 0)) + int(entry.get("tone_fit", 0))


def repair_weak_captions(captions: dict, critique: dict, description: str, facts: list[str],
                          frames_b64: list[str], client: CaptionClient,
                          threshold: int = config.CRITIQUE_THRESHOLD) -> dict:
    """Rewrite, once, only the captions the critique scored below `threshold` on either
    axis — feeding back the critique's notes as the repair instruction, not just the
    score. Verify-before-accept: each rewrite is re-critiqued against the frames and only
    kept if it scores at least as well as the original, so repair is monotonic under the
    critic (it can no longer swap a 4 for a 3). A failed or empty repair keeps the
    pre-repair caption rather than blanking it; a failed re-critique accepts the rewrite
    (the pre-verification behaviour)."""
    weak = [
        s for s in captions
        if critique.get(s, {}).get("accuracy", 5) < threshold
        or critique.get(s, {}).get("tone_fit", 5) < threshold
    ]
    if not weak:
        return captions

    repaired = client.generate_json(
        _repair_prompt(weak, captions, critique, description, facts), _caption_schema(weak),
        frames_b64=frames_b64,
    )
    rewrites = {
        s: str(repaired.get(s, "")).strip()
        for s in weak
        if str(repaired.get(s, "")).strip()
        and str(repaired.get(s, "")).strip() != captions[s]
    }
    if not rewrites:
        return captions

    result = dict(captions)
    try:
        recritique = _apply_tech_guard(
            rewrites, critique_captions(rewrites, list(rewrites), frames_b64, client),
        )
        for s, new_val in rewrites.items():
            if _critique_sum(recritique, s) >= _critique_sum(critique, s):
                result[s] = new_val
    except Exception:
        print(f"[pipeline] re-critique of repaired captions failed, accepting rewrites: "
              f"{traceback.format_exc()}", file=sys.stderr)
        result.update(rewrites)
    return result


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


def describe_with_facts(frames_b64: list[str], client: CaptionClient) -> dict:
    """One vision call: {"description": str, "facts": list[str]} grounded in frames_b64."""
    result = client.generate_json(
        DESCRIBE_FACTS_PROMPT.format(n=len(frames_b64)), _facts_schema(),
        frames_b64=frames_b64, max_tokens=1536,
    )
    return {
        "description": str(result.get("description", "")).strip(),
        "facts": [str(f).strip() for f in result.get("facts", []) if str(f).strip()],
    }


def _describe_video(video_url: str, client: CaptionClient) -> str:
    frames_b64 = _extract_frames_b64(video_url)
    return describe_with_facts(frames_b64, client)["description"]


def describe_video(video_url: str, client: CaptionClient) -> str:
    """Return just the factual scene description (no style rewriting, no facts)."""
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
    """Primary path: describe+facts -> per-style specialist candidates -> frame-grounded
    selection -> critique/repair, with the single-call path as safety net at every stage."""
    frames_b64 = _extract_frames_b64(video_url)
    grounding = describe_with_facts(frames_b64, client)
    description, facts = grounding["description"], grounding["facts"]

    candidates: dict[str, dict] = {}
    for s in styles:
        if s not in STYLE_GUIDE:
            continue  # unknown style: leave to the fallback single-call path
        try:
            candidates[s] = client.generate_json(
                _specialist_prompt(s, description, facts), CANDIDATE_SCHEMA, max_tokens=800,
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
                    _selection_prompt(description, facts, candidates), sel_schema,
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

    result = {s: result.get(s, "") for s in styles}

    # Final safety net: critique the selected captions against the frames and repair
    # anything that still scored low, once. A failure here degrades to the pre-critique
    # captions rather than risking the whole task.
    if config.ENABLE_CRITIQUE_REPAIR:
        try:
            critique = _apply_tech_guard(result, critique_captions(result, styles, frames_b64, client))
            result = repair_weak_captions(result, critique, description, facts, frames_b64, client)
        except Exception:
            print(f"[pipeline] critique/repair failed, keeping pre-repair captions: "
                  f"{traceback.format_exc()}", file=sys.stderr)

    return result
