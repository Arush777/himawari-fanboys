"""Sonnet-facts + Kimi-voice hybrid experiment arm.

This arm combines two independently supported signals:

1. The live 0.87 champion's Sonnet frame description is retained because replacing
   it wholesale with Gemini native-video description lost the 12-clip pairwise A/B.
2. The public 0.91 ProVision image actually runs Kimi K2.6 for a description audit
   and sequential style writing, despite its submission page advertising Gemini.

Pipeline per clip:
  - extract the champion's 8-20 evenly sampled frames;
  - Sonnet writes the champion factual description;
  - Kimi audits that description against the same frames, conservatively;
  - Kimi writes one caption per style, sequentially, seeing earlier captions only
    to avoid repetitive sentence structures and joke angles.

There is deliberately no per-clip routing or hardcoded eval knowledge. If Kimi fails,
the arm falls back to the Sonnet description and then the champion single-call style
writer, so the experiment cannot emit empty captions because of the new machinery.

Usage:
  INPUT_PATH=... OUTPUT_PATH=... python exp_kimi_hybrid.py
  DRY_RUN=1 INPUT_PATH=... OUTPUT_PATH=... python exp_kimi_hybrid.py
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

from openai import OpenAI

import config
import pipeline
from llm_client import ClaudeClient


KIMI_MODEL = os.environ.get(
    "KIMI_MODEL_ID", "accounts/fireworks/models/kimi-k2p6"
)
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", "3"))
DRY_RUN = os.environ.get("DRY_RUN", "").strip().lower() in {
    "1", "true", "yes", "y", "on",
}


STYLE_BRIEFS = {
    "formal": (
        "Write a polished news-agency caption: objective, precise, and declarative. "
        "State the main subject, setting, and visible action without humor, opinion, "
        "dramatic language, or speculation."
    ),
    "sarcastic": (
        "Write an unmistakably dry, deadpan caption. Attach one short ironic or "
        "mock-admiring punchline to a specific visible action; keep it light, natural, "
        "and never invent a motive, dialogue, or unseen event."
    ),
    "humorous_tech": (
        "Write a genuinely funny developer-facing caption. Map one concrete visible "
        "detail onto a coherent software, debugging, deployment, latency, cache, API, "
        "or runtime joke while keeping the actual scene recognizable."
    ),
    "humorous_non_tech": (
        "Write warm observational comedy for a general audience, with no technology "
        "or science language. Build one relatable everyday comparison from a concrete "
        "visible detail, without inventing dialogue, motives, or additional actions."
    ),
}

CAPTION_SCHEMA = {
    "type": "object",
    "properties": {"caption": {"type": "string"}},
    "required": ["caption"],
    "additionalProperties": False,
}


class KimiClient:
    """Small Fireworks/Kimi client with explicit temperature and JSON support."""

    def __init__(self, api_key: str, timeout: float = 180.0) -> None:
        self.client = OpenAI(
            api_key=api_key,
            base_url=config.FIREWORKS_BASE_URL,
            timeout=timeout,
            max_retries=3,
        )
        self._usage_lock = Lock()
        self._input_tokens = 0
        self._output_tokens = 0

    @staticmethod
    def _text(response) -> str:
        text = response.choices[0].message.content or ""
        return text.strip()

    def _record_usage(self, response) -> None:
        usage = getattr(response, "usage", None)
        if usage is None:
            return
        input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        with self._usage_lock:
            self._input_tokens += input_tokens
            self._output_tokens += output_tokens

    def usage_summary(self) -> str:
        with self._usage_lock:
            input_tokens = self._input_tokens
            output_tokens = self._output_tokens
        estimated_cost = (input_tokens * 0.95 + output_tokens * 4.0) / 1_000_000
        return (
            f"Kimi usage: {input_tokens} input tokens, {output_tokens} output "
            f"tokens, estimated Fireworks cost ${estimated_cost:.4f}"
        )

    def audit_description(self, frames_b64: list[str], draft: str) -> str:
        content = [
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{frame}"},
            }
            for frame in frames_b64
        ]
        content.append(
            {
                "type": "text",
                "text": (
                    "Audit the draft video description below against these images, "
                    "which are evenly sampled in chronological order. Preserve every "
                    "specific detail that is visibly supported. Correct contradictions "
                    "and remove or generalize unsupported identities, locations, text, "
                    "colors, counts, motives, and events. Do not infer an event between "
                    "samples. Only describe motion or chronology when the ordered images "
                    "support it. Return 2-4 dense factual sentences, with no mention of "
                    "images, samples, uncertainty, auditing, or models.\n\n"
                    f"DRAFT:\n{draft}"
                ),
            }
        )
        response = self.client.chat.completions.create(
            model=KIMI_MODEL,
            messages=[{"role": "user", "content": content}],
            max_tokens=320,
            temperature=0.1,
            reasoning_effort="none",
        )
        self._record_usage(response)
        audited = self._text(response)
        if not audited:
            raise RuntimeError("Kimi returned an empty audited description")
        return audited

    def describe_scene(self, frames_b64: list[str]) -> str:
        content = [
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{frame}"},
            }
            for frame in frames_b64
        ]
        content.append(
            {
                "type": "text",
                "text": (
                    "These images are evenly sampled in chronological order from one "
                    "short video. Describe the visible setting, main subjects, concrete "
                    "objects, and action sequence in 2-4 dense factual sentences. "
                    "Mention changes over time only when the ordered images support them. "
                    "Use generic wording for uncertain identity, location, color, or text. "
                    "Do not mention images, frames, analysis, models, or uncertainty."
                ),
            }
        )
        response = self.client.chat.completions.create(
            model=KIMI_MODEL,
            messages=[{"role": "user", "content": content}],
            max_tokens=260,
            temperature=0.2,
            reasoning_effort="none",
        )
        self._record_usage(response)
        description = self._text(response)
        if not description:
            raise RuntimeError("Kimi returned an empty scene description")
        return description

    def write_caption(
        self,
        description: str,
        style: str,
        prior_captions: dict[str, str],
    ) -> str:
        prior_block = ""
        if prior_captions:
            prior_block = (
                "\n\nEarlier captions for this same clip are shown only to prevent "
                "repetitive wording. Use a different sentence structure and, for humor, "
                "a different comedic angle. Do not contradict them or refer to them:\n"
                + "\n".join(
                    f"- {name}: {caption}"
                    for name, caption in prior_captions.items()
                )
            )

        prompt = (
            f"{STYLE_BRIEFS[style]}\n\n"
            f"Verified factual description of the clip:\n{description}\n\n"
            "Write exactly one self-contained English caption of roughly 25-50 words. "
            "Include at least one concrete scene-specific fact. Every literal claim must "
            "be supported by the description; figurative humor may exaggerate importance "
            "but must not imply new objects, speech, motives, or events. Never mention "
            "frames, images, prompts, analysis, models, detection, pipelines, or "
            "uncertainty. Return only the requested JSON object."
            f"{prior_block}"
        )
        response = self.client.chat.completions.create(
            model=KIMI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=180,
            temperature=0.2 if style == "formal" else 0.75,
            reasoning_effort="none",
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "Caption", "schema": CAPTION_SCHEMA},
            },
        )
        self._record_usage(response)
        parsed = json.loads(self._text(response))
        caption = str(parsed.get("caption", "")).strip().strip('"')
        if not caption:
            raise RuntimeError(f"Kimi returned an empty {style} caption")
        return caption


def caption_video_hybrid(
    video_url: str,
    styles: list[str],
    claude: ClaudeClient,
    kimi: KimiClient,
) -> dict[str, str]:
    if DRY_RUN:
        return {
            style: f"Dry-run caption for {style}; no model call was made."
            for style in styles
        }

    frames_b64 = pipeline._extract_frames_b64(video_url)
    description = claude.describe_frames(
        frames_b64,
        pipeline.DESCRIBE_PROMPT.format(n=len(frames_b64)),
    )

    try:
        verified_description = kimi.audit_description(frames_b64, description)
        print("[kimi audit=ok]", file=sys.stderr)
    except Exception:
        print(
            f"[kimi audit failed; using Sonnet description] {traceback.format_exc()}",
            file=sys.stderr,
        )
        verified_description = description

    result: dict[str, str] = {}
    for style in styles:
        if style not in STYLE_BRIEFS:
            continue
        try:
            result[style] = kimi.write_caption(
                verified_description, style, result
            )
        except Exception:
            print(
                f"[kimi {style} failed; will use Sonnet fallback] "
                f"{traceback.format_exc()}",
                file=sys.stderr,
            )

    missing = [style for style in styles if not result.get(style, "").strip()]
    if missing:
        fallback = pipeline.captions_from_description(
            verified_description, missing, claude
        )
        for style in missing:
            result[style] = fallback.get(style, "")

    return {style: result.get(style, "") for style in styles}


def main() -> int:
    with open(os.environ["INPUT_PATH"], encoding="utf-8") as handle:
        tasks = json.load(handle)

    if not tasks:
        raise ValueError("INPUT_PATH contains no tasks")

    claude = ClaudeClient(config.ANTHROPIC_API_KEY, config.CLAUDE_MODEL_ID)
    kimi = KimiClient(config.FIREWORKS_API_KEY)

    def run_task(task: dict) -> dict:
        try:
            captions = caption_video_hybrid(
                task["video_url"], task["styles"], claude, kimi
            )
        except Exception:
            print(
                f"[{task['task_id']}] FAILED: {traceback.format_exc()}",
                file=sys.stderr,
            )
            captions = {style: "" for style in task["styles"]}
        return {"task_id": task["task_id"], "captions": captions}

    workers = max(1, min(MAX_WORKERS, len(tasks)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(run_task, tasks))

    output_path = os.environ["OUTPUT_PATH"]
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    print(f"wrote {output_path}")
    print(kimi.usage_summary())
    return 0


if __name__ == "__main__":
    sys.exit(main())
