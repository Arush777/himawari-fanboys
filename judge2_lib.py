"""Robust dev-only LLM-as-judge harness for caption evaluation."""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import statistics
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path("/Users/arush/AMD-hackathon")
REPO = ROOT / "video-captioning-agent"
ARTIFACT_DIR = ROOT / "codex-run-artifacts" / "judge2"
FRAME_CACHE_DIR = ROOT / "eval-cache" / "frames"
RESPONSE_CACHE_DIR = ROOT / "eval-cache" / "judge-responses"
PROMPT_VERSION = "judge2-v5"
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", "4"))
ANTHROPIC_LOCK = threading.Lock()

META_RE = re.compile(
    r"\b(frame(s)?|image(s)?|screenshot|sampled|described|description(s)?|"
    r"video analysis|the clip shows)\b",
    re.IGNORECASE,
)
FOOTAGE_RE = re.compile(r"\bfootage\b", re.IGNORECASE)
TECH_JOKE_RE = re.compile(
    r"\b(wifi|wi-fi|internet|app|software|hardware|ai|artificial intelligence|"
    r"algorithm|code|coding|program|deploy|server|router|buffering|loading|"
    r"download|upload|cpu|gpu|bug|debug|reboot|update|browser|password|"
    r"streaming|podcast|laptop battery|screensaver|autocorrect|emoji)\b|\\.exe",
    re.IGNORECASE,
)
SCENE_TECH_RE = re.compile(r"\b(computer|keyboard|screen|monitor|phone|mouse|desk)\b", re.I)
BRAND_PLACE_RE = re.compile(r"(?<![.!?]\s)\b[A-Z][a-zA-Z]{2,}\b")
COMMON_CAPS = {
    "A", "An", "The", "This", "That", "These", "Those", "It", "Its", "In", "On",
    "At", "By", "For", "With", "Without", "While", "Nothing", "Someone", "Wow",
}

ABSOLUTE_PROMPT = """You are a strict evaluation judge for a video captioning competition. Above are {n}
frames sampled evenly, in chronological order, from the video being captioned.

The caption below was submitted for the style "{style}", defined as:
{style_definition}

Caption: "{caption}"

Work through these steps IN ORDER and report them in the JSON:
1. claims: list every factual claim the caption makes about the video (subjects,
   colours, actions, objects, setting, weather, text, camera). Split compound claims.
2. claim_verdicts: for each claim, exactly one of "visible" (clearly supported by the
   frames), "not_visible" (contradicted or absent), "unverifiable" (cannot be
   confirmed either way, e.g. inner thoughts, sounds, causes, what happens off-screen).
3. specificity: one sentence - does the caption contain at least one detail that
   distinguishes THIS video from other videos of the same general kind, or could it
   fit many similar videos?
4. tone_evidence: quote the exact words/phrases that create the requested tone, or
   state "none" if you cannot find any.
5. tone_mechanism_present: for sarcastic - irony, understatement, or mock praise
   aimed at something visible; for humorous styles - an identifiable joke mechanism
   (incongruity, exaggeration, wordplay, comic comparison); for formal - objective
   declarative news-agency register. true/false.

Then score on a 0-10 scale using these anchors. Calibrate: you are ranking
COMPETITION FINALISTS - most captions you will see are decent; your job is to find
what separates the best from the merely good. A typical good caption earns 6-7.
Reserve 10 for a caption that could not be improved.

accuracy_0_10:
- 0-2: the caption describes a fundamentally different scene (wrong subject,
  setting, time of day, weather, or main action).
- 3-4: the caption is otherwise accurate but contains one contradicted attribute
  or one invented object.
- Any claim marked "not_visible" -> at most 4.
- Any claim marked "unverifiable" -> at most 6.
- Caption could fit many similar videos (no distinguishing detail) -> at most 6.
- All claims visible + at least one distinctive detail -> 7-9.
- 10 only if flawless, specific, and complete with zero doubtful words.
- Do not give accuracy 10 merely because a caption is very good. If it omits any
  major visible aspect of the video, uses broad wording where a sharper visible
  detail is available, or is a strong finalist caption that could still be improved,
  cap accuracy at 9.

tone_0_10:
- Score tone independently from factual accuracy. If a caption is written in the
  requested register but has a wrong visual detail, penalize accuracy for that wrong
  detail; do not also lower tone unless the wording itself fails the style.
- tone_evidence is "none" or the text reads as a different style -> at most 3.
- Tone present but mild/hedged (a judge could mistake it for neutral) -> at most 6.
- For sarcastic captions, generic hedges like "not the most exciting", "not that
  thrilling", "I guess", "I suppose", "kind of", or "honestly" are mild sarcasm
  unless paired with a sharp ironic comparison or mock praise aimed at a visible
  detail; mild sarcasm -> at most 6.
- tone_mechanism_present true and the tone is unmistakable -> 7-9.
- 10 only if the tone is unmistakable AND the execution is genuinely sharp (for
  humour: it would actually make someone smile; for formal: publishable as-is).
- Do not give tone 10 to a merely clean, competent, or standard caption; 9 is the
  ceiling unless the wording is exceptionally polished or genuinely memorable.
- For humorous_non_tech: if the HUMOUR relies on technology/internet/science concepts
  -> at most 4 (merely mentioning a visible object like a computer is acceptable).

Respond with only a JSON object matching the schema."""

COMPARE_PROMPT = """You are a strict evaluation judge for a video captioning competition. Above are {n}
frames sampled evenly, in chronological order, from the video being captioned.

Two captions were submitted for the style "{style}", defined as:
{style_definition}

Caption A: "{caption_a}"
Caption B: "{caption_b}"

Pick the caption that would score higher on BOTH of these, weighted equally:
(1) accuracy - every claim visibly true in the frames, with at least one detail
    specific to this exact video;
(2) tone fit - the style is unmistakable and sharply executed.
Prefer the caption that is more specific AND more unmistakably in-tone. Disqualify a
caption from winning if it states anything not visible in the frames. If they are
genuinely equal in quality, answer "tie" - do not invent a winner.
If one caption contains a contradicted visible attribute (for example the wrong
colour, clothing, subject, object, or setting) and the other caption does not, the
caption with the contradicted attribute must lose; do not answer tie in that case.

Respond with only a JSON object matching the schema."""


def sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


def write_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def get_judge_frames(video_url: str) -> list[str]:
    """Return cached base64 JPEG frames for judge evaluation."""
    from video_utils import download_video, extract_frames, frame_to_b64, get_duration_seconds

    FRAME_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = sha1_text(video_url)
    cache_dir = FRAME_CACHE_DIR / key
    meta_path = cache_dir / "meta.json"
    frame_paths = sorted(cache_dir.glob("frame_*.jpg"))
    if meta_path.exists() and frame_paths:
        print(f"frame_cache HIT {key}", file=sys.stderr)
        return [frame_to_b64(str(path)) for path in frame_paths]

    print(f"frame_cache MISS {key}", file=sys.stderr)
    cache_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp_dir:
        video_path = Path(tmp_dir) / "clip.mp4"
        download_video(video_url, str(video_path))
        duration = get_duration_seconds(str(video_path))
        num_frames = min(24, max(16, math.ceil(duration / 5)))
        paths = extract_frames(str(video_path), str(cache_dir), num_frames=num_frames, max_width=768)
    meta_path.write_text(json.dumps({
        "url": video_url,
        "duration": duration,
        "frame_count": len(paths),
    }, indent=2) + "\n")
    return [frame_to_b64(path) for path in sorted(paths)]


def _strip_schema_bounds(schema: Any) -> Any:
    if isinstance(schema, dict):
        return {k: _strip_schema_bounds(v) for k, v in schema.items() if k not in {"minimum", "maximum"}}
    if isinstance(schema, list):
        return [_strip_schema_bounds(item) for item in schema]
    return schema


@dataclass(frozen=True)
class JudgeModel:
    name: str
    provider: str
    model: str

    def _client(self):
        import config
        from llm_client import ClaudeJudgeClient, FireworksClient

        if self.provider == "fireworks":
            return FireworksClient(config.FIREWORKS_API_KEY, self.model, config.FIREWORKS_BASE_URL, timeout=120)
        if self.provider == "anthropic":
            key = _fresh_env_value("ANTHROPIC_API_KEY") or config.ANTHROPIC_API_KEY
            return ClaudeJudgeClient(key, self.model, timeout=120)
        raise ValueError(f"unknown judge provider: {self.provider}")

    def generate_json(self, prompt: str, schema: dict, frames_b64: list[str]) -> dict:
        client = self._client()
        last_exc: Exception | None = None
        schemas = [_strip_schema_bounds(schema)] if self.provider == "anthropic" else [schema, _strip_schema_bounds(schema)]
        for attempt in range(3):
            for current_schema in schemas:
                try:
                    if self.provider == "anthropic":
                        with ANTHROPIC_LOCK:
                            return client.generate_json(
                                prompt, current_schema, frames_b64=frames_b64,
                                max_tokens=2000, temperature=0,
                            )
                    return client.generate_json(
                        prompt, current_schema, frames_b64=frames_b64,
                        max_tokens=2000, temperature=0,
                    )
                except Exception as exc:
                    last_exc = exc
                    if self.provider == "anthropic" and not _is_rate_limit(exc):
                        raise
            time.sleep(1 + attempt)
        assert last_exc is not None
        raise last_exc


def _fresh_env_value(key: str) -> str:
    env_path = REPO / ".env"
    if not env_path.exists():
        return ""
    for raw in env_path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        if name.strip() == key:
            return value.strip().strip('"').strip("'")
    return ""


def _is_rate_limit(exc: Exception) -> bool:
    text = str(exc).lower()
    return "429" in text or "rate" in text or "rate_limit" in text


def load_judges(path: str | Path = REPO / "judge2_models.json") -> list[JudgeModel]:
    data = read_json(path)
    return [JudgeModel(**judge) for judge in data["judges"]]


def _style_guide() -> dict:
    from pipeline import STYLE_GUIDE

    return STYLE_GUIDE


def response_cache_path(key_data: dict) -> Path:
    RESPONSE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = sha1_text(json.dumps(key_data, sort_keys=True))
    return RESPONSE_CACHE_DIR / f"{key}.json"


def response_cache_key(
    judge: JudgeModel,
    mode: str,
    video_url: str,
    style: str,
    caption_text: str,
    caption_b: str = "",
) -> dict:
    return {
        "prompt_version": PROMPT_VERSION,
        "judge_model": judge.model,
        "judge_name": judge.name,
        "mode": mode,
        "video_url": video_url,
        "style": style,
        "caption_text": caption_text,
        "caption_b": caption_b,
    }


def cached_generate_json(
    judge: JudgeModel,
    mode: str,
    video_url: str,
    style: str,
    caption_text: str,
    prompt: str,
    schema: dict,
    frames_b64: list[str],
    caption_b: str = "",
    no_cache: bool = False,
) -> dict:
    key_data = response_cache_key(judge, mode, video_url, style, caption_text, caption_b)
    path = response_cache_path(key_data)
    if not no_cache and path.exists():
        return read_json(path)["response"]
    response = judge.generate_json(prompt, schema, frames_b64)
    if not no_cache:
        write_json(path, {"key": key_data, "response": response})
    return response


def _word_count(caption: str) -> int:
    return len(re.findall(r"\b[\w'.-]+\b", caption))


def _meta_text(caption: str) -> str:
    text = re.sub(r"\bframes per second\b", "", caption, flags=re.I)
    text = re.sub(r"\bfps\b", "", text, flags=re.I)
    return text


def layer0_check(style: str, caption: str, duplicate: bool = False) -> dict:
    hard_flags: list[str] = []
    soft_flags: list[str] = []
    stripped = caption.strip()
    words = _word_count(stripped)
    if not stripped or words < 5 or words > 70:
        hard_flags.append("empty_or_degenerate")
    if duplicate:
        hard_flags.append("duplicate")
    meta_text = _meta_text(stripped)
    if META_RE.search(meta_text):
        hard_flags.append("meta_leak")
    if style != "formal" and re.search(r"\bthis video shows\b", meta_text, re.I):
        hard_flags.append("meta_leak")
    if FOOTAGE_RE.search(stripped):
        soft_flags.append("meta_footage")
    if style == "humorous_non_tech" and TECH_JOKE_RE.search(stripped):
        hard_flags.append("tech_joke_leak")
    if style == "humorous_non_tech" and SCENE_TECH_RE.search(stripped):
        soft_flags.append("scene_tech_word")
    if style == "formal" and "!" in stripped:
        soft_flags.append("exclamation_in_formal")
    for token in BRAND_PLACE_RE.findall(stripped):
        if token not in COMMON_CAPS:
            soft_flags.append("possible_brand_or_place")
            break
    return {"hard_flags": sorted(set(hard_flags)), "soft_flags": sorted(set(soft_flags))}


def absolute_schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "claims": {"type": "array", "items": {"type": "string"}},
            "claim_verdicts": {
                "type": "array",
                "items": {"type": "string", "enum": ["visible", "not_visible", "unverifiable"]},
            },
            "specificity": {"type": "string"},
            "tone_evidence": {"type": "string"},
            "tone_mechanism_present": {"type": "boolean"},
            "accuracy_0_10": {"type": "integer", "minimum": 0, "maximum": 10},
            "tone_0_10": {"type": "integer", "minimum": 0, "maximum": 10},
        },
        "required": [
            "claims", "claim_verdicts", "specificity", "tone_evidence",
            "tone_mechanism_present", "accuracy_0_10", "tone_0_10",
        ],
        "additionalProperties": False,
    }


def compare_schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "winner": {"type": "string", "enum": ["A", "B", "tie"]},
            "reason": {"type": "string"},
        },
        "required": ["winner", "reason"],
        "additionalProperties": False,
    }


def absolute_prompt(style: str, caption: str, n_frames: int) -> str:
    return ABSOLUTE_PROMPT.format(
        n=n_frames,
        style=style,
        style_definition=_style_guide()[style],
        caption=caption.replace('"', '\\"'),
    )


def compare_prompt(style: str, caption_a: str, caption_b: str, n_frames: int) -> str:
    return COMPARE_PROMPT.format(
        n=n_frames,
        style=style,
        style_definition=_style_guide()[style],
        caption_a=caption_a.replace('"', '\\"'),
        caption_b=caption_b.replace('"', '\\"'),
    )


def median(values: list[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def _clamp_int(value: Any, lo: int = 0, hi: int = 10) -> int:
    try:
        return max(lo, min(hi, int(value)))
    except Exception:
        return lo


def _task_maps(tasks_path: str | Path, results_path: str | Path) -> tuple[dict, list[dict]]:
    tasks = {task["task_id"]: task for task in read_json(tasks_path)}
    results = read_json(results_path)
    return tasks, results


def caption_items_from_results(tasks_path: str | Path, results_path: str | Path) -> list[dict]:
    tasks, results = _task_maps(tasks_path, results_path)
    items = []
    for result in results:
        task_id = result["task_id"]
        task = tasks[task_id]
        captions = result.get("captions", {})
        folded_counts: dict[str, int] = {}
        for style in task["styles"]:
            text = str(captions.get(style, "")).strip().casefold()
            folded_counts[text] = folded_counts.get(text, 0) + 1
        for style in task["styles"]:
            caption = str(captions.get(style, "")).strip()
            items.append({
                "item_id": f"{task_id}:{style}",
                "task_id": task_id,
                "video_url": task["video_url"],
                "style": style,
                "caption": caption,
                "duplicate": folded_counts.get(caption.casefold(), 0) > 1,
            })
    return items


def score_caption_items(items: list[dict], judges: list[JudgeModel],
                        no_cache: bool = False) -> list[dict]:
    frames_by_url = {url: get_judge_frames(url) for url in sorted({item["video_url"] for item in items})}
    jobs = []
    for item in items:
        layer0 = layer0_check(item["style"], item["caption"], duplicate=item.get("duplicate", False))
        for judge in judges:
            jobs.append((item, layer0, judge))

    def run(job: tuple[dict, dict, JudgeModel]) -> tuple[str, str, dict]:
        item, layer0, judge = job
        frames = frames_by_url[item["video_url"]]
        prompt = absolute_prompt(item["style"], item["caption"], len(frames))
        try:
            response = cached_generate_json(
                judge, "absolute", item["video_url"], item["style"], item["caption"],
                prompt, absolute_schema(), frames, no_cache=no_cache,
            )
        except Exception as exc:
            response = {
                "claims": [],
                "claim_verdicts": [],
                "specificity": "judge call failed",
                "tone_evidence": "none",
                "tone_mechanism_present": False,
                "accuracy_0_10": 0,
                "tone_0_10": 0,
                "error": exc.__class__.__name__,
            }
        return item["item_id"], judge.name, response

    with ThreadPoolExecutor(max_workers=max(1, min(MAX_WORKERS, len(jobs) or 1))) as pool:
        raw = list(pool.map(run, jobs))

    by_item: dict[str, dict] = {}
    for item in items:
        layer0 = layer0_check(item["style"], item["caption"], duplicate=item.get("duplicate", False))
        by_item[item["item_id"]] = {
            **item,
            "layer0": layer0,
            "judge_responses": {},
        }
    for item_id, judge_name, response in raw:
        by_item[item_id]["judge_responses"][judge_name] = response

    scored = []
    for item in items:
        row = by_item[item["item_id"]]
        accuracies = [_clamp_int(resp.get("accuracy_0_10")) for resp in row["judge_responses"].values()]
        tones = [_clamp_int(resp.get("tone_0_10")) for resp in row["judge_responses"].values()]
        combineds = [(a + t) / 2 for a, t in zip(accuracies, tones)]
        combined_raw = median(combineds)
        combined = min(combined_raw, 4.0) if row["layer0"]["hard_flags"] else combined_raw
        row["ensemble"] = {
            "accuracy_0_10": median(accuracies),
            "tone_0_10": median(tones),
            "combined_0_10_raw": combined_raw,
            "combined_0_10": combined,
        }
        scored.append(row)
    return scored


def summarize_scored(scored: list[dict]) -> dict:
    if not scored:
        return {}
    per_style = {}
    for style in sorted({row["style"] for row in scored}):
        rows = [row for row in scored if row["style"] == style]
        per_style[style] = {
            "combined_0_10": round(sum(row["ensemble"]["combined_0_10"] for row in rows) / len(rows), 3),
            "normalized": round(sum(row["ensemble"]["combined_0_10"] for row in rows) / len(rows) / 10, 4),
        }
    avg_acc = sum(row["ensemble"]["accuracy_0_10"] for row in scored) / len(scored)
    avg_tone = sum(row["ensemble"]["tone_0_10"] for row in scored) / len(scored)
    avg_combined = sum(row["ensemble"]["combined_0_10"] for row in scored) / len(scored)
    return {
        "num_captions": len(scored),
        "avg_accuracy_0_10": round(avg_acc, 3),
        "avg_tone_0_10": round(avg_tone, 3),
        "avg_combined_0_10": round(avg_combined, 3),
        "score_normalized": round(avg_combined / 10, 4),
        "hard_flag_count": sum(1 for row in scored if row["layer0"]["hard_flags"]),
        "per_style": per_style,
    }


def score_results(tasks_path: str | Path, results_path: str | Path, out_path: str | Path,
                  judges: list[JudgeModel], no_cache: bool = False) -> dict:
    items = caption_items_from_results(tasks_path, results_path)
    scored = score_caption_items(items, judges, no_cache=no_cache)
    grouped: dict[str, dict] = {}
    for row in scored:
        task = grouped.setdefault(row["task_id"], {"task_id": row["task_id"], "captions": {}})
        task["captions"][row["style"]] = {
            "caption": row["caption"],
            "layer0": row["layer0"],
            "judge_responses": row["judge_responses"],
            "ensemble": row["ensemble"],
        }
    output = {
        "prompt_version": PROMPT_VERSION,
        "judges": [judge.__dict__ for judge in judges],
        "summary": summarize_scored(scored),
        "results": [grouped[key] for key in sorted(grouped)],
    }
    write_json(out_path, output)
    return output


def compare_pairs(pairs: list[dict], judges: list[JudgeModel], no_cache: bool = False) -> list[dict]:
    frames_by_url = {url: get_judge_frames(url) for url in sorted({pair["video_url"] for pair in pairs})}
    jobs = []
    for pair in pairs:
        for judge in judges:
            for order in ("ab", "ba"):
                jobs.append((pair, judge, order))

    def run(job: tuple[dict, JudgeModel, str]) -> tuple[str, str, str, dict]:
        pair, judge, order = job
        frames = frames_by_url[pair["video_url"]]
        if order == "ab":
            cap_a, cap_b = pair["caption_a"], pair["caption_b"]
        else:
            cap_a, cap_b = pair["caption_b"], pair["caption_a"]
        prompt = compare_prompt(pair["style"], cap_a, cap_b, len(frames))
        try:
            response = cached_generate_json(
                judge, f"compare_{order}", pair["video_url"], pair["style"], cap_a,
                prompt, compare_schema(), frames, caption_b=cap_b, no_cache=no_cache,
            )
            if response.get("winner") not in {"A", "B", "tie"}:
                response["winner"] = "tie"
        except Exception as exc:
            response = {"winner": "tie", "reason": f"judge call failed: {exc.__class__.__name__}"}
        return pair["pair_id"], judge.name, order, response

    with ThreadPoolExecutor(max_workers=max(1, min(MAX_WORKERS, len(jobs) or 1))) as pool:
        raw = list(pool.map(run, jobs))

    grouped = {pair["pair_id"]: {**pair, "judge_verdicts": {}} for pair in pairs}
    for pair_id, judge_name, order, response in raw:
        verdict = grouped[pair_id]["judge_verdicts"].setdefault(judge_name, {})
        verdict[order] = response

    for pair in pairs:
        row = grouped[pair["pair_id"]]
        votes = []
        for judge_name, verdict in row["judge_verdicts"].items():
            ab = verdict.get("ab", {"winner": "tie"})
            ba = verdict.get("ba", {"winner": "tie"})
            ab_orig = ab["winner"]
            ba_orig = {"A": "B", "B": "A", "tie": "tie"}.get(ba["winner"], "tie")
            final = ab_orig if ab_orig == ba_orig else "tie"
            verdict["final"] = final
            votes.append(final)
        row["winner"] = _majority_vote(votes)
    return [grouped[pair["pair_id"]] for pair in pairs]


def _majority_vote(votes: list[str]) -> str:
    for label in ("A", "B", "tie"):
        if votes.count(label) > len(votes) / 2:
            return label
    return "tie"


def pair_items_from_results(tasks_path: str | Path, a_path: str | Path, b_path: str | Path) -> list[dict]:
    tasks = {task["task_id"]: task for task in read_json(tasks_path)}
    a_results = {row["task_id"]: row for row in read_json(a_path)}
    b_results = {row["task_id"]: row for row in read_json(b_path)}
    pairs = []
    for task_id in sorted(a_results):
        task = tasks[task_id]
        for style in task["styles"]:
            pairs.append({
                "pair_id": f"{task_id}:{style}",
                "task_id": task_id,
                "video_url": task["video_url"],
                "style": style,
                "caption_a": str(a_results[task_id]["captions"].get(style, "")).strip(),
                "caption_b": str(b_results[task_id]["captions"].get(style, "")).strip(),
            })
    return pairs


def summarize_compare(rows: list[dict]) -> dict:
    total = len(rows)
    a_wins = sum(1 for row in rows if row["winner"] == "A")
    b_wins = sum(1 for row in rows if row["winner"] == "B")
    ties = sum(1 for row in rows if row["winner"] == "tie")
    def rate(part: list[dict]) -> float:
        if not part:
            return 0.0
        return round((sum(1 for row in part if row["winner"] == "A") +
                      0.5 * sum(1 for row in part if row["winner"] == "tie")) / len(part), 4)
    return {
        "comparisons": total,
        "a_wins": a_wins,
        "b_wins": b_wins,
        "ties": ties,
        "a_win_rate": rate(rows),
        "per_style_a_win_rate": {
            style: rate([row for row in rows if row["style"] == style])
            for style in sorted({row["style"] for row in rows})
        },
        "per_task_a_win_rate": {
            task_id: rate([row for row in rows if row["task_id"] == task_id])
            for task_id in sorted({row["task_id"] for row in rows})
        },
    }


def compare_results(tasks_path: str | Path, a_path: str | Path, b_path: str | Path,
                    out_path: str | Path, judges: list[JudgeModel],
                    no_cache: bool = False) -> dict:
    pairs = pair_items_from_results(tasks_path, a_path, b_path)
    rows = compare_pairs(pairs, judges, no_cache=no_cache)
    output = {
        "prompt_version": PROMPT_VERSION,
        "judges": [judge.__dict__ for judge in judges],
        "summary": summarize_compare(rows),
        "results": rows,
    }
    write_json(out_path, output)
    return output


def validate_score_output(data: dict, expected_judges: int) -> bool:
    for task in data.get("results", []):
        for payload in task.get("captions", {}).values():
            if len(payload.get("judge_responses", {})) != expected_judges:
                return False
            if "ensemble" not in payload or "layer0" not in payload:
                return False
    return True


def run_selftest() -> list[str]:
    cases = [
        ("formal", "In every frame the kitten walks through leaves.", True, "meta_leak"),
        ("humorous_non_tech", "She types at her keyboard while staring at the monitor.", False, "tech_joke_leak"),
        ("humorous_non_tech", "She types at her keyboard while staring at the monitor.", True, "scene_tech_word"),
        ("formal", "Wow! Cars move down a road in daylight.", True, "exclamation_in_formal"),
        ("humorous_tech", "The cars render at less than 60 frames per second today.", False, "meta_leak"),
        ("formal", "Tiny cat.", True, "empty_or_degenerate"),
        ("formal", "A kitten walks across a leaf-strewn outdoor path toward the camera.", False, "empty_or_degenerate"),
        ("sarcastic", "This video shows traffic moving under yellow trees.", True, "meta_leak"),
        ("formal", "This video shows traffic moving under yellow trees.", False, "meta_leak"),
        ("humorous_non_tech", "That kitten is buffering through the leaves like slow Wi-Fi.", True, "tech_joke_leak"),
        ("formal", "A woman works at a desk beside the Acme monitor.", True, "possible_brand_or_place"),
        ("formal", "A woman works at a desk beside a large monitor.", False, "possible_brand_or_place"),
        ("formal", "A calm office worker types beside a monitor.", True, "duplicate"),
        ("humorous_tech", "The office deploy is framed by a very serious monitor.", False, "meta_leak"),
        ("formal", " ".join(["word"] * 71), True, "empty_or_degenerate"),
        ("sarcastic", "This is the most riveting office footage ever.", False, "meta_leak"),
        ("sarcastic", "This is the most riveting office footage ever.", True, "meta_footage"),
    ]
    failures = []
    for index, (style, caption, should_have, flag) in enumerate(cases, 1):
        duplicate = flag == "duplicate"
        result = layer0_check(style, caption, duplicate=duplicate)
        flags = set(result["hard_flags"] + result["soft_flags"])
        has = flag in flags
        if has != should_have:
            failures.append(f"case {index}: expected {flag} presence {should_have}, got {result}")
    return failures
