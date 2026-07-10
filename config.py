"""Configuration for the video captioning pipeline.

Values are read from the environment first (via .env locally, Streamlit Secrets on
Streamlit Cloud, or -e at `docker run` time). Never commit a real API key: .env is
gitignored, and the Docker image is pushed to a PUBLIC registry.
"""
import os

from dotenv import load_dotenv

load_dotenv()  # loads .env into os.environ if present; no-op if the file doesn't exist

FIREWORKS_API_KEY = os.environ.get("FIREWORKS_API_KEY", "")
# Fireworks is kept for judge/dev comparisons; the submitted generator uses Claude.
FIREWORKS_MODEL_ID = os.environ.get("FIREWORKS_MODEL_ID", "accounts/fireworks/models/qwen3p7-plus")
FIREWORKS_BASE_URL = "https://api.fireworks.ai/inference/v1"

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
# Primary submission model. CLAUDE_MODEL_ID is the explicit generator setting; the
# default is Sonnet: the Opus trial scored 0.80 on the hidden leaderboard vs
# Sonnet's 0.83, so Sonnet is the reverted-to champion.
CLAUDE_MODEL_ID = (
    os.environ.get("CLAUDE_MODEL_ID")
    or os.environ.get("ANTHROPIC_MODEL_ID")
    or "claude-sonnet-5"
)

# judge.py is a standalone dev tool (not part of the graded pipeline) that scores
# generated captions. JUDGE_PROVIDER picks its backend: "fireworks" (default, reuses
# FIREWORKS_API_KEY) or "anthropic" (uses ANTHROPIC_API_KEY). JUDGE_MODEL_ID is that
# backend's model.
JUDGE_PROVIDER = os.environ.get("JUDGE_PROVIDER", "fireworks")
JUDGE_MODEL_ID = os.environ.get(
    "JUDGE_MODEL_ID",
    "claude-sonnet-5" if JUDGE_PROVIDER == "anthropic" else FIREWORKS_MODEL_ID,
)

# Frame sampling: ~1 frame per SECONDS_PER_FRAME of video, clamped to
# [MIN_FRAMES, MAX_FRAMES]. Clips in the hidden set are 30s-2min, so this yields
# 8 frames for a 30s clip up to 20 frames for a 100s+ clip.
SECONDS_PER_FRAME = 5.0
MIN_FRAMES = 8
MAX_FRAMES = 20
FRAME_MAX_WIDTH = 768

# Post-selection critique+repair: after the specialist+selection stage picks a winner per
# style, critique each winner against the source frames (shares judge.py's accuracy/tone_fit
# rubric via pipeline.build_critique_prompt/build_critique_schema) and rewrite, once, any
# style scoring below CRITIQUE_THRESHOLD on either axis. Adds up to 2 extra Claude calls per
# clip. Falls back to the pre-critique captions on any failure — see pipeline.caption_video.
# Track 2 injects no env vars at `docker run` time, so this default governs the graded run;
# override at image build time via `--build-arg ENABLE_CRITIQUE_REPAIR=false` (see Dockerfile).
ENABLE_CRITIQUE_REPAIR = os.environ.get("ENABLE_CRITIQUE_REPAIR", "true").strip().lower() not in ("false", "0", "no", "")
CRITIQUE_THRESHOLD = int(os.environ.get("CRITIQUE_THRESHOLD", "4"))

# Gemini API key: used only by the standalone, capped video-native description probe
# (not wired into main.py or the graded pipeline). Borrowed credit — keep usage minimal.
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
