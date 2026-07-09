"""Configuration for the video captioning pipeline.

Values are read from the environment first (via .env locally, Streamlit Secrets on
Streamlit Cloud, or -e at `docker run` time). Never commit a real API key: .env is
gitignored, and the Docker image is pushed to a PUBLIC registry.
"""
import os

from dotenv import load_dotenv

load_dotenv()  # loads .env into os.environ if present; no-op if the file doesn't exist

FIREWORKS_API_KEY = os.environ.get("FIREWORKS_API_KEY", "")
# accounts/fireworks/models/qwen3p7-plus won an internal A/B test (independent
# Claude-Sonnet-5 judge, see judge.py) against kimi-k2p6 and kimi-k2p7-code on accuracy
# and tone_fit, at roughly 1/2.4 the token cost and faster per-clip latency. Other
# candidates from Fireworks' catalog either 404 (not deployed on this account/plan) or,
# in minimax-m3's case, ignored the json_schema constraint entirely under load and
# produced unparseable output — a same-family swap isn't safe without testing first.
FIREWORKS_MODEL_ID = os.environ.get("FIREWORKS_MODEL_ID", "accounts/fireworks/models/qwen3p7-plus")
FIREWORKS_BASE_URL = "https://api.fireworks.ai/inference/v1"

# judge.py is a standalone dev tool (not part of the graded pipeline) that scores
# generated captions. JUDGE_PROVIDER picks its backend: "fireworks" (default, reuses
# FIREWORKS_API_KEY) or "anthropic" (uses ANTHROPIC_API_KEY + the `anthropic` package,
# not installed by requirements.txt — a different model family avoids the judge being
# biased toward the generator's own outputs). JUDGE_MODEL_ID is that backend's model.
JUDGE_PROVIDER = os.environ.get("JUDGE_PROVIDER", "fireworks")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
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
