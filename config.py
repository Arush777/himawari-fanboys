"""Configuration for the video captioning pipeline.

Values are read from the environment first (via .env locally, Streamlit Secrets on
Streamlit Cloud, or -e at `docker run` time). Never commit a real API key: .env is
gitignored, and the Docker image is pushed to a PUBLIC registry.
"""
import os

from dotenv import load_dotenv

load_dotenv()  # loads .env into os.environ if present; no-op if the file doesn't exist

FIREWORKS_API_KEY = os.environ.get("FIREWORKS_API_KEY", "")
FIREWORKS_MODEL_ID = os.environ.get("FIREWORKS_MODEL_ID", "accounts/fireworks/models/kimi-k2p6")
FIREWORKS_BASE_URL = "https://api.fireworks.ai/inference/v1"

# Separate (optionally different, e.g. a stronger model) model used by judge.py to score
# generated captions. Defaults to the same model used for generation.
JUDGE_MODEL_ID = os.environ.get("JUDGE_MODEL_ID", FIREWORKS_MODEL_ID)

# Frame sampling: ~1 frame per SECONDS_PER_FRAME of video, clamped to
# [MIN_FRAMES, MAX_FRAMES]. Clips in the hidden set are 30s-2min, so this yields
# 8 frames for a 30s clip up to 20 frames for a 100s+ clip.
SECONDS_PER_FRAME = 5.0
MIN_FRAMES = 8
MAX_FRAMES = 20
FRAME_MAX_WIDTH = 768
