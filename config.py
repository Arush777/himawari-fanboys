"""Configuration for the video captioning pipeline.

Values are read from the environment first (via .env locally, Streamlit Secrets on
Streamlit Cloud, or -e at `docker run` time). Never commit a real API key: .env is
gitignored, and the Docker image is pushed to a PUBLIC registry.
"""
import os

from dotenv import load_dotenv

load_dotenv()  # loads .env into os.environ if present; no-op if the file doesn't exist

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL_ID = os.environ.get("CLAUDE_MODEL_ID", "claude-haiku-4-5")

# Frame sampling: ~1 frame per SECONDS_PER_FRAME of video, clamped to
# [MIN_FRAMES, MAX_FRAMES]. Clips in the hidden set are 30s-2min, so this yields
# 8 frames for a 30s clip up to 20 frames for a 100s+ clip.
SECONDS_PER_FRAME = 5.0
MIN_FRAMES = 8
MAX_FRAMES = 20
FRAME_MAX_WIDTH = 768
