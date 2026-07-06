"""LLM endpoint configuration for the video captioning pipeline.

Values are read from the environment first (via .env locally, or -e at `docker run` time),
with the literals below only as a local-dev fallback. Track 2 requires pushing your image to
a PUBLIC registry — never bake a real API key into config.py or a committed .env, since anyone
who pulls the image or clones the repo could extract it.
"""
import os

from dotenv import load_dotenv

load_dotenv()  # loads .env into os.environ if present; no-op if the file doesn't exist

# Vision model — used to look at sampled video frames (Qwen2-VL-72B-Instruct on RITS).
VISION_API_KEY = os.environ.get("RITS_VISION_API_KEY", "PASTE_YOUR_RITS_API_KEY_HERE")
VISION_API_ENDPOINT = os.environ.get(
    "RITS_VISION_API_ENDPOINT",
    "https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/qwen2-vl-72b-instruct",
)
VISION_MODEL_ID = os.environ.get("RITS_VISION_MODEL_ID", "Qwen/Qwen2-VL-72B-Instruct")

# Frame sampling.
NUM_FRAMES = 10
FRAME_MAX_WIDTH = 512
