# AMD Hackathon — Track 2: Video Captioning Agent

Pipeline: download clip → sample frames with ffmpeg → describe frames with a RITS vision
model (configured via `.env`, currently Qwen3-VL-235B) → rewrite the description into 4
styles, returned as strict JSON in one call. Both steps (describing and style rewriting) go
through the same vision model — no separate text model is used.

Two ways to run the same pipeline:

- **`main.py`** — batch mode for the judging harness (`/input/tasks.json` → `/output/results.json`)
- **`app.py`** — interactive Streamlit demo (paste a video URL, pick styles, see captions)

## Project structure

```
.
├── main.py              entry point: reads /input/tasks.json, writes /output/results.json
├── app.py               Streamlit demo UI wrapping the same pipeline
├── packages.txt         system deps for Streamlit Cloud (ffmpeg)
├── pipeline.py           core video -> caption logic (describe frames, rewrite into styles)
├── llm_client.py         thin OpenAI-compatible client wrapper for the RITS endpoint
├── video_utils.py        ffmpeg/ffprobe helpers: download, extract frames, base64-encode
├── config.py             loads .env and exposes API key/endpoint/model/frame settings
├── describe_videos.py    local-only helper: dumps a long detailed description per clip
├── requirements.txt      Python dependencies
├── Dockerfile            container image definition (this is what gets graded)
├── .env                  your real API key/endpoint/model (gitignored, never committed)
├── .env.example          template showing which vars .env needs (safe to commit)
├── .gitignore            excludes .env, __pycache__, *.pyc from git
├── sample_input/
│   └── tasks.json        example input: video URLs + requested caption styles
└── sample_output/
    ├── results.json       graded output: captions per style per task
    └── descriptions.json  output of describe_videos.py (not part of grading)
```

### What each file does

- **`main.py`** — The entry point that actually gets graded. Reads tasks from
  `/input/tasks.json` (or `$INPUT_PATH` if set), calls `caption_video` for each one, and
  writes `/output/results.json` (or `$OUTPUT_PATH`). Catches per-task exceptions so one
  failing clip doesn't take down the whole batch.
- **`pipeline.py`** — The actual captioning logic. `_describe_video` downloads the clip,
  extracts frames, and asks the vision model for a compact 3–5 sentence factual description.
  `caption_video` takes that description and asks the same model to rewrite it into the 4
  requested styles (`formal`, `sarcastic`, `humorous_tech`, `humorous_non_tech`) as one JSON
  object in a single call.
- **`llm_client.py`** — `RitsClient`, a small wrapper around the `openai` SDK pointed at a RITS
  endpoint. Handles retries, strips `<think>...</think>` preambles some models emit, and
  exposes `describe_frames` (vision + text) and `generate_text` (text only).
- **`video_utils.py`** — `download_video` (streams a URL to disk), `extract_frames` (evenly
  spaced JPEGs via ffmpeg, downscaled), and `frame_to_data_uri` (base64 data URI for the
  vision API).
- **`config.py`** — Calls `load_dotenv()` so values in `.env` are picked up automatically,
  then reads `RITS_VISION_API_KEY`, `RITS_VISION_API_ENDPOINT`, `RITS_VISION_MODEL_ID` from
  the environment (with code literals as fallback), plus `NUM_FRAMES` and `FRAME_MAX_WIDTH`
  for frame sampling.
- **`describe_videos.py`** — A standalone dev script (not used by `main.py` or the Docker
  image) that runs each task's video through a much longer, more detailed description prompt
  and saves the results to `descriptions.json`. Useful for inspecting what the model actually
  "sees" before it gets compressed into short styled captions.
- **`.env` / `.env.example`** — `.env` holds your real key, endpoint, and model ID and is
  gitignored. `.env.example` is the same file with a placeholder key, meant to be committed so
  anyone cloning the repo knows what to fill in.

## Changing the model

Since the key, endpoint, and model ID are all read from the environment, you can point the
pipeline at a different RITS model without touching any code — just edit `.env`:

```
RITS_VISION_API_KEY=...
RITS_VISION_API_ENDPOINT=https://.../some-other-model
RITS_VISION_MODEL_ID=Some/Other-Model-Id
```

Then rerun `python3 main.py` (see below).

## Before you build

Copy `.env.example` to `.env` and fill in your real RITS API key:

```bash
cp .env.example .env
# edit .env and set RITS_VISION_API_KEY (and optionally the endpoint/model)
```

## Local test (no Docker)

`main.py` defaults to `/input/tasks.json` and `/output/results.json` (the paths required for
grading), but both are overridable via `INPUT_PATH`/`OUTPUT_PATH` env vars for local runs:

```bash
pip install -r requirements.txt
export INPUT_PATH="$(pwd)/sample_input/tasks.json"
export OUTPUT_PATH="$(pwd)/sample_output/results.json"
python3 main.py
cat sample_output/results.json
```

`.env` is loaded automatically by `config.py`, so no need to `export` the API key manually.

To also get a long, detailed factual description per clip (not part of grading, just for
your own inspection):

```bash
export DESCRIBE_OUTPUT_PATH="$(pwd)/sample_output/descriptions.json"
python3 describe_videos.py
```

## Build & run in Docker (this is what actually gets graded)

```bash
docker buildx build --platform linux/amd64 --tag video-captioner:latest --load .

docker run --rm \
  -e RITS_VISION_API_KEY=... \
  -v "$(pwd)/sample_input:/input:ro" \
  -v "$(pwd)/sample_output:/output" \
  video-captioner:latest

cat sample_output/results.json
```

## Push for submission

```bash
docker buildx build --platform linux/amd64 --tag <registry>/<you>/video-captioner:latest --push .
```

## Streamlit demo

Locally:

```bash
pip install -r requirements.txt
streamlit run app.py        # reads .env automatically
```

On Streamlit Community Cloud (free, no server needed):

1. Go to https://share.streamlit.io, sign in with GitHub, click **Create app**.
2. Pick this repo, branch `main`, main file `app.py`.
3. In the app's **Settings → Secrets**, paste the same three variables as `.env`:

   ```toml
   RITS_VISION_API_KEY = "..."
   RITS_VISION_API_ENDPOINT = "https://.../qwen3-vl-235b-a22b-thinking"
   RITS_VISION_MODEL_ID = "Qwen/Qwen3-VL-235B-A22B-Thinking"
   ```

`packages.txt` makes Streamlit Cloud install ffmpeg; `app.py` copies the secrets into the
environment so `config.py` works unchanged. Note: the RITS endpoint must be reachable from
Streamlit's servers (public internet) for the hosted demo to work.

## Notes / tradeoffs

- 10 evenly-spaced frames per clip are sent to the vision model, downscaled to 512px wide.
  The RITS vision endpoint hard-rejects requests with more than 10 images, so 10 is the max
  this pipeline can use without erroring.
- One vision call to describe the frames + one more call (same model) to rewrite into the 4
  styles — not 4 separate calls — keeping this comfortably within the 10 minute container
  limit even for ~12 hidden clips.
- Do NOT commit `.env` or bake your real RITS API key into the image before pushing it to a
  public registry — `.env` is gitignored, and for Docker pass the key via `-e` at
  `docker run` time instead.
