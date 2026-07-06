# AMD Hackathon — Track 2: Video Captioning Agent

Pipeline: download clip → sample frames with ffmpeg (~1 frame per 5s, 8–20 frames depending
on clip length) → describe frames with Claude (vision) → rewrite the description into 4
styles in one structured-outputs call (guaranteed-valid JSON). Model defaults to
`claude-haiku-4-5` (fast + cheap, vision-capable); override via `CLAUDE_MODEL_ID`.

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
├── llm_client.py         Claude API client (vision description + structured-output JSON)
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
- **`llm_client.py`** — `ClaudeClient`, a small wrapper around the official `anthropic` SDK.
  Exposes `describe_frames` (frames as base64 image blocks + prompt) and `generate_json`
  (structured outputs — the styled captions come back as guaranteed-valid JSON).
- **`video_utils.py`** — `download_video` (streams a URL to disk), `extract_frames` (evenly
  spaced JPEGs via ffmpeg, downscaled), and `frame_to_data_uri` (base64 data URI for the
  vision API).
- **`config.py`** — Calls `load_dotenv()` so values in `.env` are picked up automatically,
  then reads `ANTHROPIC_API_KEY` and `CLAUDE_MODEL_ID` from the environment, plus the
  frame-sampling constants (`SECONDS_PER_FRAME`, `MIN_FRAMES`, `MAX_FRAMES`,
  `FRAME_MAX_WIDTH`).
- **`describe_videos.py`** — A standalone dev script (not used by `main.py` or the Docker
  image) that runs each task's video through a much longer, more detailed description prompt
  and saves the results to `descriptions.json`. Useful for inspecting what the model actually
  "sees" before it gets compressed into short styled captions.
- **`.env` / `.env.example`** — `.env` holds your real key, endpoint, and model ID and is
  gitignored. `.env.example` is the same file with a placeholder key, meant to be committed so
  anyone cloning the repo knows what to fill in.

## Changing the model

The key and model ID are read from the environment, so you can switch Claude models
without touching code — just edit `.env`:

```
ANTHROPIC_API_KEY=sk-ant-...
CLAUDE_MODEL_ID=claude-haiku-4-5   # or claude-sonnet-5 for higher caption quality
```

Then rerun `python3 main.py` (see below).

## Before you build

Copy `.env.example` to `.env` and paste your real Claude API key:

```bash
cp .env.example .env
# edit .env and set ANTHROPIC_API_KEY (and optionally CLAUDE_MODEL_ID)
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
  -e ANTHROPIC_API_KEY=sk-ant-... \
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
3. In the app's **Settings → Secrets**, paste the same variables as `.env`:

   ```toml
   ANTHROPIC_API_KEY = "sk-ant-..."
   # optional: CLAUDE_MODEL_ID = "claude-haiku-4-5"
   ```

`packages.txt` makes Streamlit Cloud install ffmpeg; `app.py` copies the secrets into the
environment so `config.py` works unchanged.

## Notes / tradeoffs

- Frame count adapts to clip length: ~1 frame per 5 seconds, clamped to 8–20, downscaled
  to 768px wide. At 768px a 16:9 frame costs ~440 tokens, so even a 20-frame clip is
  ~9K input tokens — trivially within Haiku's 200K context and roughly $0.01/clip.
- One vision call to describe the frames + one structured-outputs call to rewrite into the
  4 styles — not 4 separate calls — and clips run 4-at-a-time, keeping this comfortably
  within the 10 minute container limit even for ~12 hidden clips.
- Do NOT commit `.env` or bake your real Claude API key into the image before pushing it to
  a public registry — `.env` is gitignored, and for Docker pass the key via `-e` at
  `docker run` time instead.
