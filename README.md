# AMD Hackathon — Track 2: Video Captioning Agent

Pipeline: download clip → sample frames with ffmpeg (~1 frame per 5s, 8–20 frames depending
on clip length) → describe frames with a vision model → rewrite the description into 4
styles in one structured-outputs call (guaranteed-valid JSON). Uses the Fireworks API,
model defaults to `accounts/fireworks/models/kimi-k2p6` (vision-capable, reasoning
disabled per-call for speed); override via `FIREWORKS_MODEL_ID`.

Two ways to run the same pipeline:

- **`main.py`** — batch mode for the judging harness (`/input/tasks.json` → `/output/results.json`)
- **`app.py`** — interactive Streamlit demo (paste a video URL, pick styles, see captions)

A third script, **`judge.py`**, is an LLM-as-a-judge: it re-examines each clip's frames and
scores a `results.json`'s captions for accuracy (does it reflect the video) and tone_fit
(does it match the requested style).

## Project structure

```
.
├── main.py              entry point: reads /input/tasks.json, writes /output/results.json
├── app.py               Streamlit demo UI wrapping the same pipeline
├── judge.py             LLM-as-judge: scores a results.json against the source videos
├── packages.txt         system deps for Streamlit Cloud (ffmpeg)
├── pipeline.py           core video -> caption logic (describe frames, rewrite into styles)
├── llm_client.py         Fireworks API client (vision description + structured-output JSON)
├── video_utils.py        ffmpeg/ffprobe helpers: download, extract frames, base64-encode
├── config.py             loads .env and exposes API key/endpoint/model/frame settings
├── describe_videos.py    local-only helper: dumps a long detailed description per clip
├── requirements.txt      Python dependencies
├── Dockerfile            container image definition (this is what gets graded)
├── .env                  your real API key/model (gitignored, never committed)
├── .env.example          template showing which vars .env needs (safe to commit)
├── .gitignore            excludes .env, __pycache__, *.pyc from git
├── sample_input/
│   └── tasks.json        example input: video URLs + requested caption styles
└── sample_output/
    ├── results.json       graded output: captions per style per task
    ├── descriptions.json  output of describe_videos.py (not part of grading)
    └── judged_results.json output of judge.py (not part of grading)
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
- **`llm_client.py`** — `FireworksClient`, a small wrapper around the official `openai` SDK
  pointed at Fireworks' OpenAI-compatible endpoint. Exposes `describe_frames` (frames as
  base64 image blocks + prompt) and `generate_json` (structured outputs — the styled
  captions come back as guaranteed-valid JSON). Every call passes `reasoning_effort="none"`:
  the vision models Fireworks offers are reasoning models by default, and left alone they
  burn hundreds to thousands of tokens second-guessing an answer before writing it —
  disabling that keeps latency and cost down and `content` free of chain-of-thought text.
- **`judge.py`** — Standalone LLM-as-a-judge tool. For each task in a `results.json`, it
  re-downloads the clip and re-samples frames (the same ground truth the captioning model
  saw — not the text description it produced, so the judge isn't just checking
  self-consistency), then asks the model to score each style's caption 1-5 on `accuracy`
  and `tone_fit` with a short justification. Writes a summary + per-task scores to
  `judged_results.json`. Not part of the graded contract.
- **`video_utils.py`** — `download_video` (streams a URL to disk), `extract_frames` (evenly
  spaced JPEGs via ffmpeg, downscaled), and `frame_to_b64` (raw base64 for the vision API).
- **`config.py`** — Calls `load_dotenv()` so values in `.env` are picked up automatically,
  then reads `FIREWORKS_API_KEY`, `FIREWORKS_MODEL_ID`, and `JUDGE_MODEL_ID` from the
  environment, plus the frame-sampling constants (`SECONDS_PER_FRAME`, `MIN_FRAMES`,
  `MAX_FRAMES`, `FRAME_MAX_WIDTH`).
- **`describe_videos.py`** — A standalone dev script (not used by `main.py` or the Docker
  image) that runs each task's video through a much longer, more detailed description prompt
  and saves the results to `descriptions.json`. Useful for inspecting what the model actually
  "sees" before it gets compressed into short styled captions.
- **`.env` / `.env.example`** — `.env` holds your real key and model ID and is gitignored.
  `.env.example` is the same file with a placeholder key, meant to be committed so anyone
  cloning the repo knows what to fill in.

## Changing the model

The key and model ID are read from the environment, so you can switch models without
touching code — just edit `.env`:

```
FIREWORKS_API_KEY=fw_...
FIREWORKS_MODEL_ID=accounts/fireworks/models/kimi-k2p6   # vision-capable models only
# JUDGE_MODEL_ID=accounts/fireworks/models/kimi-k2p6      # optional, defaults to FIREWORKS_MODEL_ID
```

Then rerun `python3 main.py` (see below). Note: your Fireworks account needs access to a
vision-capable model (`supports_image_input: true` when listing `/v1/models`) — check
`https://fireworks.ai/models` for what's enabled on your key.

## Before you build

Copy `.env.example` to `.env` and paste your real Fireworks API key:

```bash
cp .env.example .env
# edit .env and set FIREWORKS_API_KEY (and optionally FIREWORKS_MODEL_ID)
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

To score a results.json with the LLM judge (not part of grading, useful for QA before
submitting):

```bash
export RESULTS_PATH="$(pwd)/sample_output/results.json"
export JUDGE_OUTPUT_PATH="$(pwd)/sample_output/judged_results.json"
python3 judge.py
cat sample_output/judged_results.json
```

## Build & run in Docker (this is what actually gets graded)

```bash
docker buildx build --platform linux/amd64 --tag video-captioner:latest --load .

docker run --rm \
  -e FIREWORKS_API_KEY=fw_... \
  -v "$(pwd)/sample_input:/input:ro" \
  -v "$(pwd)/sample_output:/output" \
  video-captioner:latest

cat sample_output/results.json
```

## Push for submission

The grading harness injects no env vars (Track 2 rules), so the real key must be baked
into the image at build time via `--build-arg` — a plain `docker run` against the pushed
image with no `-e` still needs to work:

```bash
docker buildx build --platform linux/amd64 \
  --build-arg FIREWORKS_API_KEY=fw_... \
  --tag <registry>/<you>/video-captioner:latest --push .
```

The image (including the baked-in key) goes to a registry — if that registry is public,
rotate the key after grading.

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
   FIREWORKS_API_KEY = "fw_..."
   # optional: FIREWORKS_MODEL_ID = "accounts/fireworks/models/kimi-k2p6"
   ```

`packages.txt` makes Streamlit Cloud install ffmpeg; `app.py` copies the secrets into the
environment so `config.py` works unchanged.

## Notes / tradeoffs

- Frame count adapts to clip length: ~1 frame per 5 seconds, clamped to 8–20, downscaled
  to 768px wide. At 768px a 16:9 frame costs a few hundred tokens, so even a 20-frame clip
  is a manageable input-token cost per clip.
- One vision call to describe the frames + one structured-outputs call to rewrite into the
  4 styles — not 4 separate calls — and clips run 4-at-a-time, keeping this comfortably
  within the 10 minute container limit even for ~12 hidden clips. A local end-to-end run of
  the 3 sample clips (4-way parallel) completes in well under 20 seconds.
- The Fireworks vision models available are reasoning models; every call sets
  `reasoning_effort="none"` to skip chain-of-thought. Without it, a single 4-style caption
  call can burn 1000+ completion tokens drafting and discarding alternatives before ever
  emitting JSON — several times slower and liable to hit `max_tokens` mid-thought, which
  breaks JSON parsing entirely.
- Do NOT commit `.env` — it's gitignored. The submitted image *does* bake the real key in
  via `--build-arg` (see "Push for submission" above), since the harness injects no env
  vars; if the target registry is public, treat the key as exposed and rotate it after
  grading. For your own local Docker runs, pass the key via `-e` at `docker run` time
  instead of building it in, so a locally-built dev image never carries a copy.
