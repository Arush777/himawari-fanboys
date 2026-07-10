# AMD Hackathon — Track 2: Video Captioning Agent

Graded pipeline: download each clip once → sample 8–20 chronological frames → run four
independent, style-specialized routes concurrently → emit one caption per requested style.
Formal uses Sonnet facts plus a Kimi K2.6 visual audit/writer; sarcastic uses the proven
Sonnet specialist; tech humor uses Gemini continuous-video temporal notes before Sonnet
description/styling; non-tech humor uses the grounded Patch-v1 Sonnet route. Each Sonnet
specialist generates two candidates and a frame-grounded selector chooses verbatim.

This exact runtime scored **57.29% vs 42.71%** against the previous 0.87 champion in a
clean 48-duel local pairwise evaluation and completed all 12 clips in 6m14s.

Two ways to run the same pipeline:

- **`main.py`** — thin batch entry point for the graded portfolio runtime
- **`app.py`** — stable interactive Sonnet demo (paste a video URL, pick styles, see captions)

A third script, **`judge.py`**, is an LLM-as-a-judge: it re-examines each clip's frames and
scores a `results.json`'s captions for accuracy (does it reflect the video) and tone_fit
(does it match the requested style).

## Project structure

```
.
├── main.py              entry point: reads /input/tasks.json, writes /output/results.json
├── app.py               Streamlit demo UI wrapping the same pipeline
├── exp_style_portfolio_exact.py  graded four-route portfolio orchestration
├── exp_gemini_notes.py           continuous-video temporal-note helper
├── exp_kimi_hybrid.py            Kimi K2.6 audit/writer helper
├── judge.py             LLM-as-judge: scores a results.json against the source videos
├── packages.txt         system deps for Streamlit Cloud (ffmpeg)
├── pipeline.py           core video -> caption logic (describe frames, rewrite into styles)
├── llm_client.py         Claude + Fireworks clients (vision description + structured JSON)
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

- **`main.py`** — The entry point that actually gets graded. It invokes the exact
  portfolio runtime measured by the local pairwise gate.
- **`exp_style_portfolio_exact.py`** — Reads tasks, shares only deterministic video/frame
  preprocessing, runs four independent source-faithful routes, applies global per-style
  routing, and writes the required results schema. A failed selected route falls back to
  the independently generated champion caption.
- **`pipeline.py`** — The actual captioning logic. `_describe_video` downloads the clip,
  extracts frames, and asks the vision model for a compact 3–5 sentence factual description.
  `caption_video` takes that description and asks the same model to rewrite it into the 4
  requested styles (`formal`, `sarcastic`, `humorous_tech`, `humorous_non_tech`) as one JSON
  object in a single call.
- **`llm_client.py`** — `ClaudeClient` for the submitted Sonnet generator plus
  `FireworksClient` for judge/dev experiments. Both expose `describe_frames` (frames as
  base64 image blocks + prompt) and `generate_json` (structured JSON for the styled
  captions).
- **`judge.py`** — Standalone LLM-as-a-judge tool. For each task in a `results.json`, it
  re-downloads the clip and re-samples frames (the same ground truth the captioning model
  saw — not the text description it produced, so the judge isn't just checking
  self-consistency), then asks the model to score each style's caption 1-5 on `accuracy`
  and `tone_fit` with a short justification. Writes a summary + per-task scores to
  `judged_results.json`. Not part of the graded contract.
- **`video_utils.py`** — `download_video` (streams a URL to disk), `extract_frames` (evenly
  spaced JPEGs via ffmpeg, downscaled), and `frame_to_b64` (raw base64 for the vision API).
- **`config.py`** — Calls `load_dotenv()` so values in `.env` are picked up automatically,
  then reads `ANTHROPIC_API_KEY`, `CLAUDE_MODEL_ID`, optional Fireworks judge/dev settings,
  and the frame-sampling constants (`SECONDS_PER_FRAME`, `MIN_FRAMES`, `MAX_FRAMES`,
  `FRAME_MAX_WIDTH`).
- **`describe_videos.py`** — A standalone dev script (not used by `main.py` or the Docker
  image) that runs each task's video through a much longer, more detailed description prompt
  and saves the results to `descriptions.json`. Useful for inspecting what the model actually
  "sees" before it gets compressed into short styled captions.
- **`.env` / `.env.example`** — `.env` holds your real key and model ID and is gitignored.
  `.env.example` is the same file with a placeholder key, meant to be committed so anyone
  cloning the repo knows what to fill in.

## Changing the model

The three provider keys and Sonnet model ID are read from the environment. For local
runs, configure `.env`:

```
ANTHROPIC_API_KEY=sk-ant-...
FIREWORKS_API_KEY=fw_...
GEMINI_API_KEY=...
CLAUDE_MODEL_ID=claude-sonnet-5
```

Then rerun `python3 main.py` (see below).

### Submission model history

The live 0.87 champion used Claude Sonnet. The new locally approved portfolio keeps
Sonnet as its core model and adds narrowly routed Kimi and Gemini assistance.

## Before you build

Copy `.env.example` to `.env` and paste all three provider keys:

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

`.env` is loaded automatically by `config.py`. If the interactive shell already exports
stale keys, explicitly override them from `.env` before a paid run.

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

By default the judge scores with Fireworks when `JUDGE_PROVIDER=fireworks`. To judge with
Claude instead, set:

```bash
export JUDGE_PROVIDER=anthropic
export ANTHROPIC_API_KEY=sk-ant-...
export JUDGE_MODEL_ID=claude-sonnet-5
python3 judge.py
```

## Build & run in Docker (this is what actually gets graded)

```bash
docker buildx build --platform linux/amd64 --tag video-captioner:latest --load .

docker run --rm \
  -e ANTHROPIC_API_KEY=sk-ant-... \
  -e FIREWORKS_API_KEY=fw_... \
  -e GEMINI_API_KEY=... \
  -v "$(pwd)/sample_input:/input:ro" \
  -v "$(pwd)/sample_output:/output" \
  video-captioner:latest

cat sample_output/results.json
```

## Push for submission

The grading harness injects no env vars, so all provider keys must be baked into the image
at build time. A plain `docker run` against the pushed image with no `-e` must work:

```bash
docker buildx build --platform linux/amd64 \
  --build-arg ANTHROPIC_API_KEY=sk-ant-... \
  --build-arg FIREWORKS_API_KEY=fw_... \
  --build-arg GEMINI_API_KEY=... \
  --tag <registry>/<you>/video-captioner:<immutable-tag> --push .
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
   ANTHROPIC_API_KEY = "sk-ant-..."
   # optional: CLAUDE_MODEL_ID = "claude-opus-4-8"
   ```

`packages.txt` makes Streamlit Cloud install ffmpeg; `app.py` copies the secrets into the
environment so `config.py` works unchanged.

## Notes / tradeoffs

- Frame count adapts to clip length: ~1 frame per 5 seconds, clamped to 8–20, downscaled
  to 768px wide. At 768px a 16:9 frame costs a few hundred tokens, so even a 20-frame clip
  is a manageable input-token cost per clip.
- Four independent style routes run concurrently inside each clip, while two clips run
  concurrently. The measured 12-clip runtime was 6m14s, leaving headroom under the
  10-minute limit.
- Do NOT commit `.env` — it's gitignored. The submitted image *does* bake the real key in
  via `--build-arg` (see "Push for submission" above), since the harness injects no env
  vars; if the target registry is public, treat the key as exposed and rotate it after
  grading. For your own local Docker runs, pass the key via `-e` at `docker run` time
  instead of building it in, so a locally-built dev image never carries a copy.
