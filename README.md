# 🚨 CRITICAL — DO NOT DELETE — IMPORTANT HANDOFF NOTE 🚨

> **@Arush777 — READ THIS FIRST when you're back (written 2026-07-11). DO NOT DELETE
> this section or the `submission/` folder or the backup branch until the hackathon
> results are final.**
>
> 1. **`main` was reverted to the 0.87 pipeline.** Your portfolio runtime (`811a3f0`)
>    scored **0.86** on the hidden leaderboard, so the graded runtime (`main.py`,
>    `Dockerfile`, `requirements.txt`) was restored to the 0.87 state in `54bcc43`.
>    **NOTHING IS LOST**: your full 0.86 state is preserved on branch
>    **`backup/portfolio-runtime-0.86`**. judge2, probes, and your `exp_*` modules are
>    still on `main` (they're just no longer the graded entry point).
> 2. **IMPORTANT: the submission Docker image currently lives in a DIFFERENT repo's
>    namespace**, because GHCR does not let a repo collaborator create packages under
>    your account. The graded image is
>    **`ghcr.io/novicecoderinfinity/silver-octo-guacamole:latest`** (public, digest
>    `94109ac…`), built from this repo at `54bcc43` and verified harness-style (fresh
>    anonymous pull, no env vars, all captions non-empty, ~1.5 min). The latest
>    submission call used **this repo + that image URL**. To bring the image home to
>    your namespace, rebuild from `main` per "Push for submission" below and push to
>    `ghcr.io/arush777/video-captioner:latest`, then update the submission.
> 3. **The submission-form files are in [`submission/`](submission/)** — ready to
>    copy-paste: `SUBMISSION.md` has the short (≤255 chars) and long descriptions, and
>    `presentation.pptx` is the 6-slide deck. Both describe the 0.87 pipeline that is
>    actually being graded.
> 4. Improvement work is duel-tested and waiting (facts grounding, critique/repair with
>    verify-before-accept, tech-word guard): currently a **11–11–26 tie** vs this 0.87
>    code under judge2 on 12 clips — ships only if it clears the 0.55 gate on a bigger
>    task set.

# Himawari — Style-Aware Video Captioning

Himawari is a video-captioning agent built for AMD Developer Hackathon Track 2. It
accepts short public video URLs and produces grounded captions in each requested tone:

- `formal`
- `sarcastic`
- `humorous_tech`
- `humorous_non_tech`

The implementation combines deterministic video preprocessing with external multimodal
APIs, style-aware generation, output validation, and layered fallbacks. Provider and
model choices are intentionally treated as deployment configuration rather than part of
the public project description.

## Interfaces

- `main.py` — batch entry point used by the grading harness
- `app.py` — Streamlit demonstration for interactive use
- `judge.py` and `judge2.py` — local quality-assurance tools; not part of the graded
  runtime

## Evaluator contract

The Docker image reads:

```text
/input/tasks.json
```

and writes:

```text
/output/results.json
```

Input format:

```json
[
  {
    "task_id": "example-1",
    "video_url": "https://example.com/video.mp4",
    "styles": [
      "formal",
      "sarcastic",
      "humorous_tech",
      "humorous_non_tech"
    ]
  }
]
```

Output format:

```json
[
  {
    "task_id": "example-1",
    "captions": {
      "formal": "...",
      "sarcastic": "...",
      "humorous_tech": "...",
      "humorous_non_tech": "..."
    }
  }
]
```

The runtime preserves task IDs, emits every requested style, and uses fallback paths to
avoid empty captions when an upstream service is temporarily unavailable.

## Public architecture overview

```text
video URL
  -> download and inspect media
  -> sample chronological visual evidence
  -> external multimodal analysis
  -> style-aware caption generation
  -> validation and fallback handling
  -> results.json
```

Detailed routing, provider selection, prompts, and evaluation strategy are intentionally
not documented in this public overview.

## Project structure

```text
.
├── main.py              graded batch entry point
├── app.py               Streamlit demo
├── pipeline.py          reusable captioning pipeline
├── llm_client.py        external API adapters
├── video_utils.py       download and ffmpeg/ffprobe helpers
├── config.py            environment and runtime configuration
├── judge.py             local absolute QA tool
├── judge2.py            local pairwise QA tool
├── Dockerfile           Linux/AMD64 evaluator image
├── requirements.txt     Python dependencies
├── packages.txt         system packages for Streamlit
├── .env.example         configuration template
├── sample_input/
│   └── tasks.json
└── sample_output/
    └── results.json
```

## Local setup

Requirements:

- Python 3.11+
- ffmpeg and ffprobe
- authorized credentials for the configured external APIs

Install and configure:

```bash
pip install -r requirements.txt
cp .env.example .env
```

Fill the required values in `.env`. Never commit this file.

Run the batch pipeline:

```bash
export INPUT_PATH="$(pwd)/sample_input/tasks.json"
export OUTPUT_PATH="$(pwd)/sample_output/results.json"
python3 main.py
python3 -m json.tool sample_output/results.json
```

Run the demo:

```bash
streamlit run app.py
```

The custom URL option expects a direct, publicly accessible video file URL rather than a
webpage or private preview link.

## Local Docker verification

Build a development image:

```bash
docker buildx build \
  --platform linux/amd64 \
  --tag video-captioner:local \
  --load .
```

Run it with evaluator-style mounts:

```bash
docker run --rm \
  --platform linux/amd64 \
  --env-file .env \
  -v "$(pwd)/sample_input:/input:ro" \
  -v "$(pwd)/sample_output:/output" \
  video-captioner:local
```

Verify that the process exits successfully, all task IDs match, every requested style is
present, and no caption is empty.

## Streamlit deployment

The demo deploys from `app.py`. Configure the same external-API variables listed in
`.env.example` through Streamlit Secrets. `packages.txt` installs the required media
tools.

## Security and publication

- Never commit `.env`, credentials, access tokens, or generated secret-bearing files.
- Load credentials from authorized local configuration.
- Treat any self-contained public evaluator image as sensitive and rotate its
  credentials after the event.
- Use immutable Docker tags for candidate builds.
- Do not move `:latest`, publish an image, deploy, or update the submission form without
  explicit authorization.

## Validation

Development uses held-out clips, schema checks, non-empty-output checks, and independent
local quality evaluation before a candidate is considered for submission. Local
evaluation is an engineering gate and is not presented as a guarantee of hidden
leaderboard performance.
