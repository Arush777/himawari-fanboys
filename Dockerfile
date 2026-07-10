FROM --platform=linux/amd64 python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY config.py video_utils.py llm_client.py pipeline.py \
     exp_gemini_notes.py exp_kimi_hybrid.py exp_style_portfolio_exact.py main.py ./

# Track 2 rules: no env vars are injected by the harness — credentials ship inside
# the image. Keys are supplied at build time (--build-arg), never committed to git.
ARG ANTHROPIC_API_KEY=""
ARG FIREWORKS_API_KEY=""
ARG GEMINI_API_KEY=""
ENV ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
ENV FIREWORKS_API_KEY=${FIREWORKS_API_KEY}
ENV GEMINI_API_KEY=${GEMINI_API_KEY}
ENV INPUT_PATH=/input/tasks.json
ENV OUTPUT_PATH=/output/results.json
ENV MAX_WORKERS=2

CMD ["python", "main.py"]
