FROM --platform=linux/amd64 python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY config.py video_utils.py llm_client.py pipeline.py main.py ./

# Track 2 rules: no env vars are injected by the harness — credentials ship inside
# the image. The key is supplied at build time (--build-arg), never committed to git.
ARG ANTHROPIC_API_KEY=""
ENV ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}

# Safety valve for the post-selection critique/repair pass (see config.py). Since the
# harness injects no env vars at `docker run` time, this is the only way to disable it for
# a graded run if needed: `--build-arg ENABLE_CRITIQUE_REPAIR=false` at build time.
ARG ENABLE_CRITIQUE_REPAIR="true"
ENV ENABLE_CRITIQUE_REPAIR=${ENABLE_CRITIQUE_REPAIR}

CMD ["python", "main.py"]
