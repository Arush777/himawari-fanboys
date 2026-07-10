# Submission — AMD Hackathon Track 2

## Project name

Multi-Style Video Captioning Agent — team **himawari-fanboys**

## Short description (251 chars)

An AI agent that watches short video clips and captions them in four distinct voices - formal, sarcastic, tech humour, and everyday humour. Claude vision, per-style specialist prompts, best-of-two candidates, and frame-grounded selection. Scored 0.87.

## Long description

Our Track 2 entry is an autonomous agent that turns a raw video URL into four captions with sharply different voices: a publishable formal line, dry sarcasm, developer-facing tech humour, and warm everyday humour with a hard ban on technology references. The pipeline samples 8–20 evenly spaced frames with ffmpeg, has Claude vision write a factual scene description, then gives each style its own specialist call — complete with persona and tone exemplars — which drafts two candidate captions from different angles. A final frame-grounded selection call, which sees the actual pixels, picks each style's winner, preferring captions whose every claim is verifiable over flashier but shakier ones. Every stage degrades gracefully to a simpler, proven path, so no clip ever returns an empty caption, and a full hidden batch finishes well inside the 10-minute container budget. The architecture shipped only after beating its predecessor 58.3% in blind pairwise duels, and our evaluation stack — a rubric-matched LLM judge plus a probe-validated three-model pairwise ensemble — gates every change behind a 55% win rate. Hidden-leaderboard score: 0.87.

## Artifacts

- Presentation: [`presentation.pptx`](presentation.pptx)
- Repository: https://github.com/Arush777/himawari-fanboys
- Image: `ghcr.io/novicecoderinfinity/silver-octo-guacamole:latest`
