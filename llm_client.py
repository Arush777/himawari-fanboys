"""LLM clients for the captioning pipeline (vision description + styled-caption JSON).

Fireworks exposes an OpenAI-compatible Chat Completions endpoint, so this wraps the
official `openai` SDK pointed at Fireworks' base URL. The vision models available on
Fireworks (Kimi K2) are reasoning models by default: left alone they burn hundreds to
thousands of completion tokens drafting/second-guessing an answer before writing it,
which both slows down every call and risks truncating the response (mid-thought) before
`max_tokens` is reached. Passing `reasoning_effort="none"` turns that off, so `content` is
the direct answer with no chain-of-thought preamble to strip.
"""
import json

import anthropic
from openai import OpenAI


def _content_of(response) -> str:
    return response.choices[0].message.content.strip()


class FireworksClient:
    def __init__(self, api_key: str, model_id: str, base_url: str, timeout: float = 120.0):
        self.model_id = model_id
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout, max_retries=3)

    def describe_frames(self, frames_b64: list[str], prompt: str, max_tokens: int = 1024,
                        temperature: float | None = None) -> str:
        """Send JPEG frames (raw base64, chronological order) plus a prompt; return text."""
        content = [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
            for b64 in frames_b64
        ]
        content.append({"type": "text", "text": prompt})
        kwargs = {
            "model": self.model_id,
            "max_tokens": max_tokens,
            "reasoning_effort": "none",
            "messages": [{"role": "user", "content": content}],
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        response = self.client.chat.completions.create(**kwargs)
        return _content_of(response)

    def generate_json(self, prompt: str, schema: dict, frames_b64: list[str] | None = None,
                      max_tokens: int = 1024, temperature: float | None = None) -> dict:
        """Generate a JSON object guaranteed to match `schema` (structured outputs).

        If `frames_b64` is given, the frames are attached as image content alongside the
        prompt (used by judge.py to score captions directly against the source video)."""
        if frames_b64:
            content = [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
                for b64 in frames_b64
            ]
            content.append({"type": "text", "text": prompt})
        else:
            content = prompt
        kwargs = {
            "model": self.model_id,
            "max_tokens": max_tokens,
            "reasoning_effort": "none",
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "Response", "schema": schema},
            },
            "messages": [{"role": "user", "content": content}],
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        response = self.client.chat.completions.create(**kwargs)
        return json.loads(_content_of(response))


def _text_of(response) -> str:
    return "".join(block.text for block in response.content if block.type == "text").strip()


class ClaudeClient:
    """Claude/Sonnet backend for the graded captioning pipeline."""

    def __init__(self, api_key: str, model_id: str, timeout: float = 120.0):
        self.model_id = model_id
        self.client = anthropic.Anthropic(api_key=api_key, timeout=timeout, max_retries=3)

    def describe_frames(self, frames_b64: list[str], prompt: str, max_tokens: int = 1024,
                        temperature: float | None = None) -> str:
        """Send JPEG frames (raw base64, chronological order) plus a prompt; return text."""
        content = [
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}}
            for b64 in frames_b64
        ]
        content.append({"type": "text", "text": prompt})
        kwargs = {
            "model": self.model_id,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": content}],
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        response = self.client.messages.create(**kwargs)
        return _text_of(response)

    def generate_json(self, prompt: str, schema: dict, frames_b64: list[str] | None = None,
                      max_tokens: int = 1024, temperature: float | None = None) -> dict:
        """Generate a JSON object matching `schema`.

        If `frames_b64` is given, frames are included in the same request. The graded
        pipeline uses text-only JSON rewriting; judge.py uses the frame-aware path.
        """
        content = [
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}}
            for b64 in (frames_b64 or [])
        ]
        content.append({"type": "text", "text": prompt})
        kwargs = {
            "model": self.model_id,
            "max_tokens": max_tokens,
            "output_config": {"format": {"type": "json_schema", "schema": schema}},
            "messages": [{"role": "user", "content": content}],
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        response = self.client.messages.create(**kwargs)
        return json.loads(_text_of(response))


class ClaudeJudgeClient(ClaudeClient):
    """Claude backend for judge.py.

    Scoring captions with another model family can reduce self-preference bias
    (a model tends to rate its own outputs generously).
    Implements the same `generate_json(prompt, schema, frames_b64=None)` interface as
    the generator clients so judge.py can use any backend interchangeably.
    """
