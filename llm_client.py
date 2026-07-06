"""Claude API client for the captioning pipeline (vision description + styled-caption JSON)."""
import json

import anthropic


def _text_of(response) -> str:
    return "".join(block.text for block in response.content if block.type == "text").strip()


class ClaudeClient:
    def __init__(self, api_key: str, model_id: str, timeout: float = 120.0):
        self.model_id = model_id
        self.client = anthropic.Anthropic(api_key=api_key, timeout=timeout, max_retries=3)

    def describe_frames(self, frames_b64: list[str], prompt: str, max_tokens: int = 1024) -> str:
        """Send JPEG frames (raw base64, chronological order) plus a prompt; return text."""
        content = [
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}}
            for b64 in frames_b64
        ]
        content.append({"type": "text", "text": prompt})
        response = self.client.messages.create(
            model=self.model_id,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": content}],
        )
        return _text_of(response)

    def generate_json(self, prompt: str, schema: dict, max_tokens: int = 1024) -> dict:
        """Generate a JSON object guaranteed to match `schema` (structured outputs)."""
        response = self.client.messages.create(
            model=self.model_id,
            max_tokens=max_tokens,
            output_config={"format": {"type": "json_schema", "schema": schema}},
            messages=[{"role": "user", "content": prompt}],
        )
        return json.loads(_text_of(response))
