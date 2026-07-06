"""Thin OpenAI-compatible client wrapper for RITS endpoints (text + vision)."""
import re
import time

from openai import OpenAI


def _strip_thinking(text: str) -> str:
    """Remove the reasoning preamble that some RITS thinking models (e.g. Qwen3/Qwen3-VL
    thinking variants) prepend. Some of these models omit the opening <think> tag but still
    emit the closing </think>, so strip by the last closing tag rather than a paired regex."""
    if "</think>" in text:
        return text.rsplit("</think>", 1)[-1].strip()
    return text.strip()


class RitsClient:
    def __init__(self, api_key: str, api_endpoint: str, model_id: str,
                 extra_header_name: str = "RITS_API_KEY", timeout: int = 120):
        clean_endpoint = api_endpoint if api_endpoint.endswith("/v1") else f"{api_endpoint}/v1"
        self.model_id = model_id
        self.client = OpenAI(
            api_key=api_key,
            base_url=clean_endpoint,
            default_headers={extra_header_name: api_key},
            timeout=timeout,
        )

    def _call(self, messages, max_tokens, temperature, retries=2, extra_body=None):
        last_err = None
        for attempt in range(retries + 1):
            try:
                completion = self.client.chat.completions.create(
                    model=self.model_id,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=1,
                    extra_body=extra_body,
                )
                text = completion.choices[0].message.content or ""
                return _strip_thinking(text)
            except Exception as e:
                last_err = e
                if attempt < retries:
                    time.sleep(2 * (attempt + 1))
        raise RuntimeError(f"LLM call failed after {retries + 1} attempts: {last_err}")

    def describe_frames(self, frame_data_uris: list[str], prompt: str,
                         max_tokens: int = 400, temperature: float = 0.4) -> str:
        content = [{"type": "text", "text": prompt}]
        for uri in frame_data_uris:
            content.append({"type": "image_url", "image_url": {"url": uri}})
        messages = [{"role": "user", "content": content}]
        return self._call(messages, max_tokens=max_tokens, temperature=temperature)

    def generate_text(self, prompt: str, max_tokens: int = 800, temperature: float = 0.7) -> str:
        messages = [{"role": "user", "content": prompt}]
        # Qwen3.5 on RITS emits a long unstructured "thinking" preamble unless disabled here.
        extra_body = {"chat_template_kwargs": {"enable_thinking": False}}
        return self._call(messages, max_tokens=max_tokens, temperature=temperature, extra_body=extra_body)
