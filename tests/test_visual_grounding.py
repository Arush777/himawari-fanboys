import threading
import unittest
from unittest.mock import patch

import pipeline


STYLES = ["formal", "sarcastic", "humorous_tech", "humorous_non_tech"]


class FakeClient:
    def __init__(self):
        self.specialist_frames = []
        self.lock = threading.Lock()

    def describe_frames(self, frames_b64, prompt, max_tokens=1024):
        return "A tan dog runs across a sunny field toward a distant tree line."

    def generate_json(self, prompt, schema, frames_b64=None, max_tokens=1024):
        if set(schema.get("properties", {})) == {"a", "b"}:
            with self.lock:
                self.specialist_frames.append(frames_b64)
            return {
                "a": "A tan dog runs across a sunny field toward the tree line.",
                "b": "A dog crosses bright grass and heads toward distant trees.",
            }
        return {
            style: "A tan dog runs across a sunny field toward the tree line."
            for style in schema["required"]
        }


class VisualGroundingTests(unittest.TestCase):
    def test_every_specialist_receives_the_sampled_frames(self):
        client = FakeClient()
        frames = ["frame-1", "frame-2"]

        with patch("pipeline._extract_frames_b64", return_value=frames):
            result = pipeline.caption_video("unused", STYLES, client)

        self.assertEqual(STYLES, list(result))
        self.assertTrue(all(result.values()))
        self.assertEqual(4, len(client.specialist_frames))
        self.assertTrue(all(seen == frames for seen in client.specialist_frames))


if __name__ == "__main__":
    unittest.main()
