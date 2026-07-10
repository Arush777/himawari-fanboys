"""Video download and frame extraction utilities (requires ffmpeg/ffprobe on PATH)."""
import base64
import os
import subprocess

import requests


def download_video(url: str, dest_path: str, timeout: int = 120) -> None:
    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)


def get_duration_seconds(video_path: str) -> float:
    out = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", video_path,
        ],
        capture_output=True, text=True, check=True,
    )
    return float(out.stdout.strip())


def choose_num_frames(duration_s: float, seconds_per_frame: float = 5.0,
                      min_frames: int = 8, max_frames: int = 20) -> int:
    """~1 frame per `seconds_per_frame` of video, clamped so short clips still get
    enough temporal coverage and long clips don't blow up the request size."""
    return max(min_frames, min(max_frames, round(duration_s / seconds_per_frame)))


def extract_frames(video_path: str, out_dir: str, num_frames: int = 8, max_width: int = 512) -> list[str]:
    """Extract `num_frames` evenly spaced JPEG frames, downscaled to max_width, in order."""
    duration = max(get_duration_seconds(video_path), 1.0)
    fps = num_frames / duration
    pattern = os.path.join(out_dir, "frame_%03d.jpg")
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", video_path,
            "-vf", f"fps={fps},scale='min({max_width},iw)':-2",
            "-vframes", str(num_frames),
            "-q:v", "3",
            pattern,
        ],
        capture_output=True, check=True,
    )
    return sorted(
        os.path.join(out_dir, f) for f in os.listdir(out_dir) if f.startswith("frame_")
    )


def frame_to_b64(frame_path: str) -> str:
    """Raw base64 of a JPEG frame (llm_client.py wraps this into a data: URI for the API)."""
    with open(frame_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")
