"""處理上傳影片暫存、一般影格讀取及容器 metadata。"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path
from typing import Any, BinaryIO, Callable

from PIL import Image

from engine.errors import InferenceError


def uploaded_video_path(filename: str | None, directory: str | Path) -> Path:
    suffix = Path(filename or "video.mp4").suffix.lower()
    if not suffix or len(suffix) > 12 or not suffix[1:].isalnum():
        suffix = ".mp4"
    return Path(directory) / f"upload{suffix}"


def save_upload(upload_stream: BinaryIO, destination: Path, max_bytes: int) -> int:
    """以固定大小區塊寫入暫存檔，避免把整段影片放入記憶體。"""
    size = 0
    with destination.open("wb") as output:
        while chunk := upload_stream.read(1024 * 1024):
            size += len(chunk)
            if size > max_bytes:
                raise InferenceError(413, f"Uploaded video exceeds the configured {max_bytes}-byte limit.")
            output.write(chunk)
    if size == 0:
        raise InferenceError(400, "Uploaded video is empty.")
    return size


def sample_video_frames(
    video_path: Path,
    *,
    max_frames: int,
    max_frame_side: int,
    max_duration_seconds: float,
    resize: Callable[[Image.Image, int], Image.Image],
) -> tuple[list[Image.Image], dict[str, Any]]:
    """均勻讀取影片影格並回傳來源 metadata；影格策略參數由模型設定提供。"""
    import cv2

    if max_frames < 1 or max_frame_side < 1:
        raise RuntimeError("video_max_frames and video_max_frame_side must be positive in the API YAML.")
    try:
        capture = cv2.VideoCapture(str(video_path))
    except cv2.error as exc:
        raise InferenceError(400, "Could not open uploaded video.") from exc
    try:
        if not capture.isOpened():
            raise InferenceError(400, "Uploaded file is not a supported video.")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count_value = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width_value = float(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height_value = float(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if not all(math.isfinite(value) for value in (fps, frame_count_value, width_value, height_value)):
            raise InferenceError(400, "Video metadata contains invalid values.")
        frame_count, width, height = int(frame_count_value), int(width_value), int(height_value)
        if fps <= 0 or frame_count <= 0 or width <= 0 or height <= 0:
            raise InferenceError(400, "Could not read video frame count or frame rate.")
        duration = frame_count / fps
        if not math.isfinite(duration):
            raise InferenceError(400, "Video metadata contains an invalid duration.")
        if duration > max_duration_seconds:
            raise InferenceError(413, f"Video duration {duration:.1f}s exceeds the configured {max_duration_seconds:.1f}s limit.")

        sample_count = min(max_frames, frame_count)
        if sample_count == 1:
            indices = [frame_count // 2]
        else:
            indices = [round(index * (frame_count - 1) / (sample_count - 1)) for index in range(sample_count)]
        frames: list[Image.Image] = []
        timestamps: list[float] = []
        for frame_index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok or frame is None or frame.size == 0:
                raise InferenceError(400, f"Could not decode sampled video frame {frame_index}.")
            try:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            except cv2.error as exc:
                raise InferenceError(400, f"Could not decode sampled video frame {frame_index}.") from exc
            frames.append(resize(Image.fromarray(rgb), max_frame_side))
            timestamps.append(round(frame_index / fps, 3))
        return frames, {
            "duration_seconds": round(duration, 3),
            "source_fps": round(fps, 3),
            "source_frame_count": frame_count,
            "source_width": width,
            "source_height": height,
            "sampled_frame_count": len(frames),
            "sampled_timestamps_seconds": timestamps,
        }
    finally:
        capture.release()


def probe_video_metadata(video_path: Path, *, ffprobe_bin: str, max_duration_seconds: float) -> dict[str, Any]:
    """透過 ffprobe 讀取容器 metadata，不解碼整段影片。"""
    ffprobe = shutil.which(ffprobe_bin)
    if ffprobe is None and Path(ffprobe_bin).is_file():
        ffprobe = str(Path(ffprobe_bin).resolve())
    if ffprobe is None:
        raise InferenceError(503, f"Codec video mode requires ffprobe; executable not found: {ffprobe_bin}")
    command = [
        ffprobe, "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=avg_frame_rate,r_frame_rate,width,height,nb_frames,duration:format=duration",
        "-of", "json", str(video_path),
    ]
    try:
        result = subprocess.run(command, check=False, capture_output=True, text=True, timeout=30)
    except subprocess.TimeoutExpired as exc:
        raise InferenceError(504, "Timed out while reading video metadata with ffprobe.") from exc
    except OSError as exc:
        raise InferenceError(503, f"Could not start ffprobe: {exc}") from exc
    if result.returncode != 0:
        detail = (result.stderr or "ffprobe could not read the video.").strip()[-1500:]
        raise InferenceError(400, f"Uploaded file is not a supported video: {detail}")
    try:
        payload = json.loads(result.stdout)
        stream = (payload.get("streams") or [])[0]
        duration_raw = stream.get("duration") or (payload.get("format") or {}).get("duration")
        duration = float(duration_raw)
        width = int(stream.get("width") or 0)
        height = int(stream.get("height") or 0)
        fps = 0.0
        for rate in (stream.get("avg_frame_rate"), stream.get("r_frame_rate")):
            try:
                parsed_rate = float(Fraction(str(rate)))
            except (ValueError, ZeroDivisionError, TypeError):
                continue
            if math.isfinite(parsed_rate) and parsed_rate > 0:
                fps = parsed_rate
                break
        frame_count = int(stream.get("nb_frames") or 0)
        if frame_count <= 0 and duration > 0 and fps > 0:
            frame_count = round(duration * fps)
    except (ValueError, TypeError, KeyError, IndexError, ZeroDivisionError) as exc:
        raise InferenceError(400, "Could not read uploaded video metadata.") from exc
    if not math.isfinite(duration) or duration <= 0 or width <= 0 or height <= 0:
        raise InferenceError(400, "Video metadata contains invalid duration or dimensions.")
    if duration > max_duration_seconds:
        raise InferenceError(413, f"Video duration {duration:.1f}s exceeds the configured {max_duration_seconds:.1f}s limit.")
    return {
        "duration_seconds": round(duration, 3),
        "source_fps": round(fps, 3),
        "source_frame_count": frame_count,
        "source_width": width,
        "source_height": height,
        "sampled_frame_count": None,
        "sampled_timestamps_seconds": [],
    }
