"""提供 Mage-VL-4B Transformers 圖片對話、上傳視訊與即時視訊推論能力。"""

from __future__ import annotations

import asyncio
import base64
import binascii
import csv
import importlib.metadata
import importlib
import inspect
import io
import json
import math
import os
import secrets
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
import warnings
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Iterator, Literal
from urllib.parse import urlparse
from urllib.request import Request as URLRequest, urlopen

import torch
from fastapi import APIRouter, Body, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import Response, StreamingResponse
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from starlette.concurrency import run_in_threadpool
from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig, KernelConfig, TextIteratorStreamer
import yaml


class APIModel(BaseModel):
    model_config = ConfigDict(extra="ignore")


class ImageURL(APIModel):
    url: str
    detail: str | None = None


class TextContentPart(APIModel):
    type: Literal["text"]
    text: str


class ImageContentPart(APIModel):
    type: Literal["image_url"]
    image_url: ImageURL


class ChatMessage(APIModel):
    role: Literal["system", "user", "assistant"]
    content: str | list[TextContentPart | ImageContentPart]


class GenerationRequest(APIModel):
    max_tokens: int | None = Field(default=None, ge=1)
    max_completion_tokens: int | None = Field(default=None, ge=1)
    temperature: float | None = Field(default=None, ge=0, le=2)
    top_p: float | None = Field(default=None, gt=0, le=1)
    top_k: int | None = Field(default=None, ge=0)
    min_p: float | None = Field(default=None, ge=0, le=1)
    repetition_penalty: float | None = Field(default=None, gt=0)
    no_repeat_ngram_size: int | None = Field(default=None, ge=0)
    num_beams: int | None = Field(default=None, ge=1)


class ChatCompletionRequest(GenerationRequest):
    model: str | None = None
    messages: list[ChatMessage] = Field(min_length=1)
    stream: bool = False


class AssistantMessage(APIModel):
    role: Literal["assistant"] = "assistant"
    content: str


class TokenUsage(APIModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class CompletionChoice(APIModel):
    index: int = 0
    message: AssistantMessage
    finish_reason: Literal["stop", "length"]


class ChatCompletionResponse(APIModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[CompletionChoice]
    usage: TokenUsage


class StreamDelta(APIModel):
    role: Literal["assistant"] | None = None
    content: str | None = None


class StreamChoice(APIModel):
    index: int = 0
    delta: StreamDelta
    finish_reason: Literal["stop", "length"] | None = None


class ChatCompletionChunk(APIModel):
    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int
    model: str
    choices: list[StreamChoice]
    usage: TokenUsage | None = None


class VideoUnderstandingRequest(GenerationRequest):
    question: str = Field(min_length=1)
    video_backend: Literal["frames", "codec"] | None = None
    codec_engine: Literal["hevc", "dcvc-rt"] | None = None


class VideoMetadata(APIModel):
    backend: Literal["frames", "codec"] = "codec"
    codec_engine: str | None = None
    duration_seconds: float
    source_fps: float
    source_frame_count: int
    source_width: int
    source_height: int
    sampled_frame_count: int | None = None
    sampled_timestamps_seconds: list[float] = Field(default_factory=list)
    codec_canvas_count: int | None = None
    upload_bytes: int


class RequestTimings(APIModel):
    staging_and_decode: float
    processor: float
    generation: float
    server_total: float


class GPUMemoryUsage(APIModel):
    peak_allocated: int
    peak_reserved: int


class VideoUnderstandingResponse(ChatCompletionResponse):
    object: Literal["magevl4b.video.understanding"] = "magevl4b.video.understanding"
    video: VideoMetadata
    timing_seconds: RequestTimings
    gpu_memory_bytes: GPUMemoryUsage


class HealthResponse(APIModel):
    status: Literal["ok", "starting"]
    backend: Literal["transformers"] = "transformers"
    model: str
    revision: str
    quantization: str
    live_sessions: int
    live_video_enabled: bool = True
    gpu: str | None = None
    vram_free_bytes: int | None = None
    vram_total_bytes: int | None = None


class ModelCard(APIModel):
    id: str
    object: Literal["model"] = "model"
    created: int = 0
    owned_by: str = "local"


class ModelListResponse(APIModel):
    object: Literal["list"] = "list"
    data: list[ModelCard]


class LiveSessionCreateRequest(GenerationRequest):
    question: str = Field(min_length=1, max_length=2000)
    video_backend: Literal["frames", "codec"] | None = None
    codec_engine: Literal["hevc", "dcvc-rt"] | None = None
    sample_interval_seconds: float | None = None
    window_seconds: float | None = None
    max_duration_seconds: float | None = None


class LivePublishURLs(APIModel):
    rtsp: str
    rtmp: str


class LiveSessionSettings(APIModel):
    video_backend: Literal["frames", "codec"]
    codec_engine: str | None = None
    codec_target_canvas: int | None = None
    sample_interval_seconds: float
    window_seconds: float
    segment_seconds: float | None = None
    max_duration_seconds: float
    max_frames: int


class LiveSessionResponse(APIModel):
    id: str
    status: str
    model: str
    publish: LivePublishURLs
    events_url: str
    stop_url: str
    settings: LiveSessionSettings


class LiveSessionStopResponse(APIModel):
    id: str
    status: Literal["stopping"]


@dataclass(frozen=True)
class LiveVideoSettings:
    """管理即時視訊的傳輸、取樣與編碼片段設定。"""

    rtsp_host: str = "127.0.0.1"
    rtsp_port: int = 8554
    rtmp_port: int = 1935
    public_host: str = ""
    sample_interval_seconds: float = 2.0
    window_seconds: float = 8.0
    max_duration_seconds: float = 3600.0
    max_sessions: int = 1
    open_timeout_ms: int = 10000
    read_timeout_ms: int = 5000
    reconnect_seconds: float = 2.0
    max_frames: int = 8
    max_frame_side: int = 896
    video_backend: Literal["frames", "codec"] = "codec"
    codec_engine: Literal["hevc", "dcvc-rt"] | None = None
    codec_target_canvas: int = 32
    live_codec_target_canvas: int = 16
    codec_max_pixels: int = 150000
    codec_max_pending_segments: int = 2
    ffmpeg_bin: str = "ffmpeg"


FrameAnalyzer = Callable[[str, list[Image.Image], dict[str, Any]], dict[str, Any]]
CodecAnalyzer = Callable[[str, Path, dict[str, Any], str | None, int, int, Path | None], dict[str, Any]]
GenerationOptionsBuilder = Callable[[dict[str, Any]], dict[str, Any]]
FrameTransformer = Callable[[Image.Image, int], Image.Image]


class LiveVideoSession:
    """管理串流擷取執行緒與序列化推論執行緒。"""

    def __init__(
        self,
        session_id: str,
        question: str,
        generation_options: dict[str, Any],
        settings: LiveVideoSettings,
        inference_gate: threading.Semaphore,
        analyze_frames: FrameAnalyzer,
        analyze_video: CodecAnalyzer,
        transform_frame: FrameTransformer,
    ) -> None:
        self.session_id = session_id
        self.question = question
        self.generation_options = generation_options
        self.settings = settings
        self.inference_gate = inference_gate
        self.analyze_frames = analyze_frames
        self.analyze_video = analyze_video
        self.transform_frame = transform_frame
        self.rtsp_url = f"rtsp://{settings.rtsp_host}:{settings.rtsp_port}/{session_id}"
        self.created_at = time.monotonic()
        self.finished_at: float | None = None
        self.status = "starting"
        self.stop_event = threading.Event()
        self.condition = threading.Condition()
        self.frames: deque[tuple[float, Image.Image]] = deque(maxlen=max(1, settings.max_frames))
        self.codec_workdir = Path(tempfile.mkdtemp(prefix=f"magevl4b-live-{session_id[:8]}-")) if settings.video_backend == "codec" else None
        self.codec_segments: deque[tuple[Path, float, float]] = deque()
        self.events: deque[dict[str, Any]] = deque(maxlen=256)
        self.event_id = 0
        self.capture_thread = threading.Thread(
            target=self._codec_capture_loop if settings.video_backend == "codec" else self._capture_loop,
            name=f"magevl4b-live-capture-{session_id[:8]}",
            daemon=True,
        )
        self.inference_thread = threading.Thread(
            target=self._inference_loop,
            name=f"magevl4b-live-inference-{session_id[:8]}",
            daemon=True,
        )

    def start(self) -> None:
        """啟動串流擷取與推論執行緒。"""
        self._set_status("waiting_for_stream")
        self.capture_thread.start()
        self.inference_thread.start()

    def _emit(self, event_name: str, data: dict[str, Any]) -> None:
        """保存事件並喚醒等待中的 SSE 用戶端。"""
        with self.condition:
            self.event_id += 1
            self.events.append({"id": self.event_id, "event": event_name, "data": data})
            self.condition.notify_all()

    def _set_status(self, status: str, **details: Any) -> None:
        with self.condition:
            changed = self.status != status
            self.status = status
            self.condition.notify_all()
        if changed:
            self._emit("status", {"status": status, **details})

    def events_after(self, event_id: int) -> tuple[list[dict[str, Any]], bool]:
        """回傳指定事件識別碼之後的新事件及工作階段結束狀態。"""
        with self.condition:
            available = [event for event in self.events if event["id"] > event_id]
            return available, self.finished_at is not None

    def snapshot(self) -> list[tuple[float, Image.Image]]:
        """取得分析視窗內的影格，並限制回傳數量。"""
        cutoff = time.monotonic() - self.settings.window_seconds
        with self.condition:
            frames = [item for item in self.frames if item[0] >= cutoff]
            if not frames and self.frames:
                frames = [self.frames[-1]]
            return frames[-self.settings.max_frames :]

    def request_stop(self) -> None:
        if not self.stop_event.is_set():
            self._set_status("stopping")
            self.stop_event.set()
        with self.condition:
            self.condition.notify_all()

    def is_active(self) -> bool:
        return self.finished_at is None

    def _capture_loop(self) -> None:
        """以 OpenCV 擷取 RTSP 影格，按取樣間隔更新即時影格佇列。"""
        # OpenCV 建立 FFmpeg RTSP 連線時會讀取此選項。
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
        import cv2

        if self.settings.max_frames < 1 or self.settings.max_frame_side < 1:
            self._emit("error", {"message": "max_frames and max_frame_side must be positive."})
            self.stop_event.set()
            return

        backend = getattr(cv2, "CAP_FFMPEG", 0)
        open_options: list[int] = []
        if hasattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC"):
            open_options.extend([cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, self.settings.open_timeout_ms])
        if hasattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC"):
            open_options.extend([cv2.CAP_PROP_READ_TIMEOUT_MSEC, self.settings.read_timeout_ms])

        was_connected = False
        try:
            while not self.stop_event.is_set():
                capture = None
                try:
                    capture = cv2.VideoCapture(self.rtsp_url, backend, open_options)
                    if not capture.isOpened():
                        self._set_status("waiting_for_stream")
                        # 推流端尚未連線或連線中斷時，以可中止的等待進行重試。
                        self.stop_event.wait(self.settings.reconnect_seconds)
                        continue

                    last_sample_at = 0.0
                    while not self.stop_event.is_set():
                        ok, frame = capture.read()
                        now = time.monotonic()
                        if not ok or frame is None or frame.size == 0:
                            break
                        if now - last_sample_at < self.settings.sample_interval_seconds:
                            continue
                        try:
                            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            image = self.transform_frame(Image.fromarray(rgb), self.settings.max_frame_side)
                        except (cv2.error, ValueError):
                            continue
                        with self.condition:
                            self.frames.append((now, image))
                            self.condition.notify_all()
                        if not was_connected:
                            self._set_status("stream_connected")
                            was_connected = True
                        last_sample_at = now
                except Exception as exc:
                    self._emit("error", {"phase": "capture", "message": f"Could not read the RTSP stream: {exc}"})
                    self.stop_event.set()
                finally:
                    if capture is not None:
                        capture.release()
                if self.stop_event.is_set():
                    break
                was_connected = False
                self._set_status("reconnecting")
                self.stop_event.wait(self.settings.reconnect_seconds)
        finally:
            with self.condition:
                self.condition.notify_all()

    def _queue_completed_segments(self, manifest: Path, workdir: Path, seen: set[Path]) -> bool:
        """只將 FFmpeg 已完成封裝並列出的片段加入佇列。"""
        if not manifest.is_file():
            return False
        queued_any = False
        try:
            with manifest.open("r", encoding="utf-8", newline="") as stream:
                rows = list(csv.reader(stream))
        except (OSError, csv.Error):
            return False
        for row in rows:
            if len(row) < 3:
                continue
            try:
                segment = Path(row[0])
                if not segment.is_absolute():
                    segment = workdir / segment
                segment = segment.resolve()
                start_seconds, end_seconds = float(row[1]), float(row[2])
            except (OSError, ValueError):
                continue
            if segment in seen or not segment.is_file() or end_seconds <= start_seconds:
                continue
            seen.add(segment)
            with self.condition:
                while len(self.codec_segments) >= max(1, self.settings.codec_max_pending_segments):
                    dropped, _, _ = self.codec_segments.popleft()
                    dropped.unlink(missing_ok=True)
                    self._emit("status", {"status": "segment_dropped", "reason": "codec_inference_is_behind_live_input"})
                self.codec_segments.append((segment, start_seconds, end_seconds))
                self.condition.notify_all()
            queued_any = True
        return queued_any

    def _codec_capture_loop(self) -> None:
        """將編碼後的 RTSP 串流複製為封裝完成且對齊關鍵影格的片段。"""
        if self.codec_workdir is None:
            self._emit("error", {"phase": "capture", "message": "Codec segment directory was not initialized."})
            self.stop_event.set()
            return
        ffmpeg = shutil.which(self.settings.ffmpeg_bin)
        if ffmpeg is None and Path(self.settings.ffmpeg_bin).is_file():
            ffmpeg = str(Path(self.settings.ffmpeg_bin).resolve())
        if ffmpeg is None:
            self._emit("error", {"phase": "capture", "message": f"Codec live mode requires server FFmpeg; executable not found: {self.settings.ffmpeg_bin}"})
            self.stop_event.set()
            return

        seen: set[Path] = set()
        was_connected = False
        try:
            while not self.stop_event.is_set():
                attempt_dir = self.codec_workdir / uuid.uuid4().hex
                attempt_dir.mkdir(parents=True, exist_ok=True)
                manifest = attempt_dir / "segments.csv"
                segment_pattern = attempt_dir / "segment-%06d.ts"
                command = [
                    ffmpeg,
                    "-hide_banner", "-loglevel", "error", "-nostdin",
                    "-rtsp_transport", "tcp",
                    "-timeout", str(max(1, self.settings.read_timeout_ms) * 1000),
                    "-i", self.rtsp_url,
                    "-map", "0:v:0", "-an", "-c:v", "copy",
                    "-f", "segment",
                    "-segment_format", "mpegts",
                    "-segment_time", f"{self.settings.window_seconds:.3f}",
                    "-reset_timestamps", "1",
                    "-segment_list", str(manifest),
                    "-segment_list_type", "csv",
                    str(segment_pattern),
                ]
                process: subprocess.Popen[bytes] | None = None
                try:
                    process = subprocess.Popen(
                        command,
                        stdin=subprocess.DEVNULL,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    while not self.stop_event.is_set() and process.poll() is None:
                        if self._queue_completed_segments(manifest, attempt_dir, seen) and not was_connected:
                            self._set_status("stream_connected")
                            was_connected = True
                        self.stop_event.wait(0.25)
                    if self.stop_event.is_set() and process.poll() is None:
                        process.terminate()
                    try:
                        process.wait(timeout=max(2.0, self.settings.read_timeout_ms / 1000 + 1))
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=2)
                    if self._queue_completed_segments(manifest, attempt_dir, seen) and not was_connected:
                        self._set_status("stream_connected")
                        was_connected = True
                except OSError as exc:
                    self._emit("error", {"phase": "capture", "message": f"Could not segment the RTSP video stream: {exc}"})
                    self.stop_event.set()
                    break
                finally:
                    if process is not None and process.poll() is None:
                        process.kill()
                        process.wait(timeout=2)
                if self.stop_event.is_set():
                    break
                was_connected = False
                self._set_status("reconnecting")
                self.stop_event.wait(self.settings.reconnect_seconds)
        finally:
            with self.condition:
                self.condition.notify_all()

    def _inference_loop(self) -> None:
        """依分析視窗取出影格或片段，序列化呼叫模型並發布結果事件。"""
        last_analyzed_at: float | None = None
        last_busy_event_at = 0.0
        completion_reason = "stopped"
        try:
            while not self.stop_event.is_set():
                if time.monotonic() - self.created_at >= self.settings.max_duration_seconds:
                    completion_reason = "duration_limit"
                    break

                segment: tuple[Path, float, float] | None = None
                frames: list[tuple[float, Image.Image]] = []
                if self.settings.video_backend == "codec":
                    with self.condition:
                        if self.codec_segments:
                            segment = self.codec_segments.popleft()
                    if segment is None:
                        self.stop_event.wait(0.2)
                        continue
                else:
                    frames = self.snapshot()
                    if not frames:
                        self.stop_event.wait(0.2)
                        continue
                    newest_frame_at = frames[-1][0]
                    if last_analyzed_at is not None and newest_frame_at - last_analyzed_at < self.settings.sample_interval_seconds:
                        self.stop_event.wait(0.2)
                        continue
                if not self.inference_gate.acquire(blocking=False):
                    now = time.monotonic()
                    if now - last_busy_event_at >= 5:
                        self._emit("status", {"status": "waiting_for_model"})
                        last_busy_event_at = now
                    if segment is not None:
                        with self.condition:
                            while len(self.codec_segments) >= max(1, self.settings.codec_max_pending_segments):
                                dropped, _, _ = self.codec_segments.pop()
                                dropped.unlink(missing_ok=True)
                                self._emit("status", {"status": "segment_dropped", "reason": "codec_inference_is_behind_live_input"})
                            self.codec_segments.appendleft(segment)
                    self.stop_event.wait(0.5)
                    continue

                try:
                    self._set_status("analyzing")
                    if segment is not None:
                        segment_path, segment_start, segment_end = segment
                        result = self.analyze_video(
                            self.question,
                            segment_path,
                            self.generation_options,
                            self.settings.codec_engine,
                            self.settings.live_codec_target_canvas,
                            self.settings.codec_max_pixels,
                            self.codec_workdir / "codec-cache" if self.codec_workdir is not None else None,
                        )
                        self._emit("analysis", {
                            "analysis_id": f"live-{uuid.uuid4().hex}",
                            **result,
                            "video_backend": "codec",
                            "codec_engine": self.settings.codec_engine or CODEC_ENGINE,
                            "codec_canvas_count": result.get("codec_canvas_count"),
                            "segment_start_seconds": round(segment_start, 3),
                            "segment_end_seconds": round(segment_end, 3),
                        })
                        segment_path.unlink(missing_ok=True)
                    else:
                        images = [image for _, image in frames]
                        result = self.analyze_frames(self.question, images, self.generation_options)
                        self._emit("analysis", {
                            "analysis_id": f"live-{uuid.uuid4().hex}",
                            **result,
                            "video_backend": "frames",
                            "frame_count": len(frames),
                            "frame_timestamps_seconds": [round(at - self.created_at, 3) for at, _ in frames],
                            "latest_frame_seconds": round(newest_frame_at - self.created_at, 3),
                        })
                        last_analyzed_at = newest_frame_at
                    self._set_status("stream_connected")
                except Exception as exc:
                    completion_reason = "inference_error"
                    self._emit("error", {"phase": "inference", "message": str(exc)})
                    break
                finally:
                    self.inference_gate.release()
        finally:
            self.stop_event.set()
            with self.condition:
                self.condition.notify_all()
            if self.capture_thread.is_alive() and threading.current_thread() is not self.capture_thread:
                self.capture_thread.join(timeout=max(5.0, self.settings.read_timeout_ms / 1000 + self.settings.reconnect_seconds + 2))
            for segment_path, _, _ in list(self.codec_segments):
                segment_path.unlink(missing_ok=True)
            if self.codec_workdir is not None:
                shutil.rmtree(self.codec_workdir, ignore_errors=True)
            self.finished_at = time.monotonic()
            self._set_status("stopped", reason=completion_reason)
            self._emit("completed", {"reason": completion_reason})


class LiveVideoService:
    """管理即時工作階段，並提供 MediaMTX 與 SSE 共用的 HTTP 介面。"""

    def __init__(
        self,
        *,
        model_name: str,
        settings: LiveVideoSettings,
        inference_gate: threading.Semaphore,
        analyze_frames: FrameAnalyzer,
        analyze_video: CodecAnalyzer,
        build_generation_options: GenerationOptionsBuilder,
        transform_frame: FrameTransformer,
        create_request_model: type[BaseModel],
        session_response_model: type[BaseModel],
        stop_response_model: type[BaseModel],
    ) -> None:
        self.model_name = model_name
        self.settings = settings
        self.inference_gate = inference_gate
        self.analyze_frames = analyze_frames
        self.analyze_video = analyze_video
        self.build_generation_options = build_generation_options
        self.transform_frame = transform_frame
        self.create_request_model = create_request_model
        self.session_response_model = session_response_model
        self.stop_response_model = stop_response_model
        self._sessions: dict[str, LiveVideoSession] = {}
        self._sessions_lock = threading.Lock()
        self.router = APIRouter()
        self._register_routes()

    def active_count(self) -> int:
        with self._sessions_lock:
            return sum(session.is_active() for session in self._sessions.values())

    def stop_all(self) -> None:
        with self._sessions_lock:
            sessions = list(self._sessions.values())
        for session in sessions:
            session.request_stop()
        for session in sessions:
            if session.inference_thread.is_alive():
                session.inference_thread.join()

    def _prune_finished(self) -> None:
        expired_before = time.monotonic() - 900
        with self._sessions_lock:
            expired = [
                session_id
                for session_id, session in self._sessions.items()
                if session.finished_at is not None and session.finished_at < expired_before
            ]
            for session_id in expired:
                del self._sessions[session_id]

    def _find(self, session_id: str) -> LiveVideoSession:
        with self._sessions_lock:
            session = self._sessions.get(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Live session was not found or has expired.")
        return session

    def _register_routes(self) -> None:
        async def create_session(request: Request, payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
            """建立暫時的 MediaMTX 推流路徑，並啟動串流分析器。"""
            try:
                session_request = self.create_request_model.model_validate(payload)
            except ValidationError as exc:
                raise RequestValidationError(exc.errors()) from exc
            question = session_request.question
            if not isinstance(question, str) or not question.strip() or len(question) > 2000:
                raise HTTPException(status_code=400, detail="question must be a non-empty string of at most 2000 characters.")

            try:
                sample_interval = float(session_request.sample_interval_seconds if session_request.sample_interval_seconds is not None else self.settings.sample_interval_seconds)
                window_seconds = float(session_request.window_seconds if session_request.window_seconds is not None else self.settings.window_seconds)
                max_duration = float(session_request.max_duration_seconds if session_request.max_duration_seconds is not None else self.settings.max_duration_seconds)
            except (TypeError, ValueError) as exc:
                raise HTTPException(status_code=400, detail="Live timing options must be numbers.") from exc
            video_backend = session_request.video_backend or self.settings.video_backend
            codec_engine = session_request.codec_engine or self.settings.codec_engine
            if not 0.25 <= sample_interval <= 60:
                raise HTTPException(status_code=400, detail="sample_interval_seconds must be between 0.25 and 60.")
            if not 1 <= window_seconds <= 60 or (video_backend == "frames" and window_seconds < sample_interval):
                raise HTTPException(status_code=400, detail="window_seconds must be between 1 and 60; frame mode also requires it to be at least the sampling interval.")
            if not 1 <= max_duration <= self.settings.max_duration_seconds:
                raise HTTPException(
                    status_code=400,
                    detail=f"max_duration_seconds must be between 1 and {self.settings.max_duration_seconds:g}.",
                )
            if video_backend == "codec" and (self.settings.live_codec_target_canvas <= 0 or self.settings.codec_max_pixels <= 0):
                raise HTTPException(status_code=500, detail="Codec target_canvas and max_pixels must be positive in the API YAML.")
            if video_backend == "codec":
                _require_codec_runtime(codec_engine)
                _require_processor_codec_support()
                if codec_engine == "dcvc-rt":
                    _codec_processor_config(codec_engine, self.settings.live_codec_target_canvas, None)

            generation_request = session_request.model_dump(
                include={"max_tokens", "max_completion_tokens", "temperature", "top_p", "top_k"},
                exclude_none=True,
            )
            try:
                generation_options = self.build_generation_options(generation_request)
            except (TypeError, ValueError, OverflowError) as exc:
                raise HTTPException(status_code=400, detail=f"Invalid generation option: {exc}") from exc

            self._prune_finished()
            with self._sessions_lock:
                active_count = sum(session.is_active() for session in self._sessions.values())
                if active_count >= self.settings.max_sessions:
                    raise HTTPException(status_code=429, detail=f"At most {self.settings.max_sessions} live session(s) may run at once.")
                session_id = secrets.token_urlsafe(24)
                session_settings = LiveVideoSettings(
                    **{
                        **self.settings.__dict__,
                        "sample_interval_seconds": sample_interval,
                        "window_seconds": window_seconds,
                        "max_duration_seconds": max_duration,
                        "video_backend": video_backend,
                        "codec_engine": codec_engine,
                    }
                )
                session = LiveVideoSession(
                    session_id=session_id,
                    question=question.strip(),
                    generation_options=generation_options,
                    settings=session_settings,
                    inference_gate=self.inference_gate,
                    analyze_frames=self.analyze_frames,
                    analyze_video=self.analyze_video,
                    transform_frame=self.transform_frame,
                )
                self._sessions[session_id] = session
            session.start()

            public_host = self.settings.public_host or request.url.hostname or "127.0.0.1"
            return {
                "id": session_id,
                "status": session.status,
                "model": self.model_name,
                "publish": {
                    "rtsp": f"rtsp://{public_host}:{self.settings.rtsp_port}/{session_id}",
                    "rtmp": f"rtmp://{public_host}:{self.settings.rtmp_port}/{session_id}",
                },
                "events_url": f"{str(request.base_url).rstrip('/')}/v1/live/sessions/{session_id}/events",
                "stop_url": f"{str(request.base_url).rstrip('/')}/v1/live/sessions/{session_id}",
                "settings": {
                    "video_backend": video_backend,
                    "codec_engine": codec_engine,
                    "codec_target_canvas": self.settings.live_codec_target_canvas if video_backend == "codec" else None,
                    "sample_interval_seconds": sample_interval,
                    "window_seconds": window_seconds,
                    "segment_seconds": window_seconds if video_backend == "codec" else None,
                    "max_duration_seconds": max_duration,
                    "max_frames": self.settings.max_frames,
                },
            }

        self.router.add_api_route(
            "/v1/live/sessions",
            create_session,
            methods=["POST"],
            response_model=self.session_response_model,
        )

        @self.router.get("/v1/live/sessions/{session_id}/events")
        async def session_events(session_id: str, request: Request) -> StreamingResponse:
            session = self._find(session_id)
            try:
                last_event_id = max(0, int(request.headers.get("last-event-id", "0")))
            except ValueError:
                last_event_id = 0

            async def event_stream():
                nonlocal last_event_id
                last_keepalive = time.monotonic()
                while True:
                    if await request.is_disconnected():
                        return
                    events, finished = session.events_after(last_event_id)
                    for event in events:
                        last_event_id = event["id"]
                        data = json.dumps(event["data"], ensure_ascii=False)
                        yield f"id: {event['id']}\nevent: {event['event']}\ndata: {data}\n\n"
                    if finished and not events:
                        return
                    if time.monotonic() - last_keepalive >= 15:
                        yield ": keep-alive\n\n"
                        last_keepalive = time.monotonic()
                    await asyncio.sleep(0.25)

            return StreamingResponse(
                event_stream(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
            )

        def stop_session(session_id: str) -> dict[str, Any]:
            session = self._find(session_id)
            session.request_stop()
            return {"id": session_id, "status": "stopping"}

        self.router.add_api_route(
            "/v1/live/sessions/{session_id}",
            stop_session,
            methods=["DELETE"],
            response_model=self.stop_response_model,
        )

        @self.router.post("/v1/live/mediamtx/auth")
        async def authorize_mediamtx_client(request: Request) -> Response:
            """僅接受作用中即時工作階段建立的 RTSP／RTMP 路徑。"""
            try:
                body = await request.json()
            except Exception as exc:
                raise HTTPException(status_code=400, detail="Request body must be valid JSON.") from exc
            if not isinstance(body, dict):
                raise HTTPException(status_code=400, detail="Request body must be a JSON object.")
            action, protocol, path = body.get("action"), body.get("protocol"), body.get("path")
            if action not in {"publish", "read"} or protocol not in {"rtsp", "rtmp"} or not isinstance(path, str):
                raise HTTPException(status_code=401, detail="MediaMTX action is not allowed.")
            session_id = path.strip("/")
            with self._sessions_lock:
                session = self._sessions.get(session_id)
                allowed = session is not None and session.is_active() and not session.stop_event.is_set()
            if not allowed:
                raise HTTPException(status_code=401, detail="MediaMTX stream path is not an active live session.")
            return Response(status_code=204)


CONFIG_PATH = os.environ.get("INFERENCE_CONFIG")
if not CONFIG_PATH:
    raise RuntimeError("Set INFERENCE_CONFIG to the selected model YAML before importing the API app.")
CONFIG = yaml.safe_load(Path(CONFIG_PATH).read_text(encoding="utf-8"))
NETWORK_CONFIG_PATH = os.environ.get("INFERENCE_NETWORK_CONFIG")
if not NETWORK_CONFIG_PATH:
    raise RuntimeError("Set INFERENCE_NETWORK_CONFIG to the selected network YAML before importing the API app.")
NETWORK_CONFIG = yaml.safe_load(Path(NETWORK_CONFIG_PATH).read_text(encoding="utf-8"))
if not isinstance(NETWORK_CONFIG, dict):
    raise RuntimeError("The selected network YAML must contain a mapping at the top level.")
PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_CONFIG = CONFIG["model"]
QUANTIZATION_CONFIG = MODEL_CONFIG["quantization"]
LOCAL_FILES_ONLY = bool(CONFIG.get("runtime", {}).get("local_files_only", False))
API_CONFIG = CONFIG["api"]
GENERATION_CONFIG = CONFIG["generation"]
MODEL_ID = MODEL_CONFIG["id"]
MODEL_REVISION = MODEL_CONFIG["revision"]


def _resolve_model_load_path() -> str:
    """確保固定 revision 的模型快照位於專案 HF 快取並回傳其路徑。"""
    hf_home_value = os.environ.get("HF_HOME") or CONFIG.get("runtime", {}).get("hf_home", "checkpoints")
    hf_home = Path(os.path.expandvars(os.path.expanduser(str(hf_home_value))))
    if not hf_home.is_absolute():
        hf_home = PROJECT_ROOT / hf_home
    cache_dir = hf_home / "hub"
    try:
        from huggingface_hub import snapshot_download

        return str(
            snapshot_download(
                repo_id=MODEL_ID,
                revision=MODEL_REVISION,
                cache_dir=str(cache_dir),
                local_files_only=LOCAL_FILES_ONLY,
            )
        )
    except Exception as exc:
        raise RuntimeError(
            f"Could not prepare model revision {MODEL_REVISION} in the project cache {cache_dir}: {exc}"
        ) from exc


MODEL_LOAD_PATH = _resolve_model_load_path()
MAX_IMAGE_SIDE = int(API_CONFIG["max_image_side"])
IMAGE_RESAMPLING = str(API_CONFIG["image_resampling"]).lower()
MAX_CONTEXT_TOKENS = int(API_CONFIG["max_context_tokens"])
MAX_NEW_TOKENS = int(API_CONFIG["max_new_tokens"])
MAX_CONCURRENT_REQUESTS = int(API_CONFIG["max_concurrent_requests"])
MAX_IMAGES_PER_REQUEST = int(API_CONFIG["max_images_per_request"])
MAX_IMAGE_BYTES = int(API_CONFIG["max_image_bytes"])
MAX_IMAGE_PIXELS = int(API_CONFIG["max_image_pixels"])
MAX_CHAT_REQUEST_BYTES = int(API_CONFIG["max_chat_request_bytes"])
MAX_LIVE_REQUEST_BYTES = int(API_CONFIG["max_live_request_bytes"])
MAX_VIDEO_BYTES = int(API_CONFIG["max_video_bytes"])
MAX_VIDEO_REQUEST_BYTES = int(API_CONFIG["max_video_request_bytes"])
MAX_VIDEO_DURATION_SECONDS = float(API_CONFIG["max_video_duration_seconds"])
VIDEO_MAX_FRAMES = int(API_CONFIG["video_max_frames"])
VIDEO_MAX_FRAME_SIDE = int(API_CONFIG["video_max_frame_side"])
VIDEO_BACKEND = str(API_CONFIG.get("video_backend", "codec")).lower()
if VIDEO_BACKEND not in {"frames", "codec"}:
    raise RuntimeError("api.video_backend must be either 'frames' or 'codec'.")
CODEC_CONFIG = API_CONFIG.get("codec", {})
if not isinstance(CODEC_CONFIG, dict):
    raise RuntimeError("api.codec must be a mapping in the API YAML.")
CODEC_ENGINE = str(CODEC_CONFIG["engine"]).lower() if CODEC_CONFIG.get("engine") else None
CODEC_TARGET_CANVAS = int(CODEC_CONFIG.get("target_canvas", 32))
CODEC_MAX_PIXELS = int(CODEC_CONFIG.get("max_pixels", 150000))
CODEC_FFMPEG_BIN = str(CODEC_CONFIG.get("ffmpeg_bin", "ffmpeg"))
CODEC_FFPROBE_BIN = str(CODEC_CONFIG.get("ffprobe_bin", "ffprobe"))
CODEC_TIMEOUT_SECONDS = int(CODEC_CONFIG.get("timeout_seconds", 900))
if CODEC_ENGINE not in {None, "hevc", "dcvc-rt"}:
    raise RuntimeError("api.codec.engine must be 'hevc', 'dcvc-rt', or null.")
if CODEC_TARGET_CANVAS <= 0 or CODEC_MAX_PIXELS <= 0 or CODEC_TIMEOUT_SECONDS <= 0:
    raise RuntimeError("api.codec target_canvas, max_pixels, and timeout_seconds must be positive.")
IMAGE_URL_TIMEOUT = float(API_CONFIG["image_url_timeout_seconds"])
STREAM_TIMEOUT = float(API_CONFIG["stream_timeout_seconds"])
LIVE_CONFIG = API_CONFIG.get("live_stream", {})
if not isinstance(LIVE_CONFIG, dict):
    raise RuntimeError("api.live_stream must be a mapping in the API YAML.")
LIVE_NETWORK_CONFIG = NETWORK_CONFIG.get("live_stream", {})
if not isinstance(LIVE_NETWORK_CONFIG, dict):
    raise RuntimeError("network.live_stream must be a mapping in the network YAML.")
LIVE_SETTINGS = LiveVideoSettings(
    rtsp_host=str(LIVE_NETWORK_CONFIG.get("mediamtx_rtsp_host", "127.0.0.1")),
    rtsp_port=int(LIVE_NETWORK_CONFIG.get("mediamtx_rtsp_port", 8554)),
    rtmp_port=int(LIVE_NETWORK_CONFIG.get("mediamtx_rtmp_port", 1935)),
    # 優先使用新命名的環境變數，並暫時保留舊名稱以相容既有部署設定。
    public_host=str(
        LIVE_NETWORK_CONFIG.get("public_host")
        or os.environ.get("MAGEVL4B_LIVE_PUBLIC_HOST")
        or os.environ.get("MAGEVL_LIVE_PUBLIC_HOST", "")
    ).strip(),
    sample_interval_seconds=float(LIVE_CONFIG.get("sample_interval_seconds", 2.0)),
    window_seconds=float(LIVE_CONFIG.get("window_seconds", 8.0)),
    max_duration_seconds=float(LIVE_CONFIG.get("max_duration_seconds", 3600)),
    max_sessions=int(LIVE_CONFIG.get("max_sessions", 1)),
    open_timeout_ms=int(float(LIVE_CONFIG.get("open_timeout_seconds", 10)) * 1000),
    read_timeout_ms=int(float(LIVE_CONFIG.get("read_timeout_seconds", 5)) * 1000),
    reconnect_seconds=float(LIVE_CONFIG.get("reconnect_interval_seconds", 2.0)),
    max_frames=VIDEO_MAX_FRAMES,
    max_frame_side=VIDEO_MAX_FRAME_SIDE,
    video_backend=str(LIVE_CONFIG.get("video_backend", VIDEO_BACKEND)).lower(),
    codec_engine=CODEC_ENGINE,
    codec_target_canvas=CODEC_TARGET_CANVAS,
    live_codec_target_canvas=int(LIVE_CONFIG.get("codec_target_canvas", 16)),
    codec_max_pixels=CODEC_MAX_PIXELS,
    codec_max_pending_segments=int(LIVE_CONFIG.get("codec_max_pending_segments", 2)),
    ffmpeg_bin=CODEC_FFMPEG_BIN,
)
if LIVE_SETTINGS.video_backend not in {"frames", "codec"}:
    raise RuntimeError("api.live_stream.video_backend must be either 'frames' or 'codec'.")
if LIVE_SETTINGS.codec_max_pending_segments < 1:
    raise RuntimeError("api.live_stream.codec_max_pending_segments must be positive.")

class TransformersModelState:
    """保存已載入模型與此服務程序使用的同步物件。"""

    def __init__(self) -> None:
        self.model: Any = None
        self.processor: Any = None
        self.generation_lock = threading.Semaphore(MAX_CONCURRENT_REQUESTS)
        self.live_video: LiveVideoService | None = None


RUNTIME = TransformersModelState()
_RESAMPLING_FILTERS = {
    "nearest": Image.Resampling.NEAREST,
    "bilinear": Image.Resampling.BILINEAR,
    "bicubic": Image.Resampling.BICUBIC,
    "lanczos": Image.Resampling.LANCZOS,
}


def _resize_image(image: Image.Image, max_side: int) -> Image.Image:
    if IMAGE_RESAMPLING not in _RESAMPLING_FILTERS:
        raise RuntimeError(f"Unsupported image resampling mode in config: {IMAGE_RESAMPLING}")
    image.thumbnail((max_side, max_side), _RESAMPLING_FILTERS[IMAGE_RESAMPLING])
    return image


def _load_image(source: str) -> Image.Image:
    if source.startswith("data:"):
        header, separator, payload = source.partition(",")
        if not separator or ";base64" not in header.lower():
            raise HTTPException(status_code=400, detail="Image data URLs must use base64 encoding.")
        try:
            raw = base64.b64decode(payload, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise HTTPException(status_code=400, detail="Invalid base64 image payload.") from exc
    else:
        parsed = urlparse(source)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise HTTPException(status_code=400, detail="image_url must be an http(s) URL or a base64 data URL.")
        request = URLRequest(source, headers={"User-Agent": "MageVLDeployment/0.1"})
        try:
            with urlopen(request, timeout=IMAGE_URL_TIMEOUT) as response:
                if response.status != 200:
                    raise HTTPException(status_code=400, detail="Image URL did not return HTTP 200.")
                raw = response.read(MAX_IMAGE_BYTES + 1)
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Could not fetch image_url: {exc}") from exc
    if len(raw) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail=f"Image exceeds MAX_IMAGE_BYTES={MAX_IMAGE_BYTES}.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(raw)) as opened:
                if opened.width * opened.height > MAX_IMAGE_PIXELS:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Image exceeds max_image_pixels={MAX_IMAGE_PIXELS}.",
                    )
                opened.load()
                image = opened.convert("RGB")
    except HTTPException:
        raise
    except (Image.DecompressionBombWarning, Image.DecompressionBombError) as exc:
        raise HTTPException(
            status_code=413,
            detail=f"Image exceeds max_image_pixels={MAX_IMAGE_PIXELS}.",
        ) from exc
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=400, detail="Payload is not a supported image.") from exc
    return _resize_image(image, MAX_IMAGE_SIDE)


def _move_inputs_to_model(inputs: Any) -> dict[str, Any]:
    input_device = RUNTIME.model.get_input_embeddings().weight.device
    model_dtype = RUNTIME.model.dtype
    moved: dict[str, Any] = {}
    for key, value in inputs.items():
        if isinstance(value, torch.Tensor):
            if key in {"pixel_values", "pixel_values_videos"} and value.is_floating_point():
                moved[key] = value.to(device=input_device, dtype=model_dtype)
            else:
                moved[key] = value.to(device=input_device)
        else:
            moved[key] = value
    return moved


def _sample_video_frames(video_path: Path) -> tuple[list[Image.Image], dict[str, Any]]:
    """均勻取樣上傳視訊的影格，並回傳影格與來源中繼資料。"""
    import cv2

    if VIDEO_MAX_FRAMES < 1 or VIDEO_MAX_FRAME_SIDE < 1:
        raise RuntimeError("video_max_frames and video_max_frame_side must be positive in the API YAML.")
    try:
        capture = cv2.VideoCapture(str(video_path))
    except cv2.error as exc:
        raise HTTPException(status_code=400, detail="Could not open uploaded video.") from exc
    try:
        if not capture.isOpened():
            raise HTTPException(status_code=400, detail="Uploaded file is not a supported video.")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count_value = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width_value = float(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height_value = float(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if not all(math.isfinite(value) for value in (fps, frame_count_value, width_value, height_value)):
            raise HTTPException(status_code=400, detail="Video metadata contains invalid values.")
        frame_count = int(frame_count_value)
        width = int(width_value)
        height = int(height_value)
        if fps <= 0 or frame_count <= 0 or width <= 0 or height <= 0:
            raise HTTPException(status_code=400, detail="Could not read video frame count or frame rate.")
        duration = frame_count / fps
        if not math.isfinite(duration):
            raise HTTPException(status_code=400, detail="Video metadata contains an invalid duration.")
        if duration > MAX_VIDEO_DURATION_SECONDS:
            raise HTTPException(
                status_code=413,
                detail=f"Video duration {duration:.1f}s exceeds the configured {MAX_VIDEO_DURATION_SECONDS:.1f}s limit.",
            )

        # 均勻涵蓋整段影片，同時遵守設定的影格數上限。
        sample_count = min(VIDEO_MAX_FRAMES, frame_count)
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
                raise HTTPException(status_code=400, detail=f"Could not decode sampled video frame {frame_index}.")
            try:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            except cv2.error as exc:
                raise HTTPException(status_code=400, detail=f"Could not decode sampled video frame {frame_index}.") from exc
            frames.append(_resize_image(Image.fromarray(rgb), VIDEO_MAX_FRAME_SIDE))
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


def _probe_video_metadata(video_path: Path) -> dict[str, Any]:
    """讀取視訊容器中繼資料，避免編碼處理前先解碼所有影格。"""
    ffprobe = shutil.which(CODEC_FFPROBE_BIN)
    if ffprobe is None and Path(CODEC_FFPROBE_BIN).is_file():
        ffprobe = str(Path(CODEC_FFPROBE_BIN).resolve())
    if ffprobe is None:
        raise HTTPException(
            status_code=503,
            detail=f"Codec video mode requires ffprobe; executable not found: {CODEC_FFPROBE_BIN}",
        )
    command = [
        ffprobe, "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=avg_frame_rate,r_frame_rate,width,height,nb_frames,duration:format=duration",
        "-of", "json", str(video_path),
    ]
    try:
        result = subprocess.run(command, check=False, capture_output=True, text=True, timeout=30)
    except subprocess.TimeoutExpired as exc:
        raise HTTPException(status_code=504, detail="Timed out while reading video metadata with ffprobe.") from exc
    except OSError as exc:
        raise HTTPException(status_code=503, detail=f"Could not start ffprobe: {exc}") from exc
    if result.returncode != 0:
        detail = (result.stderr or "ffprobe could not read the video.").strip()[-1500:]
        raise HTTPException(status_code=400, detail=f"Uploaded file is not a supported video: {detail}")
    try:
        payload = json.loads(result.stdout)
        streams = payload.get("streams") or []
        stream = streams[0]
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
        raise HTTPException(status_code=400, detail="Could not read uploaded video metadata.") from exc
    if not math.isfinite(duration) or duration <= 0 or width <= 0 or height <= 0:
        raise HTTPException(status_code=400, detail="Video metadata contains invalid duration or dimensions.")
    if duration > MAX_VIDEO_DURATION_SECONDS:
        raise HTTPException(
            status_code=413,
            detail=f"Video duration {duration:.1f}s exceeds the configured {MAX_VIDEO_DURATION_SECONDS:.1f}s limit.",
        )
    return {
        "duration_seconds": round(duration, 3),
        "source_fps": round(fps, 3),
        "source_frame_count": frame_count,
        "source_width": width,
        "source_height": height,
        "sampled_frame_count": None,
        "sampled_timestamps_seconds": [],
    }


def _require_codec_runtime(engine: str | None) -> None:
    """選用編碼器缺少主機端前處理工具時，提供明確錯誤訊息。"""
    for name, executable in (("ffmpeg", CODEC_FFMPEG_BIN), ("ffprobe", CODEC_FFPROBE_BIN)):
        if shutil.which(executable) is None and not Path(executable).is_file():
            raise HTTPException(status_code=503, detail=f"Codec video mode requires {name}; executable not found: {executable}")
    try:
        importlib.metadata.version("codec-video-prep")
    except importlib.metadata.PackageNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail="Codec video mode requires the Python package codec-video-prep; run the Transformers init.sh environment setup.",
        ) from exc
    if engine == "hevc":
        cv_preinfer = os.environ.get("CV_PREINFER_BIN", "cv-preinfer")
        if shutil.which(cv_preinfer) is None and not Path(cv_preinfer).is_file():
            raise HTTPException(
                status_code=503,
                detail=f"Codec engine 'hevc' requires cv-preinfer on PATH (or CV_PREINFER_BIN): {cv_preinfer}",
            )


def _codec_processor_config(
    engine: str | None,
    target_canvas: int,
    cache_root: Path | None,
) -> dict[str, Any]:
    config: dict[str, Any] = {"target_canvas": target_canvas, "timeout_seconds": CODEC_TIMEOUT_SECONDS}
    if engine:
        config["engine"] = engine
    image_processor = getattr(RUNTIME.processor, "image_processor", None)
    patch_size = getattr(image_processor, "patch_size", None)
    if patch_size is not None:
        try:
            patch_size = int(patch_size)
        except (TypeError, ValueError):
            patch_size = 0
        if patch_size > 0:
            # 將編碼 patch 座標對齊目前載入權重使用的影像網格。
            config["patch"] = patch_size
    if cache_root is not None:
        config["cache_root"] = cache_root
    if engine == "dcvc-rt":
        model_path = Path(MODEL_LOAD_PATH)
        if not model_path.is_dir():
            try:
                from huggingface_hub import snapshot_download

                model_path = Path(snapshot_download(
                    repo_id=MODEL_ID,
                    revision=MODEL_REVISION,
                    local_files_only=LOCAL_FILES_ONLY,
                ))
            except Exception as exc:
                raise HTTPException(status_code=503, detail=f"Could not locate the loaded model snapshot for DCVC-RT: {exc}") from exc
        neural_codec_path = model_path / "neural_codec"
        if not neural_codec_path.is_dir():
            raise HTTPException(status_code=503, detail=f"DCVC-RT files are missing from the model snapshot: {neural_codec_path}")
        device = RUNTIME.model.get_input_embeddings().weight.device
        config["dcvc"] = {"pkg_dir": str(neural_codec_path), "device": str(device)}
    # 模型處理器的遠端程式碼會提供編碼設定結構；依實際載入的結構篩選
    # 服務選項，避免 API 綁定特定版權重使用的 dataclass 欄位。
    try:
        processor_module = importlib.import_module(RUNTIME.processor.__class__.__module__)
        codec_module = importlib.import_module(
            ".codec_video_processing_mage_vl",
            package=processor_module.__package__,
        )
        codec_config_type = getattr(codec_module, "CodecConfig")
        supported = set(inspect.signature(codec_config_type).parameters)
    except (ImportError, AttributeError, TypeError, ValueError):
        supported = {"engine", "target_canvas", "dcvc"}
    return {key: value for key, value in config.items() if key in supported}


def _require_processor_codec_support() -> None:
    """檢查已載入模型的處理器介面，不將程式固定綁定至特定版本。"""
    if RUNTIME.processor is None:
        raise HTTPException(status_code=503, detail="The Mage-VL-4B processor is not ready yet.")
    try:
        parameters = inspect.signature(RUNTIME.processor.__call__).parameters
    except (TypeError, ValueError):
        return
    if not {"video_backend", "codec_config"}.issubset(parameters):
        raise HTTPException(
            status_code=501,
            detail="The loaded Mage-VL-4B processor does not expose the codec video backend; use video_backend=frames or load a model snapshot that includes it.",
        )


def _video_inputs(
    question: str,
    video: list[Image.Image] | Path,
    *,
    backend: str = "frames",
    codec_engine: str | None = None,
    target_canvas: int = CODEC_TARGET_CANVAS,
    max_pixels: int = CODEC_MAX_PIXELS,
    codec_cache_root: Path | None = None,
) -> tuple[Any, int, int | None]:
    messages = [{"role": "user", "content": [
        {"type": "video"},
        {"type": "text", "text": question},
    ]}]
    prompt = RUNTIME.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if backend == "frames":
        inputs = RUNTIME.processor(text=[prompt], videos=[video], return_tensors="pt", padding=True)
    elif backend == "codec":
        _require_codec_runtime(codec_engine)
        _require_processor_codec_support()
        if not isinstance(video, Path):
            raise ValueError("Codec video processing requires a local video file path.")
        codec_config = _codec_processor_config(codec_engine, target_canvas, codec_cache_root)
        try:
            inputs = RUNTIME.processor(
                text=[prompt],
                videos=[str(video)],
                video_backend="codec",
                max_pixels=max_pixels,
                codec_config=codec_config,
                return_tensors="pt",
                padding=True,
            )
        except TypeError as exc:
            raise HTTPException(
                status_code=501,
                detail="The loaded Mage-VL-4B processor does not expose the codec video backend; use video_backend=frames or update the model processor files.",
            ) from exc
        except (ImportError, ModuleNotFoundError, FileNotFoundError) as exc:
            raise HTTPException(status_code=503, detail=f"Codec preprocessing dependency is unavailable: {exc}") from exc
        except subprocess.TimeoutExpired as exc:
            raise HTTPException(status_code=504, detail="Codec video preprocessing exceeded its configured timeout.") from exc
        except (RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=f"Codec video preprocessing failed: {exc}") from exc
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported video_backend: {backend}")
    input_tokens = int(inputs["input_ids"].shape[-1])
    grids = inputs.get("image_grid_thw")
    canvas_count = int(grids.shape[0]) if backend == "codec" and isinstance(grids, torch.Tensor) else None
    return _move_inputs_to_model(inputs), input_tokens, canvas_count


def _uploaded_video_path(filename: str | None, directory: str) -> Path:
    suffix = Path(filename or "video.mp4").suffix.lower()
    if not suffix or len(suffix) > 12 or not suffix[1:].isalnum():
        suffix = ".mp4"
    return Path(directory) / f"upload{suffix}"


def _save_upload(video: UploadFile, destination: Path) -> int:
    """分塊寫入上傳內容，並在超過大小上限時中止與刪除暫存檔。"""
    size = 0
    with destination.open("wb") as output:
        while chunk := video.file.read(1024 * 1024):
            size += len(chunk)
            if size > MAX_VIDEO_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail=f"Uploaded video exceeds the configured {MAX_VIDEO_BYTES}-byte limit.",
                )
            output.write(chunk)
    if size == 0:
        raise HTTPException(status_code=400, detail="Uploaded video is empty.")
    return size


def _normalize_messages(raw_messages: Any) -> tuple[list[dict[str, Any]], list[Image.Image]]:
    """驗證對話內容並載入圖片，整理成模型處理器需要的格式。"""
    if not isinstance(raw_messages, list) or not raw_messages:
        raise HTTPException(status_code=400, detail="messages must be a non-empty array.")
    normalized: list[dict[str, Any]] = []
    images: list[Image.Image] = []
    for item in raw_messages:
        if not isinstance(item, dict) or item.get("role") not in {"system", "user", "assistant"}:
            raise HTTPException(status_code=400, detail="Each message needs a supported role and content.")
        content = item.get("content")
        if isinstance(content, str):
            normalized.append({"role": item["role"], "content": content})
            continue
        if not isinstance(content, list):
            raise HTTPException(status_code=400, detail="Message content must be text or an array of text/image parts.")
        parts: list[dict[str, Any]] = []
        for part in content:
            if not isinstance(part, dict):
                raise HTTPException(status_code=400, detail="Each content part must be an object.")
            if part.get("type") == "text" and isinstance(part.get("text"), str):
                parts.append({"type": "text", "text": part["text"]})
            elif part.get("type") == "image_url":
                image_url = part.get("image_url")
                source = image_url.get("url") if isinstance(image_url, dict) else None
                if not isinstance(source, str):
                    raise HTTPException(status_code=400, detail="image_url.url must be a string.")
                images.append(_load_image(source))
                parts.append({"type": "image"})
                if len(images) > MAX_IMAGES_PER_REQUEST:
                    raise HTTPException(status_code=413, detail=f"At most {MAX_IMAGES_PER_REQUEST} image(s) are allowed per request.")
            else:
                raise HTTPException(status_code=400, detail="Only text and image_url content parts are supported.")
        normalized.append({"role": item["role"], "content": parts})
    has_image = any(
        part.get("type") == "image"
        for message in normalized
        for part in message.get("content", []) if isinstance(message.get("content"), list)
    )
    if not has_image:
        raise HTTPException(status_code=400, detail="This initial API accepts image requests; include at least one image_url.")
    return normalized, images


def _prepare_inputs(messages: list[dict[str, Any]], images: list[Image.Image]) -> tuple[Any, int]:
    """套用對話模板、準備模型輸入張量，並計算提示詞 token 數。"""
    prompt = RUNTIME.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = RUNTIME.processor(text=[prompt], images=images, return_tensors="pt")
    input_tokens = int(inputs["input_ids"].shape[-1])
    return _move_inputs_to_model(inputs), input_tokens


def _generation_args(request_body: dict[str, Any]) -> dict[str, Any]:
    """合併模型預設值與請求參數，產生受 API 限制的生成設定。"""
    max_tokens = int(request_body.get("max_tokens", request_body.get("max_completion_tokens", GENERATION_CONFIG["max_tokens_default"])))
    if not 1 <= max_tokens <= MAX_NEW_TOKENS:
        raise HTTPException(status_code=400, detail=f"max_tokens must be between 1 and {MAX_NEW_TOKENS}.")
    temperature = float(request_body.get("temperature", GENERATION_CONFIG["temperature"]))
    top_p = float(request_body.get("top_p", GENERATION_CONFIG["top_p"]))
    if not 0 <= temperature <= 2 or not 0 < top_p <= 1:
        raise HTTPException(status_code=400, detail="temperature must be 0..2 and top_p must be in (0,1].")
    args: dict[str, Any] = {"max_new_tokens": max_tokens, "do_sample": temperature > 0}
    if temperature > 0:
        args.update(temperature=temperature, top_p=top_p)
        top_k = request_body.get("top_k", GENERATION_CONFIG.get("top_k"))
        min_p = request_body.get("min_p", GENERATION_CONFIG.get("min_p"))
        if top_k is not None:
            top_k = int(top_k)
            if top_k < 0:
                raise HTTPException(status_code=400, detail="top_k must be a non-negative integer.")
            args["top_k"] = top_k
        if min_p is not None:
            min_p = float(min_p)
            if not 0 <= min_p <= 1:
                raise HTTPException(status_code=400, detail="min_p must be in [0,1].")
            args["min_p"] = min_p

    repetition_penalty = float(
        request_body.get("repetition_penalty", GENERATION_CONFIG.get("repetition_penalty", 1.0))
    )
    if repetition_penalty <= 0:
        raise HTTPException(status_code=400, detail="repetition_penalty must be greater than zero.")
    args["repetition_penalty"] = repetition_penalty

    no_repeat_ngram_size = int(
        request_body.get("no_repeat_ngram_size", GENERATION_CONFIG.get("no_repeat_ngram_size", 0))
    )
    if no_repeat_ngram_size < 0:
        raise HTTPException(status_code=400, detail="no_repeat_ngram_size must be non-negative.")
    if no_repeat_ngram_size:
        args["no_repeat_ngram_size"] = no_repeat_ngram_size

    num_beams = int(request_body.get("num_beams", GENERATION_CONFIG.get("num_beams", 1)))
    if num_beams < 1:
        raise HTTPException(status_code=400, detail="num_beams must be at least 1.")
    if num_beams > 1:
        args["num_beams"] = num_beams

    args["use_cache"] = bool(GENERATION_CONFIG.get("use_cache", True))
    cache_implementation = GENERATION_CONFIG.get("cache_implementation")
    if cache_implementation:
        args["cache_implementation"] = str(cache_implementation)
    cache_config = GENERATION_CONFIG.get("cache_config")
    if cache_config is not None:
        if not isinstance(cache_config, dict):
            raise RuntimeError("generation.cache_config must be a mapping in the API YAML.")
        args["cache_config"] = cache_config

    advanced_kwargs = GENERATION_CONFIG.get("advanced_kwargs", {})
    if not isinstance(advanced_kwargs, dict):
        raise RuntimeError("generation.advanced_kwargs must be a mapping in the API YAML.")
    reserved_kwargs = {"input_ids", "inputs", "max_new_tokens", "streamer", "return_dict_in_generate"}
    if reserved_kwargs.intersection(advanced_kwargs):
        raise RuntimeError(
            "generation.advanced_kwargs cannot override API-managed generation arguments: "
            + ", ".join(sorted(reserved_kwargs.intersection(advanced_kwargs)))
        )
    args.update(advanced_kwargs)
    return args


def _check_context(prompt_tokens: int, generation_args: dict[str, Any]) -> None:
    if prompt_tokens + generation_args["max_new_tokens"] > MAX_CONTEXT_TOKENS:
        raise HTTPException(status_code=413, detail=f"Prompt plus requested output exceeds {MAX_CONTEXT_TOKENS} tokens.")


def _analyze_live_frames(
    question: str,
    frames: list[Image.Image],
    generation_args: dict[str, Any],
) -> dict[str, Any]:
    """將連續影格結果轉換為共用的即時工作階段格式。"""
    started = time.perf_counter()
    inputs, prompt_tokens, _ = _video_inputs(question, frames, backend="frames")
    _check_context(prompt_tokens, generation_args)
    with torch.inference_mode():
        output = RUNTIME.model.generate(**inputs, **generation_args)
    generated_ids = output[0, inputs["input_ids"].shape[-1] :]
    answer = RUNTIME.processor.batch_decode(generated_ids.unsqueeze(0), skip_special_tokens=True)[0].strip()
    return {
        "answer": answer,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": int(generated_ids.numel()),
        "generation_seconds": round(time.perf_counter() - started, 3),
    }


def _analyze_live_video(
    question: str,
    video_path: Path,
    generation_args: dict[str, Any],
    codec_engine: str | None,
    target_canvas: int,
    max_pixels: int,
    codec_cache_root: Path | None,
) -> dict[str, Any]:
    """對一段已封裝完成的即時視訊執行編碼前處理與文字生成。"""
    started = time.perf_counter()
    inputs, prompt_tokens, canvas_count = _video_inputs(
        question,
        video_path,
        backend="codec",
        codec_engine=codec_engine,
        target_canvas=target_canvas,
        max_pixels=max_pixels,
        codec_cache_root=codec_cache_root,
    )
    _check_context(prompt_tokens, generation_args)
    processor_finished = time.perf_counter()
    with torch.inference_mode():
        output = RUNTIME.model.generate(**inputs, **generation_args)
    generated_ids = output[0, inputs["input_ids"].shape[-1] :]
    answer = RUNTIME.processor.batch_decode(generated_ids.unsqueeze(0), skip_special_tokens=True)[0].strip()
    generated_at = time.perf_counter()
    return {
        "answer": answer,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": int(generated_ids.numel()),
        "processor_seconds": round(processor_finished - started, 3),
        "generation_seconds": round(generated_at - processor_finished, 3),
        "codec_canvas_count": canvas_count,
    }


LIVE_VIDEO = LiveVideoService(
    model_name=MODEL_ID,
    settings=LIVE_SETTINGS,
    inference_gate=RUNTIME.generation_lock,
    analyze_frames=_analyze_live_frames,
    analyze_video=_analyze_live_video,
    build_generation_options=_generation_args,
    transform_frame=_resize_image,
    create_request_model=LiveSessionCreateRequest,
    session_response_model=LiveSessionResponse,
    stop_response_model=LiveSessionStopResponse,
)
RUNTIME.live_video = LIVE_VIDEO


def _response(content: str, prompt_tokens: int, completion_tokens: int, request_id: str, finish_reason: str) -> dict[str, Any]:
    return {
        "id": request_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": MODEL_ID,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": finish_reason}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "total_tokens": prompt_tokens + completion_tokens},
    }


def _generate(request_body: dict[str, Any]) -> dict[str, Any]:
    """執行單次非串流生成；同一時間只允許設定數量內的請求進入模型。"""
    if not RUNTIME.generation_lock.acquire(blocking=False):
        raise HTTPException(status_code=429, detail="The model is busy; retry after the active request completes.")
    try:
        messages, images = _normalize_messages(request_body.get("messages"))
        inputs, prompt_tokens = _prepare_inputs(messages, images)
        generation_args = _generation_args(request_body)
        _check_context(prompt_tokens, generation_args)
        with torch.inference_mode():
            output = RUNTIME.model.generate(**inputs, **generation_args)
        generated_ids = output[0, inputs["input_ids"].shape[-1] :]
        completion_tokens = int(generated_ids.numel())
        answer = RUNTIME.processor.batch_decode(generated_ids.unsqueeze(0), skip_special_tokens=True)[0].strip()
        finish_reason = "length" if completion_tokens >= generation_args["max_new_tokens"] else "stop"
        return _response(answer, prompt_tokens, completion_tokens, f"chatcmpl-{uuid.uuid4().hex}", finish_reason)
    finally:
        RUNTIME.generation_lock.release()


def _stream(request_body: dict[str, Any]) -> Iterator[str]:
    """透過 SSE 逐段傳送模型輸出，並在串流結束時釋放推論名額。"""
    request_id = f"chatcmpl-{uuid.uuid4().hex}"
    if not RUNTIME.generation_lock.acquire(blocking=False):
        yield f"data: {json.dumps({'error': {'message': 'The model is busy.', 'type': 'server_error'}})}\n\n"
        yield "data: [DONE]\n\n"
        return
    thread: threading.Thread | None = None
    try:
        messages, images = _normalize_messages(request_body.get("messages"))
        inputs, prompt_tokens = _prepare_inputs(messages, images)
        generation_args = _generation_args(request_body)
        _check_context(prompt_tokens, generation_args)
        streamer = TextIteratorStreamer(RUNTIME.processor.tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=STREAM_TIMEOUT)
        worker_error: list[BaseException] = []

        def worker() -> None:
            try:
                with torch.inference_mode():
                    RUNTIME.model.generate(**inputs, streamer=streamer, **generation_args)
            except BaseException as exc:
                worker_error.append(exc)
                streamer.end()

        # 生成器會在目前執行緒逐段讀取 streamer；模型生成則放到背景執行緒。
        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        yield f"data: {json.dumps({'id': request_id, 'object': 'chat.completion.chunk', 'created': int(time.time()), 'model': MODEL_ID, 'choices': [{'index': 0, 'delta': {'role': 'assistant'}, 'finish_reason': None}]})}\n\n"
        parts: list[str] = []
        for text in streamer:
            if text:
                parts.append(text)
                chunk = {"id": request_id, "object": "chat.completion.chunk", "created": int(time.time()), "model": MODEL_ID, "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}]}
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
        thread.join()
        if worker_error:
            yield f"data: {json.dumps({'error': {'message': str(worker_error[0]), 'type': 'server_error'}}, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
            return
        answer = "".join(parts).strip()
        completion_tokens = len(RUNTIME.processor.tokenizer.encode(answer, add_special_tokens=False))
        finish_reason = "length" if completion_tokens >= generation_args["max_new_tokens"] else "stop"
        final_chunk = {"id": request_id, "object": "chat.completion.chunk", "created": int(time.time()), "model": MODEL_ID, "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}], "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "total_tokens": prompt_tokens + completion_tokens}}
        yield f"data: {json.dumps(final_chunk)}\n\n"
        yield "data: [DONE]\n\n"
    except HTTPException as exc:
        yield f"data: {json.dumps({'error': {'message': exc.detail, 'type': 'invalid_request_error'}}, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"
    finally:
        if thread is not None and thread.is_alive():
            thread.join()
        RUNTIME.generation_lock.release()


@asynccontextmanager
async def lifespan(_: Any):
    """服務啟動時載入模型，結束時停止即時工作階段並釋放 GPU 資源。"""
    optimization = CONFIG.get("optimization", {})
    pytorch_settings = optimization.get("pytorch", {})
    torch.backends.cuda.matmul.allow_tf32 = bool(pytorch_settings.get("allow_tf32", False))
    torch.backends.cudnn.allow_tf32 = bool(pytorch_settings.get("cudnn_allow_tf32", False))
    torch.backends.cudnn.benchmark = bool(pytorch_settings.get("cudnn_benchmark", False))
    matmul_precision = pytorch_settings.get("float32_matmul_precision")
    if matmul_precision is not None:
        if matmul_precision not in {"highest", "high", "medium"}:
            raise RuntimeError("optimization.pytorch.float32_matmul_precision must be highest, high, or medium.")
        torch.set_float32_matmul_precision(matmul_precision)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; this API is configured for NVIDIA GPU serving.")
    dtype = getattr(torch, MODEL_CONFIG["dtype"])
    kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "device_map": MODEL_CONFIG["device_map"],
        "revision": MODEL_REVISION,
        "dtype": dtype,
        "local_files_only": LOCAL_FILES_ONLY,
    }
    for key in (
        "attn_implementation",
        "experts_implementation",
        "use_kernels",
        "use_safetensors",
        "weights_only",
        "offload_buffers",
        "disable_mmap",
    ):
        if MODEL_CONFIG.get(key) is not None:
            kwargs[key] = MODEL_CONFIG[key]
    kernel_config = MODEL_CONFIG.get("kernel_config")
    if kernel_config is not None:
        if not isinstance(kernel_config, dict):
            raise RuntimeError("model.kernel_config must be a mapping in the API YAML.")
        kwargs["kernel_config"] = KernelConfig(**kernel_config)
    if MODEL_CONFIG.get("offload_folder"):
        offload_folder = Path(os.path.expandvars(os.path.expanduser(MODEL_CONFIG["offload_folder"])))
        if not offload_folder.is_absolute():
            offload_folder = PROJECT_ROOT / offload_folder
        offload_folder.mkdir(parents=True, exist_ok=True)
        kwargs["offload_folder"] = str(offload_folder)
    method = QUANTIZATION_CONFIG["method"]
    if method == "nf4":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=str(QUANTIZATION_CONFIG.get("quant_type", "nf4")),
            bnb_4bit_use_double_quant=bool(QUANTIZATION_CONFIG["double_quant"]),
            bnb_4bit_compute_dtype=getattr(torch, QUANTIZATION_CONFIG["compute_dtype"]),
            bnb_4bit_quant_storage=getattr(torch, QUANTIZATION_CONFIG["quant_storage"])
            if QUANTIZATION_CONFIG.get("quant_storage") else None,
        )
    elif method == "int8":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=bool(QUANTIZATION_CONFIG["llm_int8_enable_fp32_cpu_offload"]),
            llm_int8_threshold=float(QUANTIZATION_CONFIG["llm_int8_threshold"]),
            llm_int8_has_fp16_weight=bool(QUANTIZATION_CONFIG.get("llm_int8_has_fp16_weight", False)),
            llm_int8_skip_modules=QUANTIZATION_CONFIG.get("llm_int8_skip_modules"),
        )
    elif method not in {"none", "prequantized"}:
        raise RuntimeError(f"Unsupported Transformers quantization method in config: {method}")
    max_memory: dict[int | str, str] = {}
    if QUANTIZATION_CONFIG.get("gpu_max_memory"):
        max_memory[0] = str(QUANTIZATION_CONFIG["gpu_max_memory"])
    if QUANTIZATION_CONFIG.get("cpu_max_memory"):
        max_memory["cpu"] = str(QUANTIZATION_CONFIG["cpu_max_memory"])
    if max_memory:
        kwargs["max_memory"] = max_memory
    processor_config = CONFIG.get("processor", {})
    processor_kwargs: dict[str, Any] = {"trust_remote_code": True, "revision": MODEL_REVISION, "local_files_only": LOCAL_FILES_ONLY}
    if processor_config.get("use_fast") is not None:
        processor_kwargs["use_fast"] = bool(processor_config["use_fast"])
    # 先載入處理器，再載入模型；兩者使用相同 revision 與本機快取策略。
    RUNTIME.processor = AutoProcessor.from_pretrained(
        MODEL_LOAD_PATH,
        **processor_kwargs,
    )
    RUNTIME.model = AutoModelForCausalLM.from_pretrained(MODEL_LOAD_PATH, **kwargs).eval()
    compile_settings = optimization.get("torch_compile", {})
    if compile_settings.get("enabled", False):
        RUNTIME.model = torch.compile(
            RUNTIME.model,
            backend=str(compile_settings.get("backend", "inductor")),
            mode=str(compile_settings.get("mode", "default")),
            fullgraph=bool(compile_settings.get("fullgraph", False)),
            dynamic=bool(compile_settings.get("dynamic", True)),
        )
    # API 關閉前先停止串流工作階段，避免背景執行緒繼續使用已卸載的模型。
    yield
    LIVE_VIDEO.stop_all()
    RUNTIME.model = None
    RUNTIME.processor = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class TransformersInferenceService:
    """向 HTTP 路由提供模型推論操作，不在此處處理路由細節。"""

    def __init__(self, runtime: TransformersModelState) -> None:
        self.runtime = runtime

    @property
    def live_video(self) -> LiveVideoService:
        return self.runtime.live_video

    def generate_chat(self, request_body: dict[str, Any]) -> dict[str, Any]:
        return _generate(request_body)

    def stream_chat(self, request_body: dict[str, Any]) -> Iterator[str]:
        return _stream(request_body)

    def generation_options(self, request_body: dict[str, Any]) -> dict[str, Any]:
        return _generation_args(request_body)

    def health(self) -> dict[str, Any]:
        ready = self.runtime.model is not None
        data: dict[str, Any] = {
            "status": "ok" if ready else "starting",
            "backend": "transformers",
            "model": MODEL_ID,
            "revision": MODEL_REVISION,
            "quantization": QUANTIZATION_CONFIG["method"],
            "live_sessions": self.live_video.active_count(),
            "live_video_enabled": True,
        }
        if ready and torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            data.update(
                gpu=torch.cuda.get_device_name(0),
                vram_free_bytes=free,
                vram_total_bytes=total,
            )
        return data

    def model_list(self) -> dict[str, Any]:
        return {"object": "list", "data": [{"id": MODEL_ID, "object": "model", "created": 0, "owned_by": "local"}]}

    def understand_video(self, video: UploadFile, request: VideoUnderstandingRequest) -> dict[str, Any]:
        """以影格取樣或模型編碼後端分析上傳的視訊。"""
        question = request.question.strip()
        if not question:
            raise HTTPException(status_code=400, detail="question must not be empty.")
        video_backend = request.video_backend or VIDEO_BACKEND
        codec_engine = request.codec_engine or CODEC_ENGINE
        request_body = request.model_dump(exclude_none=True, exclude={"question", "video_backend", "codec_engine"})
        generation_args = _generation_args(request_body)
        if not self.runtime.generation_lock.acquire(blocking=False):
            raise HTTPException(status_code=429, detail="The model is busy; retry after the active request completes.")

        request_started = time.perf_counter()
        try:
            torch.cuda.reset_peak_memory_stats()
            with tempfile.TemporaryDirectory(prefix="magevl4b-video-upload-") as temp_dir:
                destination = _uploaded_video_path(video.filename, temp_dir)
                upload_started = time.perf_counter()
                upload_bytes = _save_upload(video, destination)
                if video_backend == "frames":
                    video_input, video_metadata = _sample_video_frames(destination)
                    video_metadata["backend"] = "frames"
                    video_metadata["codec_engine"] = None
                elif video_backend == "codec":
                    video_metadata = _probe_video_metadata(destination)
                    video_metadata["backend"] = "codec"
                    video_metadata["codec_engine"] = codec_engine or CODEC_ENGINE
                    video_input = destination
                else:
                    raise HTTPException(status_code=400, detail=f"Unsupported video_backend: {video_backend}")
                decode_seconds = time.perf_counter() - upload_started
                processor_started = time.perf_counter()
                inputs, prompt_tokens, canvas_count = _video_inputs(
                    question,
                    video_input,
                    backend=video_backend,
                    codec_engine=codec_engine,
                    target_canvas=CODEC_TARGET_CANVAS,
                    max_pixels=CODEC_MAX_PIXELS,
                    codec_cache_root=Path(temp_dir) / "codec-cache" if video_backend == "codec" else None,
                )
                processor_seconds = time.perf_counter() - processor_started
                if video_backend == "codec":
                    video_metadata["codec_canvas_count"] = canvas_count
                _check_context(prompt_tokens, generation_args)
                generation_started = time.perf_counter()
                with torch.inference_mode():
                    output = self.runtime.model.generate(**inputs, **generation_args)
                generation_seconds = time.perf_counter() - generation_started
                generated_ids = output[0, inputs["input_ids"].shape[-1] :]
                completion_tokens = int(generated_ids.numel())
                answer = self.runtime.processor.batch_decode(generated_ids.unsqueeze(0), skip_special_tokens=True)[0].strip()
                finish_reason = "length" if completion_tokens >= generation_args["max_new_tokens"] else "stop"
                response = _response(answer, prompt_tokens, completion_tokens, f"videocmpl-{uuid.uuid4().hex}", finish_reason)
                response["object"] = "magevl4b.video.understanding"
                response["video"] = {**video_metadata, "upload_bytes": upload_bytes}
                response["timing_seconds"] = {
                    "staging_and_decode": round(decode_seconds, 4),
                    "processor": round(processor_seconds, 4),
                    "generation": round(generation_seconds, 4),
                    "server_total": round(time.perf_counter() - request_started, 4),
                }
                response["gpu_memory_bytes"] = {
                    "peak_allocated": torch.cuda.max_memory_allocated(),
                    "peak_reserved": torch.cuda.max_memory_reserved(),
                }
                return response
        finally:
            video.file.close()
            self.runtime.generation_lock.release()


INFERENCE_SERVICE = TransformersInferenceService(RUNTIME)


