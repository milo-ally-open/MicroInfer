"""管理即時視訊來源、工作階段事件與背景工作執行緒。"""

from __future__ import annotations

import csv
import os
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal

from PIL import Image

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

LiveInferenceDispatcher = Callable[..., tuple[bool, dict[str, Any] | None]]
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
        try_inference: LiveInferenceDispatcher,
        transform_frame: FrameTransformer,
    ) -> None:
        self.session_id = session_id
        self.question = question
        self.generation_options = generation_options
        self.settings = settings
        self.try_inference = try_inference
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
                try:
                    if segment is not None:
                        segment_path, segment_start, segment_end = segment
                        admitted, result = self.try_inference(
                            "analyze_live_video",
                            self.question,
                            segment_path,
                            self.generation_options,
                            self.settings.codec_engine,
                            self.settings.live_codec_target_canvas,
                            self.settings.codec_max_pixels,
                            self.codec_workdir / "codec-cache" if self.codec_workdir is not None else None,
                            on_admitted=lambda: self._set_status("analyzing"),
                        )
                        if not admitted:
                            last_busy_event_at = self._handle_busy_model(last_busy_event_at, segment)
                            continue
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
                        admitted, result = self.try_inference(
                            "analyze_live_frames", self.question, images, self.generation_options,
                            on_admitted=lambda: self._set_status("analyzing"),
                        )
                        if not admitted:
                            last_busy_event_at = self._handle_busy_model(last_busy_event_at, None)
                            continue
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

    def _handle_busy_model(self, last_busy_event_at: float, segment: tuple[Path, float, float] | None) -> float:
        """保留已封裝片段並通知用戶端等待共用 Engine 名額。"""
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
        return last_busy_event_at


class LiveSessionLimitError(RuntimeError):
    """作用中的即時工作階段已達設定上限。"""


class LiveSessionNotFoundError(LookupError):
    """找不到指定的即時工作階段。"""


class LiveSessionManager:
    """保存即時工作階段並管理其建立、查詢與停止。"""

    def __init__(self, *, settings: LiveVideoSettings, try_inference: LiveInferenceDispatcher, transform_frame: FrameTransformer) -> None:
        self.settings = settings
        self.try_inference = try_inference
        self.transform_frame = transform_frame
        self._sessions: dict[str, LiveVideoSession] = {}
        self._sessions_lock = threading.Lock()

    def active_count(self) -> int:
        with self._sessions_lock:
            return sum(session.is_active() for session in self._sessions.values())

    def stop_all(self) -> None:
        with self._sessions_lock:
            sessions = list(self._sessions.values())
        for session in sessions:
            session.request_stop()
        for session in sessions:
            for worker in (session.capture_thread, session.inference_thread):
                if worker.is_alive():
                    worker.join()

    def _prune_finished(self) -> None:
        expired_before = time.monotonic() - 900
        with self._sessions_lock:
            expired = [sid for sid, session in self._sessions.items() if session.finished_at is not None and session.finished_at < expired_before]
            for session_id in expired:
                del self._sessions[session_id]

    def start_session(self, *, session_id: str, question: str, generation_options: dict[str, Any], settings: LiveVideoSettings) -> LiveVideoSession:
        self._prune_finished()
        with self._sessions_lock:
            active_count = sum(session.is_active() for session in self._sessions.values())
            if active_count >= settings.max_sessions:
                raise LiveSessionLimitError(f"At most {settings.max_sessions} live session(s) may run at once.")
            session = LiveVideoSession(
                session_id=session_id, question=question, generation_options=generation_options,
                settings=settings, try_inference=self.try_inference, transform_frame=self.transform_frame,
            )
            self._sessions[session_id] = session
        session.start()
        return session

    def get_session(self, session_id: str) -> LiveVideoSession:
        with self._sessions_lock:
            session = self._sessions.get(session_id)
        if session is None:
            raise LiveSessionNotFoundError(session_id)
        return session
