"""Engine Core：協調推論請求、Token Scheduler、KV 預留與 Worker 執行。"""

from __future__ import annotations

import tempfile
import queue
import threading
import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO

from engine.errors import InferenceError
from engine.kv_cache import KVCacheManager
from engine.scheduler import SchedulerFullError, TokenScheduler
from engine.worker import ModelWorker
from media.video import save_upload, uploaded_video_path


@dataclass(frozen=True)
class InferenceResult:
    """不依賴 HTTP 協定的單次推論結果。"""

    content: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class InferenceStreamEvent:
    """不依賴 SSE／OpenAI schema 的生成事件。"""

    kind: str
    text: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    finish_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class _GenerationContext:
    """由 Engine Core 排程、並將結果送至單一 API stream 的請求狀態。"""

    request_id: str
    finish_operation: str
    events: queue.Queue[Any] = field(default_factory=queue.Queue)
    terminal: bool = False
    cancelled: bool = False


class InferenceEngine:
    """在 API Server 與模型 Worker 之間執行排程、快取預留及結果協調。"""

    def __init__(
        self,
        worker: ModelWorker,
        *,
        model_id: str,
        max_concurrent_requests: int,
        max_waiting_requests: int = 16,
        max_num_scheduled_tokens: int = 8192,
        batch_wait_seconds: float = 0.01,
        kv_cache_block_size_tokens: int = 16,
        kv_cache_capacity_tokens: int = 32768,
        max_video_bytes: int = 209715200,
    ) -> None:
        self.worker = worker
        self.model_id = model_id
        self.scheduler = TokenScheduler(
            max_num_seqs=max_concurrent_requests,
            max_waiting_requests=max_waiting_requests,
            max_num_scheduled_tokens=max_num_scheduled_tokens,
            batch_wait_seconds=batch_wait_seconds,
        )
        self.kv_cache = KVCacheManager(
            block_size_tokens=kv_cache_block_size_tokens,
            capacity_tokens=kv_cache_capacity_tokens,
        )
        self.max_video_bytes = max_video_bytes
        self._state = "new"
        self._state_lock = threading.Lock()
        self._live_sessions: Any = None
        self._contexts: dict[str, _GenerationContext] = {}
        self._contexts_lock = threading.RLock()
        self._registered_requests: set[str] = set()
        self._cancelled_requests: set[str] = set()
        self._dispatcher_stop = threading.Event()
        self._scheduler_wakeup = threading.Event()
        self._dispatcher: threading.Thread | None = None

    @property
    def state(self) -> str:
        with self._state_lock:
            return self._state

    @property
    def ready(self) -> bool:
        return self.state == "ready"

    def start(self) -> None:
        with self._state_lock:
            if self._state == "ready":
                return
            if self._state in {"loading", "closing", "closed"}:
                raise RuntimeError(f"Cannot start inference engine in state {self._state}.")
            self._state = "loading"
        try:
            self.worker.start()
            profile = self.worker.health_info().get("kv_cache_profile", {})
            self.kv_cache.set_bytes_per_token(int(profile.get("bytes_per_token", 0)))
        except BaseException:
            with self._state_lock:
                self._state = "failed"
            raise
        with self._state_lock:
            self._state = "ready"
        self._dispatcher_stop.clear()
        self._dispatcher = threading.Thread(
            target=self._dispatch_batches, name="inference-engine-core", daemon=True
        )
        self._dispatcher.start()

    def close(self) -> None:
        with self._state_lock:
            if self._state == "closed":
                return
            self._state = "closing"
        if self._live_sessions is not None:
            self._live_sessions.stop_all()
        self.scheduler.wait_idle()
        self._dispatcher_stop.set()
        self._scheduler_wakeup.set()
        if self._dispatcher is not None:
            self._dispatcher.join(timeout=30)
        try:
            self.worker.close()
        finally:
            with self._state_lock:
                self._state = "closed"

    def attach_live_sessions(self, manager: Any) -> None:
        """將 Live manager 納入 Engine 生命週期。"""
        if self._live_sessions is not None:
            raise RuntimeError("A live session manager is already attached.")
        self._live_sessions = manager

    def _require_ready(self) -> None:
        if not self.ready:
            raise InferenceError(503, "The inference engine is not ready.")

    def _run_generation(
        self,
        prepare_operation: str,
        prepare_args: tuple[Any, ...],
        finish_operation: str,
        *,
        on_admitted: Any = None,
        request_id: str | None = None,
    ) -> Iterator[InferenceStreamEvent]:
        """準備並登記請求；Engine Core 背景迴圈負責排程、執行及逐事件回傳。"""
        self._require_ready()
        request_id = request_id or uuid.uuid4().hex
        prepared = False
        lease_created = False
        context: _GenerationContext | None = None
        try:
            if self._request_was_cancelled(request_id):
                raise GeneratorExit
            # prepare 可能在內部建立部分狀態後失敗；之後一律執行冪等取消清理。
            prepared = True
            descriptor = self.worker.invoke(prepare_operation, request_id, *prepare_args)
            if self._request_was_cancelled(request_id):
                raise GeneratorExit
            prompt_tokens = int(descriptor["prompt_tokens"])
            max_output_tokens = int(descriptor["max_new_tokens"])
            lease = self.kv_cache.reserve(
                request_id, prompt_tokens=prompt_tokens, max_output_tokens=max_output_tokens
            )
            if lease is None:
                raise InferenceError(429, "The KV cache token capacity is exhausted; retry after active requests finish.")
            lease_created = True
            self.worker.invoke("bind_generation_cache", request_id, lease.block_ids)
            if self._request_was_cancelled(request_id):
                raise GeneratorExit
            context = _GenerationContext(request_id=request_id, finish_operation=finish_operation)
            with self._contexts_lock:
                self._contexts[request_id] = context
            try:
                self.scheduler.add_request(
                    request_id, prompt_tokens=prompt_tokens, max_output_tokens=max_output_tokens
                )
            except SchedulerFullError as exc:
                raise InferenceError(429, "The inference scheduler is full; retry after active requests finish.") from exc
            except ValueError as exc:
                raise InferenceError(413, "The prompt exceeds the scheduler's per-turn token budget.") from exc
            with self._contexts_lock:
                cancelled_before_schedule = context.cancelled
            if cancelled_before_schedule:
                self.scheduler.cancel_request(request_id)
                raise GeneratorExit
            if on_admitted is not None:
                on_admitted()
            context.events.put(InferenceStreamEvent("start", prompt_tokens=prompt_tokens))
            self._scheduler_wakeup.set()
            while True:
                event = context.events.get()
                if isinstance(event, BaseException):
                    raise event
                yield event
                if event.kind == "finish":
                    return
        finally:
            if context is not None:
                self._cancel_context(context)
            else:
                if lease_created:
                    self.kv_cache.release(request_id)
                if prepared:
                    self._cancel_worker_generation(request_id)

    def _dispatch_batches(self) -> None:
        """集中執行 Scheduler 批次，並把每個結果送回對應的請求 stream。"""
        while not self._dispatcher_stop.is_set():
            batch = self.scheduler.schedule_batch()
            if batch is None:
                self._scheduler_wakeup.wait(timeout=0.1)
                self._scheduler_wakeup.clear()
                continue
            request_ids = {item.request_id for item in batch.work_items}
            try:
                results = tuple(self.worker.execute_batch(batch.work_items))
                self.scheduler.complete_batch(results)
            except BaseException as exc:
                self.scheduler.abort_batch(request_ids)
                for request_id in request_ids:
                    context = self._get_context(request_id)
                    if context is not None:
                        self._fail_context(context, exc)
                continue

            for result in results:
                context = self._get_context(result.request_id)
                if context is None or context.terminal or context.cancelled:
                    continue
                if result.text_delta:
                    context.events.put(InferenceStreamEvent("text", text=result.text_delta))
                if result.finished:
                    try:
                        final = self.worker.invoke(context.finish_operation, context.request_id)
                        self._finish_context(context, final)
                    except BaseException as exc:
                        self._fail_context(context, exc)
            self._scheduler_wakeup.set()

    def _get_context(self, request_id: str) -> _GenerationContext | None:
        with self._contexts_lock:
            return self._contexts.get(request_id)

    def cancel_request(self, request_id: str) -> None:
        """取消指定請求並喚醒其事件 iterator，供 EngineClient IPC 中斷使用。"""
        with self._contexts_lock:
            context = self._contexts.get(request_id)
            if context is None:
                if request_id in self._registered_requests:
                    self._cancelled_requests.add(request_id)
                return
        self._cancel_context(context)
        context.events.put(GeneratorExit())

    def register_request(self, request_id: str) -> None:
        """在 Engine Core 接收 IPC 後、啟動 handler 前登記可取消的工作。"""
        if not request_id:
            raise ValueError("request_id is required.")
        with self._contexts_lock:
            if request_id in self._registered_requests or request_id in self._contexts:
                raise ValueError(f"Duplicate active inference request: {request_id}")
            self._registered_requests.add(request_id)

    def unregister_request(self, request_id: str) -> None:
        """清除 IPC request 註冊及未消耗的提前取消旗標。"""
        with self._contexts_lock:
            self._registered_requests.discard(request_id)
            self._cancelled_requests.discard(request_id)

    def _request_was_cancelled(self, request_id: str) -> bool:
        with self._contexts_lock:
            return request_id in self._cancelled_requests

    def _finish_context(self, context: _GenerationContext, result: InferenceResult) -> None:
        with self._contexts_lock:
            if context.terminal or context.cancelled:
                return
            context.terminal = True
            self._contexts.pop(context.request_id, None)
        self.kv_cache.release(context.request_id)
        context.events.put(
            InferenceStreamEvent(
                "finish",
                text=result.content,
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.completion_tokens,
                finish_reason=result.finish_reason,
                metadata=result.metadata,
            )
        )

    def _fail_context(self, context: _GenerationContext, error: BaseException) -> None:
        with self._contexts_lock:
            if context.terminal:
                return
            context.terminal = True
            self._contexts.pop(context.request_id, None)
        self.scheduler.cancel_request(context.request_id)
        self.kv_cache.release(context.request_id)
        self._cancel_worker_generation(context.request_id)
        context.events.put(error)

    def _cancel_context(self, context: _GenerationContext) -> None:
        with self._contexts_lock:
            if context.terminal:
                return
            context.terminal = True
            context.cancelled = True
            self._contexts.pop(context.request_id, None)
        self.scheduler.cancel_request(context.request_id)
        self.kv_cache.release(context.request_id)
        self._cancel_worker_generation(context.request_id)

    def _cancel_worker_generation(self, request_id: str) -> None:
        try:
            self.worker.invoke("cancel_generation", request_id)
        except BaseException:
            pass

    def generate_chat(
        self, request_body: dict[str, Any], *, request_id: str | None = None
    ) -> InferenceResult:
        result = self._collect_result(
            self._run_generation(
                "prepare_chat_generation", (request_body,), "finish_chat_generation", request_id=request_id
            )
        )
        return result

    def stream_chat(
        self, request_body: dict[str, Any], *, request_id: str | None = None
    ) -> Iterator[InferenceStreamEvent]:
        yield from self._run_generation(
            "prepare_chat_generation", (request_body,), "finish_chat_generation", request_id=request_id
        )

    def understand_video(
        self, upload_stream: BinaryIO, filename: str | None, request_body: dict[str, Any]
    ) -> InferenceResult:
        with tempfile.TemporaryDirectory(prefix="inference-video-upload-") as temp_dir:
            destination = uploaded_video_path(filename, temp_dir)
            started = time.perf_counter()
            upload_bytes = save_upload(upload_stream, destination, self.max_video_bytes)
            staging_seconds = time.perf_counter() - started
            return self.understand_video_path(
                str(destination), filename, request_body, upload_bytes, staging_seconds
            )

    def understand_video_path(
        self,
        video_path: str,
        filename: str | None,
        request_body: dict[str, Any],
        upload_bytes: int,
        staging_seconds: float = 0.0,
        *,
        request_id: str | None = None,
    ) -> InferenceResult:
        """執行 API Server 已安全暫存的影片，不把 bytes 跨 Engine IPC 傳輸。"""
        return self._collect_result(self._run_generation(
            "prepare_video_generation",
            (video_path, filename, request_body, upload_bytes, staging_seconds),
            "finish_video_generation",
            request_id=request_id,
        ))

    @staticmethod
    def _collect_result(events: Iterator[InferenceStreamEvent]) -> InferenceResult:
        answer: list[str] = []
        final: InferenceStreamEvent | None = None
        for event in events:
            if event.kind == "text" and event.text:
                answer.append(event.text)
            elif event.kind == "finish":
                final = event
        if final is None:
            raise RuntimeError("Model Worker completed without a finish event.")
        return InferenceResult(
            content=final.text if final.text is not None else "".join(answer).strip(),
            prompt_tokens=int(final.prompt_tokens or 0),
            completion_tokens=int(final.completion_tokens or 0),
            finish_reason=final.finish_reason or "stop",
            metadata=final.metadata,
        )

    def generation_options(self, request_body: dict[str, Any]) -> dict[str, Any]:
        self._require_ready()
        return self.worker.invoke("generation_options", request_body)

    def validate_live_video_backend(self, backend: str, codec_engine: str | None) -> None:
        self._require_ready()
        self.worker.invoke("validate_live_video_backend", backend, codec_engine)

    def try_live_inference(
        self, operation: str, *args: Any, on_admitted: Any = None, request_id: str | None = None
    ) -> tuple[bool, Any]:
        """讓 Live 分析共用公平 Token Scheduler 與 KV block 預留。"""
        self._require_ready()
        prepare_operations = {
            "analyze_live_frames": "prepare_live_frames_generation",
            "analyze_live_video": "prepare_live_video_generation",
        }
        prepare_operation = prepare_operations.get(operation)
        if prepare_operation is None:
            raise ValueError(f"Unsupported live inference operation: {operation}")
        try:
            result = self._collect_result(self._run_generation(
                prepare_operation,
                args,
                "finish_live_generation",
                on_admitted=on_admitted,
                request_id=request_id,
            ))
        except InferenceError as exc:
            if exc.status_code == 429:
                return False, None
            raise
        return True, {
            "answer": result.content,
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            **result.metadata,
        }

    def health(self) -> dict[str, Any]:
        data = self.worker.health_info()
        data.update(
            status="ok" if self.ready else "starting",
            live_sessions=self.live_session_count(),
            scheduler=self.scheduler.snapshot(),
            kv_cache=self.kv_cache.snapshot(),
            worker_pid=getattr(self.worker, "pid", None),
        )
        if self.ready:
            data.update(self.worker.gpu_status())
        return data

    def live_settings(self) -> Any:
        return self._require_live_sessions().settings

    def live_session_count(self) -> int:
        if self._live_sessions is None:
            return 0
        return self._live_sessions.active_count()

    def start_live_session(self, **kwargs: Any) -> Any:
        return self._require_live_sessions().start_session(**kwargs)

    def get_live_session(self, session_id: str) -> Any:
        return self._require_live_sessions().get_session(session_id)

    def stop_live_session(self, session_id: str) -> None:
        self.get_live_session(session_id).request_stop()

    def _require_live_sessions(self) -> Any:
        if self._live_sessions is None:
            raise InferenceError(503, "Live video sessions are not configured.")
        return self._live_sessions
