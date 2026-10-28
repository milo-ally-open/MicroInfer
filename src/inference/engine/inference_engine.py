"""协调请求生命周期、调度、KV block 与 Worker 执行。"""

from __future__ import annotations

import logging
import queue
import tempfile
import threading
import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO

from inference.engine.block_manager import BlockManager
from inference.engine.scheduler import SchedulerFullError, TokenScheduler
from inference.engine.sequence import (
    InferenceError, InferenceResult, InferenceStreamEvent, PreparedChatInput, WorkerResult,
)
from inference.processors.video import save_upload, uploaded_video_path


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EngineOutputEvent:
    """不携带 HTTP 格式的单笔输出增量。"""

    kind: str
    text: str | None = None
    token_id: int | None = None


class EngineOutputProcessor:
    """保留 request ID 对应并统一 Token 与文字 Runner 的输出形状。"""

    def process(self, result: WorkerResult) -> tuple[EngineOutputEvent, ...]:
        if result.token_id is not None and result.text_delta:
            raise ValueError("A Worker result cannot contain both token_id and text_delta.")
        if result.token_id is not None:
            return (EngineOutputEvent("token", token_id=int(result.token_id)),)
        if result.text_delta:
            return (EngineOutputEvent("text", text=result.text_delta),)
        return ()



@dataclass
class _GenerationContext:
    """由 Engine Core 调度、并将结果送至单一 API stream 的请求状态。"""

    request_id: str
    finish_operation: str
    # 单一请求的输出量受 max_new_tokens 限制；Queue 保留跨 Thread 的事件顺序。
    events: queue.Queue[Any] = field(default_factory=queue.Queue)
    terminal: bool = False
    cancelled: bool = False


class InferenceEngine:
    """在 API Server 与模型 Worker 之间执行调度、缓存预留及结果协调。"""

    def __init__(
        self,
        executor: Any,
        *,
        model_id: str,
        max_concurrent_requests: int,
        max_waiting_requests: int = 16,
        max_num_scheduled_tokens: int = 8192,
        prefill_chunk_size_tokens: int | None = None,
        batch_wait_seconds: float = 0.01,
        kv_cache_block_size_tokens: int = 16,
        kv_cache_capacity_tokens: int = 32768,
        max_video_bytes: int = 209715200,
    ) -> None:
        self.executor = executor
        self.output_processor = EngineOutputProcessor()
        self.model_id = model_id
        self.scheduler = TokenScheduler(
            max_num_seqs=max_concurrent_requests,
            max_waiting_requests=max_waiting_requests,
            max_num_scheduled_tokens=max_num_scheduled_tokens,
            batch_wait_seconds=batch_wait_seconds,
            prefill_chunk_size_tokens=prefill_chunk_size_tokens,
        )
        self.block_manager = BlockManager(
            block_size_tokens=kv_cache_block_size_tokens,
            capacity_tokens=kv_cache_capacity_tokens,
        )
        # 保留旧健康信息/内部调用名称；资源所有权由 BlockManager 统一管理。
        self.kv_cache = self.block_manager
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
            self.executor.start()
            profile = self.executor.health_info().get("kv_cache_profile", {})
            if "physical_blocks" in profile:
                allocation = self.kv_cache.snapshot()
                if int(profile["physical_blocks"]) != allocation["capacity_blocks"]:
                    raise RuntimeError("Engine KV block count does not match the Worker physical page pool.")
                if int(profile.get("block_size_tokens", 0)) != self.kv_cache.block_size_tokens:
                    raise RuntimeError("Engine KV block size does not match the Worker physical page pool.")
            self.kv_cache.set_bytes_per_token(int(profile.get("bytes_per_token", 0)))
        except BaseException:
            try:
                self.executor.close()
            except BaseException:
                logger.exception("Could not close the Worker after Engine startup failed")
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
            if self._state in {"closing", "closed"}:
                return
            self._state = "closing"
        try:
            if self._live_sessions is not None:
                self._live_sessions.stop_all()
            self.scheduler.wait_idle()
        finally:
            try:
                # 即使停止 Live 或等待请求时出错，也要唤醒背景回圈并释放 Worker。
                self._dispatcher_stop.set()
                self._scheduler_wakeup.set()
                if self._dispatcher is not None:
                    self._dispatcher.join(timeout=30)
                    if self._dispatcher.is_alive():
                        logger.error("Inference Engine dispatcher did not stop within 30 seconds")
                self.executor.close()
            finally:
                with self._state_lock:
                    self._state = "closed"

    def attach_live_sessions(self, manager: Any) -> None:
        """将 Live manager 纳入 Engine 生命周期。"""
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
        """准备并登记请求；Engine Core 背景回圈负责调度、执行及逐事件返回。"""
        self._require_ready()
        request_id = request_id or uuid.uuid4().hex
        prepared = False
        lease_created = False
        context: _GenerationContext | None = None
        try:
            if self._request_was_cancelled(request_id):
                raise GeneratorExit
            # prepare 可能在内部创建部分状态后失败；之后一律执行幂等取消清理。
            prepared = True
            descriptor = self.executor.invoke(prepare_operation, request_id, *prepare_args)
            if self._request_was_cancelled(request_id):
                raise GeneratorExit
            prompt_tokens = int(descriptor["prompt_tokens"])
            max_output_tokens = int(descriptor["max_new_tokens"])
            lease = self.kv_cache.reserve(
                request_id,
                prompt_tokens=prompt_tokens,
                max_output_tokens=max_output_tokens,
            )
            if lease is None:
                raise InferenceError(429, "The KV cache token capacity is exhausted; retry after active requests finish.")
            lease_created = True
            self.executor.invoke("bind_generation_cache", request_id, lease.block_ids)
            if self._request_was_cancelled(request_id):
                raise GeneratorExit
            context = _GenerationContext(request_id=request_id, finish_operation=finish_operation)
            with self._contexts_lock:
                self._contexts[request_id] = context
            try:
                self.scheduler.add_request(
                    request_id,
                    prompt_tokens=prompt_tokens,
                    max_output_tokens=max_output_tokens,
                    prefill_min_chunk_tokens=int(descriptor.get("prefill_min_chunk_tokens", 1)),
                    cache_block_ids=lease.block_ids,
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
                try:
                    event = context.events.get(timeout=0.5)
                except queue.Empty:
                    # 避免 Engine Core 背景线程意外退出后，API stream 永久等待。
                    if self._dispatcher is None or not self._dispatcher.is_alive():
                        raise RuntimeError("Inference Engine dispatcher stopped before completing the request.")
                    continue
                if isinstance(event, BaseException):
                    raise event
                yield event
                if event.kind == "finish":
                    return
        finally:
            if context is not None:
                self._cancel_context(context)
            else:
                if prepared:
                    self._cancel_worker_generation(request_id)
                if lease_created:
                    self.kv_cache.release(request_id)

    def _dispatch_batches(self) -> None:
        """集中执行 Scheduler 批次，并把每个结果送回对应的请求 stream。"""
        while not self._dispatcher_stop.is_set():
            batch = self.scheduler.schedule_batch(self.block_manager)
            if batch is None:
                self._scheduler_wakeup.wait(timeout=0.1)
                self._scheduler_wakeup.clear()
                continue
            request_ids = {item.request_id for item in batch.work_items}
            try:
                # Scheduler 只有在 Worker 返回完整批次后才提交状态更新。
                results = tuple(self.executor.execute_batch(batch.work_items))
                self.scheduler.complete_batch(results)
            except Exception as exc:
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
                try:
                    for output in self.output_processor.process(result):
                        context.events.put(
                            InferenceStreamEvent(
                                output.kind,
                                text=output.text,
                                token_id=output.token_id,
                            )
                        )
                    if result.finished:
                        final = self.executor.invoke(context.finish_operation, context.request_id)
                        self._finish_context(context, final)
                except Exception as exc:
                    logger.exception("Could not publish Worker result for request %s", result.request_id)
                    self._fail_context(context, exc)
            self._scheduler_wakeup.set()

    def _get_context(self, request_id: str) -> _GenerationContext | None:
        with self._contexts_lock:
            return self._contexts.get(request_id)

    def cancel_request(self, request_id: str) -> None:
        """取消指定请求并唤醒其事件 iterator，供 EngineClient IPC 中断使用。"""
        with self._contexts_lock:
            context = self._contexts.get(request_id)
            if context is None:
                if request_id in self._registered_requests:
                    self._cancelled_requests.add(request_id)
                return
        self._cancel_context(context)
        context.events.put(GeneratorExit())

    def register_request(self, request_id: str) -> None:
        """在 Engine Core 接收 IPC 后、启动 handler 前登记可取消的工作。"""
        if not request_id:
            raise ValueError("request_id is required.")
        with self._contexts_lock:
            if request_id in self._registered_requests or request_id in self._contexts:
                raise ValueError(f"Duplicate active inference request: {request_id}")
            self._registered_requests.add(request_id)

    def unregister_request(self, request_id: str) -> None:
        """清除 IPC request 注册及未消耗的提前取消旗标。"""
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
                token_ids=result.token_ids,
            )
        )

    def _fail_context(self, context: _GenerationContext, error: BaseException) -> None:
        with self._contexts_lock:
            if context.terminal:
                return
            context.terminal = True
            self._contexts.pop(context.request_id, None)
        self.scheduler.cancel_request(context.request_id)
        self._cancel_worker_generation(context.request_id)
        self.kv_cache.release(context.request_id)
        context.events.put(error)

    def _cancel_context(self, context: _GenerationContext) -> None:
        with self._contexts_lock:
            if context.terminal:
                return
            context.terminal = True
            context.cancelled = True
            self._contexts.pop(context.request_id, None)
        self.scheduler.cancel_request(context.request_id)
        self._cancel_worker_generation(context.request_id)
        self.kv_cache.release(context.request_id)

    def _cancel_worker_generation(self, request_id: str) -> None:
        try:
            self.executor.invoke("cancel_generation", request_id)
        except BaseException:
            logger.exception("Could not cancel Worker generation %s", request_id)

    def generate_chat(
        self,
        request_body: dict[str, Any],
        prepared_input: PreparedChatInput | None = None,
        *,
        request_id: str | None = None,
    ) -> InferenceResult:
        prepare_args = (request_body,) if prepared_input is None else (request_body, prepared_input)
        result = self._collect_result(
            self._run_generation(
                "prepare_chat_generation", prepare_args, "finish_chat_generation", request_id=request_id
            )
        )
        return result

    def stream_chat(
        self,
        request_body: dict[str, Any],
        prepared_input: PreparedChatInput | None = None,
        *,
        request_id: str | None = None,
    ) -> Iterator[InferenceStreamEvent]:
        prepare_args = (request_body,) if prepared_input is None else (request_body, prepared_input)
        yield from self._run_generation(
            "prepare_chat_generation", prepare_args, "finish_chat_generation", request_id=request_id
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
        """执行 API Server 已安全暂存的视频，不把 bytes 跨 Engine IPC 传输。"""
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
            token_ids=final.token_ids,
        )

    def generation_options(self, request_body: dict[str, Any]) -> dict[str, Any]:
        self._require_ready()
        return self.executor.invoke("generation_options", request_body)

    def validate_live_video_backend(self, backend: str, codec_engine: str | None) -> None:
        self._require_ready()
        self.executor.invoke("validate_live_video_backend", backend, codec_engine)

    def try_live_inference(
        self, operation: str, *args: Any, on_admitted: Any = None, request_id: str | None = None
    ) -> tuple[bool, Any]:
        """让 Live 分析共用公平 Token Scheduler 与 KV block 预留。"""
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
        data = self.executor.health_info()
        data.update(
            status="ok" if self.ready else "starting",
            live_sessions=self.live_session_count(),
            scheduler=self.scheduler.snapshot(),
            kv_cache=self.kv_cache.snapshot(),
            worker_pid=getattr(self.executor, "pid", None),
        )
        if self.ready:
            data.update(self.executor.gpu_status())
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
