"""提供 API Server 到獨立 Engine Core 的本機 IPC client。"""

from __future__ import annotations

import importlib
import multiprocessing
import os
import queue
import tempfile
import threading
import time
import uuid
from collections.abc import Iterator
from typing import Any

from engine.contracts import PreparedChatInput
from engine.errors import InferenceError
from media.video import save_upload, uploaded_video_path


class EngineClientError(RuntimeError):
    """Engine Core IPC 或程序生命週期錯誤。"""


class EngineClientStream:
    """可跨執行緒取消的 EngineClient event iterator。"""

    def __init__(
        self,
        client: EngineClient,
        request_body: dict[str, Any],
        prepared_input: PreparedChatInput | None = None,
    ) -> None:
        self._client = client
        self._request_id = uuid.uuid4().hex
        self._pending = client._register(self._request_id)
        self._deadline = time.monotonic() + client.request_timeout_seconds
        self._state_lock = threading.Lock()
        self._cancel_sent = False
        self._finished = False
        try:
            client._send({
                "kind": "stream",
                "request_id": self._request_id,
                "operation": "stream_chat",
                "args": (request_body,) if prepared_input is None else (request_body, prepared_input),
            })
        except BaseException:
            self._finish()
            raise

    def __iter__(self) -> EngineClientStream:
        return self

    def __next__(self) -> Any:
        with self._state_lock:
            if self._finished:
                raise StopIteration
        try:
            message = self._pending.get(timeout=max(0.01, self._deadline - time.monotonic()))
        except queue.Empty as exc:
            self.close()
            raise TimeoutError(
                f"Engine Core stream request exceeded {self._client.request_timeout_seconds:g} seconds."
            ) from exc
        kind = message.get("kind")
        if kind == "event":
            return message["value"]
        if kind in {"done", "local_stop"}:
            self._finish()
            raise StopIteration
        if kind == "error":
            self._finish()
            self._client._raise_remote(message)
        self._finish()
        raise EngineClientError(f"Unexpected Engine Core stream message: {kind}")

    def cancel(self) -> None:
        """向 Engine Core 發送取消，並等候目前批次完成其清理路徑。"""
        with self._state_lock:
            if self._finished:
                return
            should_send = not self._cancel_sent
            self._cancel_sent = True
        if should_send:
            try:
                self._client._send({"kind": "cancel", "request_id": self._request_id})
            except EngineClientError:
                self._finish()
                return

        deadline = time.monotonic() + min(30.0, self._client.request_timeout_seconds)
        while time.monotonic() < deadline:
            try:
                message = self._pending.get(timeout=min(0.25, deadline - time.monotonic()))
            except queue.Empty:
                continue
            if message.get("kind") in {"done", "error", "local_stop"}:
                self._finish()
                return
        self._finish()

    def close(self) -> None:
        self.cancel()

    def _finish(self) -> None:
        with self._state_lock:
            if self._finished:
                return
            self._finished = True
        self._client._unregister(self._request_id)
        # 喚醒可能仍在 IPC queue 讀取中的 __next__ 執行緒。
        self._pending.put({"kind": "local_stop"})


def _load_factory(factory_path: str) -> Any:
    module_name, separator, attribute = factory_path.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError("Engine Core factory must use 'module:function' notation.")
    factory = getattr(importlib.import_module(module_name), attribute)
    if not callable(factory):
        raise TypeError(f"Engine Core factory is not callable: {factory_path}")
    return factory


def _error_message(exc: BaseException) -> dict[str, Any]:
    message = {"error_type": type(exc).__name__, "message": str(exc)}
    if isinstance(exc, InferenceError):
        message.update(status_code=exc.status_code, detail=exc.detail)
    return message


def _send(connection: Any, lock: threading.Lock, message: dict[str, Any]) -> None:
    try:
        with lock:
            connection.send(message)
    except (EOFError, OSError, BrokenPipeError):
        pass


def _engine_core_process_main(connection: Any, factory_path: str) -> None:
    """啟動 Engine Core，並以可多工的 request handler 回應 API Server。"""
    send_lock = threading.Lock()
    state_lock = threading.RLock()
    handlers: dict[str, threading.Thread] = {}
    cancellable_requests: set[str] = set()
    engine: Any = None
    engine_closed = False

    def send(message: dict[str, Any]) -> None:
        _send(connection, send_lock, message)

    try:
        engine = _load_factory(factory_path)()
        engine.start()
        send({"kind": "ready", "pid": os.getpid()})
    except BaseException as exc:
        send({"kind": "startup_error", **_error_message(exc)})
        try:
            if engine is not None:
                engine.close()
        except BaseException:
            pass
        connection.close()
        return

    def handle(message: dict[str, Any]) -> None:
        request_id = str(message["request_id"])
        try:
            operation = str(message["operation"])
            args = tuple(message.get("args", ()))
            kwargs = dict(message.get("kwargs", {}))
            if operation in {"generate_chat", "understand_video_path", "try_live_inference"}:
                kwargs["request_id"] = request_id
            if message["kind"] == "stream":
                stream = engine.stream_chat(*args, request_id=request_id)
                try:
                    for event in stream:
                        send({"kind": "event", "request_id": request_id, "value": event})
                finally:
                    stream.close()
                send({"kind": "done", "request_id": request_id})
            else:
                if operation == "try_live_inference" and message.get("notify_admitted"):
                    kwargs["on_admitted"] = lambda: send({"kind": "admitted", "request_id": request_id})
                result = getattr(engine, operation)(*args, **kwargs)
                send({"kind": "result", "request_id": request_id, "value": result})
        except GeneratorExit:
            send({"kind": "done", "request_id": request_id})
        except BaseException as exc:
            send({"kind": "error", "request_id": request_id, **_error_message(exc)})
        finally:
            try:
                engine.unregister_request(request_id)
            except BaseException:
                pass
            with state_lock:
                handlers.pop(request_id, None)
                cancellable_requests.discard(request_id)

    def cancel(request_id: str) -> None:
        try:
            with state_lock:
                should_cancel = request_id in cancellable_requests
            if should_cancel:
                engine.cancel_request(request_id)
        except BaseException:
            pass

    try:
        while True:
            try:
                message = connection.recv()
            except (EOFError, OSError):
                break
            kind = message.get("kind")
            if kind == "cancel":
                threading.Thread(target=cancel, args=(str(message["request_id"]),), daemon=True).start()
                continue
            if kind == "close":
                with state_lock:
                    active_ids = tuple(handlers)
                    active_handlers = tuple(handlers.values())
                for request_id in active_ids:
                    threading.Thread(target=cancel, args=(request_id,), daemon=True).start()
                for handler in active_handlers:
                    handler.join(timeout=30)
                try:
                    engine.close()
                    engine_closed = True
                    send({"kind": "closed", "request_id": message["request_id"]})
                except BaseException as exc:
                    send({"kind": "error", "request_id": message["request_id"], **_error_message(exc)})
                break
            if kind not in {"invoke", "stream"}:
                send({"kind": "error", "request_id": message.get("request_id", ""), "error_type": "ValueError", "message": f"Unsupported Engine Core message: {kind}"})
                continue
            request_id = str(message.get("request_id", ""))
            if not request_id:
                send({"kind": "error", "request_id": request_id, "error_type": "ValueError", "message": "request_id is required."})
                continue
            with state_lock:
                if request_id in handlers:
                    send({"kind": "error", "request_id": request_id, "error_type": "ValueError", "message": "Duplicate Engine Core request_id."})
                    continue
                operation = str(message.get("operation", ""))
                is_cancellable = operation in {"generate_chat", "understand_video_path", "try_live_inference"} or kind == "stream"
                if is_cancellable:
                    try:
                        engine.register_request(request_id)
                    except BaseException as exc:
                        send({"kind": "error", "request_id": request_id, **_error_message(exc)})
                        continue
                    cancellable_requests.add(request_id)
                handler = threading.Thread(target=handle, args=(message,), name=f"engine-client-{request_id[:8]}", daemon=True)
                handlers[request_id] = handler
                handler.start()
    finally:
        if engine is not None and not engine_closed:
            try:
                engine.close()
            except BaseException:
                pass
        connection.close()


class EngineClient:
    """API Server 使用的多工 Engine Core IPC proxy。"""

    def __init__(
        self,
        factory_path: str,
        *,
        startup_timeout_seconds: float,
        request_timeout_seconds: float,
        max_video_bytes: int,
    ) -> None:
        if min(startup_timeout_seconds, request_timeout_seconds, max_video_bytes) <= 0:
            raise ValueError("Engine Core timeouts and video size limit must be positive.")
        self.factory_path = factory_path
        self.startup_timeout_seconds = float(startup_timeout_seconds)
        self.request_timeout_seconds = float(request_timeout_seconds)
        self.max_video_bytes = int(max_video_bytes)
        self._process: Any = None
        self._connection: Any = None
        self._send_lock = threading.Lock()
        self._pending_lock = threading.RLock()
        self._pending: dict[str, queue.Queue[dict[str, Any]]] = {}
        self._reader: threading.Thread | None = None
        self._closing = threading.Event()
        self._engine_core_pid: int | None = None
        self._live_sessions: Any = None

    @property
    def pid(self) -> int | None:
        return self._engine_core_pid

    def start(self) -> None:
        if self._process is not None and self._process.is_alive():
            return
        context = multiprocessing.get_context("spawn")
        parent, child = context.Pipe(duplex=True)
        process = context.Process(
            target=_engine_core_process_main,
            args=(child, self.factory_path),
            name="inference-engine-core",
        )
        process.start()
        child.close()
        self._process = process
        self._connection = parent
        if not parent.poll(self.startup_timeout_seconds):
            self._stop_process(force=True)
            raise TimeoutError(f"Engine Core did not start within {self.startup_timeout_seconds:g} seconds.")
        message = parent.recv()
        if message.get("kind") != "ready":
            self._stop_process(force=True)
            self._raise_remote(message)
        self._engine_core_pid = int(message["pid"])
        self._closing.clear()
        self._reader = threading.Thread(target=self._reader_loop, name="engine-core-ipc-reader", daemon=True)
        self._reader.start()

    def close(self) -> None:
        if self._live_sessions is not None:
            self._live_sessions.stop_all()
        if self._process is None:
            return
        if self._process.is_alive() and self._connection is not None:
            try:
                self._request_message("close", {}, expected={"closed"})
            except BaseException:
                pass
        self._closing.set()
        if self._reader is not None:
            self._reader.join(timeout=2)
        self._stop_process(force=False)

    def health(self) -> dict[str, Any]:
        data = dict(self._invoke("health"))
        data["api_pid"] = os.getpid()
        data["engine_core_pid"] = self._engine_core_pid
        data["live_sessions"] = self.live_session_count()
        return data

    def generate_chat(
        self, request_body: dict[str, Any], prepared_input: PreparedChatInput | None = None
    ) -> Any:
        if prepared_input is None:
            return self._invoke("generate_chat", request_body)
        return self._invoke("generate_chat", request_body, prepared_input)

    def stream_chat(
        self, request_body: dict[str, Any], prepared_input: PreparedChatInput | None = None
    ) -> EngineClientStream:
        return EngineClientStream(self, request_body, prepared_input)

    def understand_video(self, upload_stream: Any, filename: str | None, request_body: dict[str, Any]) -> Any:
        with tempfile.TemporaryDirectory(prefix="api-inference-video-") as temp_dir:
            destination = uploaded_video_path(filename, temp_dir)
            started = time.perf_counter()
            upload_bytes = save_upload(upload_stream, destination, self.max_video_bytes)
            staging_seconds = time.perf_counter() - started
            return self._invoke(
                "understand_video_path",
                str(destination),
                filename,
                request_body,
                upload_bytes,
                staging_seconds,
            )

    def generation_options(self, request_body: dict[str, Any]) -> dict[str, Any]:
        return self._invoke("generation_options", request_body)

    def validate_live_video_backend(self, backend: str, codec_engine: str | None) -> None:
        self._invoke("validate_live_video_backend", backend, codec_engine)

    def try_live_inference(self, operation: str, *args: Any, on_admitted: Any = None) -> tuple[bool, Any]:
        result = self._invoke(
            "try_live_inference",
            operation,
            *args,
            notify_admitted=on_admitted is not None,
            on_admitted=on_admitted,
        )
        return result

    def attach_live_sessions(self, manager: Any) -> None:
        self._live_sessions = manager

    def live_settings(self) -> Any:
        return self._require_live_sessions().settings

    def live_session_count(self) -> int:
        return 0 if self._live_sessions is None else self._live_sessions.active_count()

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

    def _invoke(self, operation: str, *args: Any, notify_admitted: bool = False, on_admitted: Any = None) -> Any:
        request_id = uuid.uuid4().hex
        return self._request_message(
            "invoke",
            {
                "request_id": request_id,
                "operation": operation,
                "args": args,
                "kwargs": {},
                "notify_admitted": notify_admitted,
            },
            expected={"result"},
            on_admitted=on_admitted,
        )

    def _request_message(
        self,
        kind: str,
        message: dict[str, Any],
        *,
        expected: set[str],
        on_admitted: Any = None,
    ) -> Any:
        request_id = str(message.get("request_id") or uuid.uuid4().hex)
        message["request_id"] = request_id
        pending = self._register(request_id)
        try:
            self._send({"kind": kind, **message})
            deadline = time.monotonic() + self.request_timeout_seconds
            while True:
                item = pending.get(timeout=max(0.01, deadline - time.monotonic()))
                response_kind = item.get("kind")
                if response_kind == "admitted":
                    if on_admitted is not None:
                        on_admitted()
                    continue
                if response_kind in expected:
                    return item.get("value")
                if response_kind == "error":
                    self._raise_remote(item)
                raise EngineClientError(f"Unexpected Engine Core response: {response_kind}")
        except queue.Empty as exc:
            self._send({"kind": "cancel", "request_id": request_id})
            raise TimeoutError(f"Engine Core request exceeded {self.request_timeout_seconds:g} seconds.") from exc
        finally:
            self._unregister(request_id)

    def _register(self, request_id: str) -> queue.Queue[dict[str, Any]]:
        pending: queue.Queue[dict[str, Any]] = queue.Queue()
        with self._pending_lock:
            if request_id in self._pending:
                raise EngineClientError(f"Duplicate local Engine Core request: {request_id}")
            self._pending[request_id] = pending
        return pending

    def _unregister(self, request_id: str) -> None:
        with self._pending_lock:
            self._pending.pop(request_id, None)

    def _send(self, message: dict[str, Any]) -> None:
        if self._connection is None or self._process is None or not self._process.is_alive():
            raise EngineClientError("Engine Core process is not running.")
        try:
            with self._send_lock:
                self._connection.send(message)
        except (EOFError, OSError, BrokenPipeError) as exc:
            raise EngineClientError("Engine Core IPC connection is closed.") from exc

    def _reader_loop(self) -> None:
        try:
            while not self._closing.is_set():
                message = self._connection.recv()
                request_id = str(message.get("request_id", ""))
                with self._pending_lock:
                    pending = self._pending.get(request_id)
                if pending is not None:
                    pending.put(message)
        except (EOFError, OSError) as exc:
            error = EngineClientError(f"Engine Core IPC reader stopped: {exc}")
            with self._pending_lock:
                pending_requests = tuple(self._pending.values())
            for pending in pending_requests:
                pending.put({"kind": "error", "error_type": "EngineClientError", "message": str(error)})

    @staticmethod
    def _raise_remote(message: dict[str, Any]) -> None:
        if message.get("error_type") == "InferenceError":
            detail = str(message.get("detail", message.get("message", "Inference failed.")))
            raise InferenceError(int(message.get("status_code", 500)), detail)
        if message.get("error_type") == "TimeoutError":
            raise TimeoutError(str(message.get("message", "Engine Core request timed out.")))
        raise EngineClientError(
            f"Engine Core failed ({message.get('error_type', 'Error')}): {message.get('message', 'unknown error')}"
        )

    def _stop_process(self, *, force: bool) -> None:
        process, connection = self._process, self._connection
        if process is not None:
            if not force:
                process.join(timeout=30)
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
        if connection is not None:
            connection.close()
        self._process = None
        self._connection = None
        self._engine_core_pid = None
