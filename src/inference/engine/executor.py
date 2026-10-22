"""管理 Engine Core 到 Model Worker 的程序與 IPC 執行邊界。"""

from __future__ import annotations

import multiprocessing
import threading
import time
from collections.abc import Iterator, Sequence
from typing import Any, Protocol

from engine.contracts import ScheduledWorkItem, WorkerResult
from engine.worker import WorkerProcessError, _resolve_factory_path, _worker_process_main


class Executor(Protocol):
    """Engine Core 使用的執行介面，不暴露程序或硬體細節。"""

    @property
    def pid(self) -> int | None: ...

    def start(self) -> None: ...

    def invoke(self, operation: str, *args: Any, **kwargs: Any) -> Any: ...

    def execute_batch(self, requests: Sequence[ScheduledWorkItem]) -> Sequence[WorkerResult]: ...

    def stream(self, operation: str, *args: Any, **kwargs: Any) -> Iterator[Any]: ...

    def health_info(self) -> dict[str, Any]: ...

    def gpu_status(self) -> dict[str, Any]: ...

    def close(self) -> None: ...


class LocalProcessExecutor:
    """以本機 spawn 子程序執行一個模型 Worker，並以 Pipe 傳遞 Engine RPC。"""

    def __init__(
        self,
        factory_path: str,
        *,
        startup_timeout_seconds: float = 2400,
        request_timeout_seconds: float = 1800,
    ) -> None:
        if startup_timeout_seconds <= 0 or request_timeout_seconds <= 0:
            raise ValueError("Worker process timeouts must be positive.")
        _resolve_factory_path(factory_path)
        self.factory_path = factory_path
        self.startup_timeout_seconds = startup_timeout_seconds
        self.request_timeout_seconds = request_timeout_seconds
        self._process: Any = None
        self._connection: Any = None
        self._rpc_lock = threading.RLock()
        self._health: dict[str, Any] = {}

    @property
    def pid(self) -> int | None:
        return self._process.pid if self._process is not None else None

    def start(self) -> None:
        if self._process is not None and self._process.is_alive():
            return
        context = multiprocessing.get_context("spawn")
        parent, child = context.Pipe(duplex=True)
        process = context.Process(
            target=_worker_process_main,
            args=(child, self.factory_path),
            name="inference-model-worker",
        )
        process.start()
        child.close()
        self._process = process
        self._connection = parent
        message = self._receive(self.startup_timeout_seconds)
        if message.get("kind") == "error":
            self._stop_process()
            raise WorkerProcessError(
                f"Model Worker startup failed ({message.get('error_type')}): {message.get('message')}"
            )
        if message.get("kind") != "ready":
            self._stop_process()
            raise WorkerProcessError("Model Worker exited without a ready handshake.")
        self._health = dict(message.get("health") or {})

    def invoke(self, operation: str, *args: Any, **kwargs: Any) -> Any:
        return self._request("invoke", operation=operation, args=args, kwargs=kwargs)

    def execute_batch(self, requests: Sequence[ScheduledWorkItem]) -> Sequence[WorkerResult]:
        return self.invoke("execute_batch", tuple(requests))

    def stream(self, operation: str, *args: Any, **kwargs: Any) -> Iterator[Any]:
        with self._rpc_lock:
            self._ensure_running()
            self._connection.send({"kind": "stream", "operation": operation, "args": args, "kwargs": kwargs})
            while True:
                message = self._receive(self.request_timeout_seconds)
                if message["kind"] == "event":
                    yield message["value"]
                elif message["kind"] == "done":
                    return
                else:
                    self._raise_worker_error(message)

    def health_info(self) -> dict[str, Any]:
        if self._process is None:
            return dict(self._health)
        return self._request("invoke", operation="__health_info__")

    def gpu_status(self) -> dict[str, Any]:
        if self._process is None:
            return {}
        return self._request("invoke", operation="__gpu_status__")

    def close(self) -> None:
        with self._rpc_lock:
            if self._process is None:
                return
            if self._process.is_alive():
                try:
                    self._connection.send({"kind": "close"})
                    message = self._receive(min(self.request_timeout_seconds, 30))
                    if message.get("kind") == "error":
                        self._raise_worker_error(message)
                finally:
                    self._process.join(timeout=30)
            self._stop_process(force=self._process.is_alive())

    def _request(self, kind: str, **message: Any) -> Any:
        with self._rpc_lock:
            self._ensure_running()
            self._connection.send({"kind": kind, **message})
            response = self._receive(self.request_timeout_seconds)
            if response.get("kind") != "result":
                self._raise_worker_error(response)
            return response.get("value")

    def _receive(self, timeout: float) -> dict[str, Any]:
        if self._connection is None:
            raise WorkerProcessError("Model Worker process is not connected.")
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._connection.poll(min(0.25, max(0, deadline - time.monotonic()))):
                try:
                    return self._connection.recv()
                except (EOFError, OSError) as exc:
                    raise WorkerProcessError("Model Worker process closed its IPC connection.") from exc
            if self._process is not None and not self._process.is_alive():
                raise WorkerProcessError(f"Model Worker process exited with code {self._process.exitcode}.")
        self._stop_process(force=True)
        raise TimeoutError(f"Model Worker did not respond within {timeout:g} seconds.")

    def _ensure_running(self) -> None:
        if self._process is None or self._connection is None or not self._process.is_alive():
            raise WorkerProcessError("Model Worker process is not running.")

    @staticmethod
    def _raise_worker_error(message: dict[str, Any]) -> None:
        raise WorkerProcessError(
            f"Model Worker failed ({message.get('error_type', 'Error')}): "
            f"{message.get('message', 'unknown error')}"
        )

    def _stop_process(self, *, force: bool = False) -> None:
        process, connection = self._process, self._connection
        if process is not None:
            if force and process.is_alive():
                process.terminate()
            process.join(timeout=5)
        if connection is not None:
            connection.close()
        self._process = None
        self._connection = None
