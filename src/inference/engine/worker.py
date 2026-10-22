"""定義 Engine 與模型執行 Worker 之間的呼叫邊界。"""

from __future__ import annotations

import importlib
import multiprocessing
import threading
import time
from collections.abc import Iterator
from typing import Any, Protocol


class ModelWorker(Protocol):
    """可由程序內或獨立程序中的模型 Worker 實作。"""

    def start(self) -> None: ...

    def invoke(self, operation: str, *args: Any, **kwargs: Any) -> Any: ...

    def stream(self, operation: str, *args: Any, **kwargs: Any) -> Iterator[Any]: ...

    def health_info(self) -> dict[str, Any]: ...

    def gpu_status(self) -> dict[str, Any]: ...

    def close(self) -> None: ...


class InProcessModelWorker:
    """目前使用的單程序 Worker；Runner 與 Adapter 可整體替換。"""

    def __init__(self, model_runner: Any, model_adapter: Any) -> None:
        self.model_runner = model_runner
        self.model_adapter = model_adapter

    def start(self) -> None:
        self.model_runner.load()

    def invoke(self, operation: str, *args: Any, **kwargs: Any) -> Any:
        method = getattr(self.model_adapter, operation)
        return method(*args, **kwargs)

    def stream(self, operation: str, *args: Any, **kwargs: Any) -> Iterator[Any]:
        method = getattr(self.model_adapter, operation)
        yield from method(*args, **kwargs)

    def health_info(self) -> dict[str, Any]:
        return self.model_adapter.health_info()

    def gpu_status(self) -> dict[str, Any]:
        return self.model_runner.gpu_status()

    def close(self) -> None:
        self.model_runner.close()


class WorkerProcessError(RuntimeError):
    """Model Worker 子程序回傳的錯誤。"""


def _resolve_factory(path: str) -> Any:
    module_name, separator, attribute = path.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError("Worker factory must use 'module:function' notation.")
    return getattr(importlib.import_module(module_name), attribute)


def _worker_process_main(connection: Any, factory_path: str) -> None:
    """在獨立子程序載入模型，並以 Pipe 執行 Engine RPC。"""
    worker: ModelWorker | None = None
    worker_closed = False
    try:
        worker = _resolve_factory(factory_path)()
        worker.start()
        connection.send({"kind": "ready", "health": worker.health_info()})
    except BaseException as exc:
        try:
            connection.send({"kind": "error", "error_type": type(exc).__name__, "message": str(exc)})
        finally:
            connection.close()
        return

    while True:
        try:
            message = connection.recv()
        except (EOFError, OSError):
            break
        kind = message.get("kind")
        if kind == "close":
            try:
                worker.close()
                worker_closed = True
                connection.send({"kind": "result", "value": None})
            except BaseException as exc:
                connection.send({"kind": "error", "error_type": type(exc).__name__, "message": str(exc)})
            break
        try:
            if kind == "invoke":
                operation = message["operation"]
                if operation == "__health_info__":
                    result = worker.health_info()
                elif operation == "__gpu_status__":
                    result = worker.gpu_status()
                else:
                    result = worker.invoke(operation, *message.get("args", ()), **message.get("kwargs", {}))
                connection.send({"kind": "result", "value": result})
            elif kind == "stream":
                for event in worker.stream(
                    message["operation"], *message.get("args", ()), **message.get("kwargs", {})
                ):
                    connection.send({"kind": "event", "value": event})
                connection.send({"kind": "done"})
            else:
                raise ValueError(f"Unsupported Worker message kind: {kind}")
        except BaseException as exc:
            connection.send({"kind": "error", "error_type": type(exc).__name__, "message": str(exc)})
    try:
        if worker is not None and not worker_closed:
            worker.close()
    except BaseException:
        pass
    connection.close()


class MultiprocessModelWorker:
    """以獨立子程序執行模型 Worker，避免 API Server 載入 GPU Runtime。"""

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
        process = context.Process(target=_worker_process_main, args=(child, self.factory_path), name="inference-model-worker")
        process.start()
        child.close()
        self._process = process
        self._connection = parent
        message = self._receive(self.startup_timeout_seconds)
        if message.get("kind") == "error":
            self._stop_process()
            raise WorkerProcessError(f"Model Worker startup failed ({message.get('error_type')}): {message.get('message')}")
        if message.get("kind") != "ready":
            self._stop_process()
            raise WorkerProcessError("Model Worker exited without a ready handshake.")
        self._health = dict(message.get("health") or {})

    def invoke(self, operation: str, *args: Any, **kwargs: Any) -> Any:
        return self._request("invoke", operation=operation, args=args, kwargs=kwargs)

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
        raise WorkerProcessError(f"Model Worker failed ({message.get('error_type', 'Error')}): {message.get('message', 'unknown error')}")

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


def _resolve_factory_path(factory_path: str) -> None:
    module_name, separator, attribute = factory_path.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError("Worker factory must use 'module:function' notation.")
