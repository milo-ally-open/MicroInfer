"""定義 Engine 與模型執行 Worker 之間的呼叫邊界。"""

from __future__ import annotations

import importlib
import logging
from collections.abc import Sequence
from typing import Any, Protocol

from engine.contracts import ScheduledWorkItem, WorkerResult


class ModelWorker(Protocol):
    """可由程序內或獨立程序中的模型 Worker 實作。"""

    def start(self) -> None: ...

    def invoke(self, operation: str, *args: Any, **kwargs: Any) -> Any: ...

    def execute_batch(self, requests: Sequence[ScheduledWorkItem]) -> Sequence[WorkerResult]: ...

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

    def execute_batch(self, requests: Sequence[ScheduledWorkItem]) -> Sequence[WorkerResult]:
        return self.model_adapter.execute_batch(requests)

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
            else:
                raise ValueError(f"Unsupported Worker message kind: {kind}")
        except BaseException as exc:
            logging.getLogger(__name__).exception("Model Worker operation failed")
            connection.send({"kind": "error", "error_type": type(exc).__name__, "message": str(exc)})
    try:
        if worker is not None and not worker_closed:
            worker.close()
    except BaseException:
        pass
    connection.close()


def _resolve_factory_path(factory_path: str) -> None:
    module_name, separator, attribute = factory_path.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError("Worker factory must use 'module:function' notation.")
