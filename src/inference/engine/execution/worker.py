"""定义 Engine 与模型执行 Worker 之间的呼叫边界。"""

from __future__ import annotations

import importlib
import logging
from collections.abc import Sequence
from typing import Any, Protocol

from inference.engine.contracts import ScheduledWorkItem, WorkerResult


logger = logging.getLogger(__name__)


class ModelWorker(Protocol):
    """可由程序内或独立程序中的模型 Worker 实作。"""

    def start(self) -> None: ...

    def invoke(self, operation: str, *args: Any, **kwargs: Any) -> Any: ...

    def execute_batch(self, requests: Sequence[ScheduledWorkItem]) -> Sequence[WorkerResult]: ...

    def health_info(self) -> dict[str, Any]: ...

    def gpu_status(self) -> dict[str, Any]: ...

    def close(self) -> None: ...


class ModelRuntime:
    """以单一模型 Runtime 门面组合模型执行与请求适配元件。"""

    def __init__(self, model_runner: Any, model_adapter: Any) -> None:
        self.model_runner = model_runner
        self.model_adapter = model_adapter

    def load(self) -> None:
        self.model_runner.load()

    def invoke(self, operation: str, *args: Any, **kwargs: Any) -> Any:
        return getattr(self.model_adapter, operation)(*args, **kwargs)

    def execute_batch(self, requests: Sequence[ScheduledWorkItem]) -> Sequence[WorkerResult]:
        return self.model_adapter.execute_batch(requests)

    def health_info(self) -> dict[str, Any]:
        return self.model_adapter.health_info()

    def gpu_status(self) -> dict[str, Any]:
        return self.model_runner.gpu_status()

    def close(self) -> None:
        self.model_runner.close()

    def __getattr__(self, operation: str) -> Any:
        """将 Engine 的模型操作转交至模型 Adapter，维持单一 Runtime 契约。"""
        return getattr(self.model_adapter, operation)


class InProcessModelWorker:
    """以同一个 Runtime 物件执行 Worker RPC。"""

    def __init__(self, runtime: Any, model_adapter: Any | None = None) -> None:
        # 保留测试与旧内部呼叫的二参数形式；正式模型入口只传入 Runtime。
        self.runtime = ModelRuntime(runtime, model_adapter) if model_adapter is not None else runtime

    def start(self) -> None:
        self.runtime.load()

    def invoke(self, operation: str, *args: Any, **kwargs: Any) -> Any:
        invoke = getattr(self.runtime, "invoke", None)
        if callable(invoke):
            return invoke(operation, *args, **kwargs)
        return getattr(self.runtime, operation)(*args, **kwargs)

    def execute_batch(self, requests: Sequence[ScheduledWorkItem]) -> Sequence[WorkerResult]:
        return self.runtime.execute_batch(requests)

    def health_info(self) -> dict[str, Any]:
        return self.runtime.health_info()

    def gpu_status(self) -> dict[str, Any]:
        return self.runtime.gpu_status()

    def close(self) -> None:
        self.runtime.close()


class WorkerProcessError(RuntimeError):
    """Model Worker 子程序回传的错误。"""


def _resolve_factory(path: str) -> Any:
    module_name, separator, attribute = path.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError("Worker factory must use 'module:function' notation.")
    return getattr(importlib.import_module(module_name), attribute)


def _worker_process_main(connection: Any, factory_path: str) -> None:
    """在独立子程序载入模型，并以 Pipe 执行 Engine RPC。"""
    worker: ModelWorker | None = None
    worker_closed = False
    try:
        worker = _resolve_factory(factory_path)()
        worker.start()
        connection.send({"kind": "ready", "health": worker.health_info()})
    except BaseException as exc:
        # Runtime.load 可能在权重载入途中已配置 GPU／CPU 资源；启动失败也要执行同一个 close。
        if worker is not None:
            try:
                worker.close()
            except BaseException:
                logger.exception("Could not clean up a partially started Model Worker")
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
            logger.exception("Model Worker operation failed")
            connection.send({"kind": "error", "error_type": type(exc).__name__, "message": str(exc)})
    try:
        if worker is not None and not worker_closed:
            worker.close()
    except BaseException:
        logger.exception("Could not close Model Worker during process shutdown")
    connection.close()


def _resolve_factory_path(factory_path: str) -> None:
    module_name, separator, attribute = factory_path.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError("Worker factory must use 'module:function' notation.")
