"""定義 Engine 與模型執行 Worker 之間的呼叫邊界。"""

from __future__ import annotations

import importlib
from collections.abc import Iterator, Sequence
from typing import Any, Protocol

from engine.contracts import ScheduledWorkItem, WorkerResult


class ModelWorker(Protocol):
    """可由程序內或獨立程序中的模型 Worker 實作。"""

    def start(self) -> None: ...

    def invoke(self, operation: str, *args: Any, **kwargs: Any) -> Any: ...

    def execute_batch(self, requests: Sequence[ScheduledWorkItem]) -> Sequence[WorkerResult]: ...

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

    def execute_batch(self, requests: Sequence[ScheduledWorkItem]) -> Sequence[WorkerResult]:
        method = getattr(self.model_adapter, "execute_batch", None)
        if callable(method):
            return method(requests)

        # 舊版 fake Adapter 可逐項執行；正式模型 Adapter 使用明確的批次契約。
        results: list[WorkerResult] = []
        for request in requests:
            operation = {
                "prefill": "prefill_generation",
                "decode": "decode_generation_step",
            }.get(request.phase.value)
            if operation is None:
                raise ValueError(f"Unsupported model execution phase: {request.phase.value}")
            method = getattr(self.model_adapter, operation)
            step = method(request.request_id, request.token_budget)
            results.append(
                WorkerResult(
                    request_id=request.request_id,
                    text_delta=str(step.get("text", "")),
                    generated_tokens=int(step.get("generated_tokens", 0)),
                    finished=bool(step.get("done", False)),
                    finish_reason=step.get("finish_reason"),
                    prompt_tokens_processed=(
                        request.prompt_offset + request.token_budget
                        if request.phase.value == "prefill" else None
                    ),
                )
            )
        return results

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


def _resolve_factory_path(factory_path: str) -> None:
    module_name, separator, attribute = factory_path.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError("Worker factory must use 'module:function' notation.")
