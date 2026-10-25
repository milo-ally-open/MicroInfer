"""Engine Core 对外执行后端所需的最小介面。"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

from inference.engine.contracts import ScheduledWorkItem, WorkerResult


class Executor(Protocol):
    """Engine Core 使用的执行契约，不暴露程序或硬体细节。"""

    @property
    def pid(self) -> int | None: ...

    def start(self) -> None: ...

    def invoke(self, operation: str, *args: Any, **kwargs: Any) -> Any: ...

    def execute_batch(self, requests: Sequence[ScheduledWorkItem]) -> Sequence[WorkerResult]: ...

    def health_info(self) -> dict[str, Any]: ...

    def gpu_status(self) -> dict[str, Any]: ...

    def close(self) -> None: ...
