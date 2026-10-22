"""定義推論引擎元件的線上契約與排程資料結構。

Engine 請求可帶小型資料或本機媒體參照，不傳影片 bytes、GPU tensor 或模型 Cache。
Worker 批次入口已接入現行路徑；Mage-VL Runner 暫時限制每批一個 work item。
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol


class RequestStatus(str, Enum):
    ACCEPTED = "accepted"
    PREPARING = "preparing"
    WAITING = "waiting"
    RUNNING = "running"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


class ExecutionPhase(str, Enum):
    PREFILL = "prefill"
    DECODE = "decode"
    ATOMIC = "atomic"


class EngineEventKind(str, Enum):
    START = "start"
    TEXT = "text"
    FINISH = "finish"
    ERROR = "error"


@dataclass(frozen=True)
class EngineRequest:
    request_id: str
    model_alias: str
    operation: str
    payload: Mapping[str, Any] = field(default_factory=dict)
    sampling_params: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.request_id or not self.model_alias or not self.operation:
            raise ValueError("request_id, model_alias, and operation are required.")


@dataclass(frozen=True)
class CacheRequest:
    request_id: str
    prompt_tokens: int
    max_output_tokens: int

    def __post_init__(self) -> None:
        if not self.request_id or self.prompt_tokens < 1 or self.max_output_tokens < 1:
            raise ValueError("A cache request requires an ID and positive token counts.")


@dataclass(frozen=True)
class CacheReservation:
    """Block ID 是邏輯識別碼，不代表已配置實體 GPU page。"""

    request_id: str
    block_ids: tuple[int, ...]
    reserved_tokens: int
    estimated_bytes: int


@dataclass
class RequestState:
    request: EngineRequest
    status: RequestStatus = RequestStatus.ACCEPTED
    phase: ExecutionPhase = ExecutionPhase.PREFILL
    prompt_tokens: int = 0
    processed_prompt_tokens: int = 0
    generated_tokens: int = 0
    max_output_tokens: int = 0
    cache_reservation: CacheReservation | None = None


@dataclass(frozen=True)
class ScheduledWorkItem:
    request_id: str
    phase: ExecutionPhase
    token_budget: int
    prompt_offset: int = 0
    cache_block_ids: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not self.request_id or self.token_budget < 1 or self.prompt_offset < 0:
            raise ValueError("A work item requires an ID, positive budget, and non-negative offset.")


@dataclass(frozen=True)
class ScheduledBatch:
    work_items: tuple[ScheduledWorkItem, ...]
    total_scheduled_tokens: int

    def __post_init__(self) -> None:
        if not self.work_items:
            raise ValueError("A scheduled batch must contain at least one work item.")
        request_ids = [item.request_id for item in self.work_items]
        if len(request_ids) != len(set(request_ids)):
            raise ValueError("A scheduled batch must not contain duplicate request ids.")
        expected_tokens = sum(item.token_budget for item in self.work_items)
        if self.total_scheduled_tokens != expected_tokens:
            raise ValueError("total_scheduled_tokens must equal the sum of item token budgets.")


@dataclass(frozen=True)
class WorkerResult:
    request_id: str
    text_delta: str = ""
    generated_tokens: int = 0
    finished: bool = False
    finish_reason: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EngineEvent:
    request_id: str
    kind: EngineEventKind
    text: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    finish_reason: str | None = None
    error_code: str | None = None
    error_message: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


class EngineClientProtocol(Protocol):
    def submit(self, request: EngineRequest) -> None: ...

    def events(self, request_id: str) -> Iterator[EngineEvent]: ...

    def cancel(self, request_id: str) -> None: ...


class SchedulerProtocol(Protocol):
    def add_request(self, state: RequestState) -> None: ...

    def schedule(self) -> ScheduledBatch | None: ...

    def postprocess(self, results: Sequence[WorkerResult]) -> None: ...

    def cancel_request(self, request_id: str) -> None: ...


class KVCacheManagerProtocol(Protocol):
    def reserve_batch(
        self, requests: Sequence[CacheRequest]
    ) -> Mapping[str, CacheReservation] | None: ...

    def release(self, request_id: str) -> None: ...


class ModelWorkerProtocol(Protocol):
    def start(self) -> None: ...

    def execute_batch(self, requests: Sequence[ScheduledWorkItem]) -> Sequence[WorkerResult]: ...

    def close(self) -> None: ...
