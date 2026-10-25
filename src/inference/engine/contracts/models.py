"""定义推论引擎线上使用的请求与批次资料结构。

Engine 请求可带小型资料或本机媒体参照，不传影片 bytes、GPU tensor 或模型 Cache。
Worker 批次入口已接入现行路径；Mage-VL Runner 会合并相容输入，不相容时逐请求执行。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ExecutionPhase(str, Enum):
    PREFILL = "prefill"
    DECODE = "decode"


@dataclass(frozen=True)
class PreparedChatInput:
    """由 API 输入处理器建立、可安全跨 Engine IPC 传递的模型输入。"""

    model_alias: str
    input_ids: tuple[int, ...]
    sampling_params: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.model_alias or not self.input_ids:
            raise ValueError("Prepared chat input requires a model alias and at least one token.")
        if any(token_id < 0 for token_id in self.input_ids):
            raise ValueError("Prepared chat input token IDs must be non-negative.")


@dataclass(frozen=True)
class InferenceResult:
    """跨 API、Engine 与模型 Adapter 传递的单次推论结果。"""

    content: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str
    metadata: Mapping[str, Any] = field(default_factory=dict)
    token_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class InferenceStreamEvent:
    """不依赖 SSE 或 OpenAI schema 的逐请求生成事件。"""

    kind: str
    text: str | None = None
    token_id: int | None = None
    token_ids: tuple[int, ...] = ()
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    finish_reason: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


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
    token_id: int | None = None
    generated_tokens: int = 0
    finished: bool = False
    finish_reason: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    prompt_tokens_processed: int | None = None
