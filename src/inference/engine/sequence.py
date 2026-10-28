"""定义推理引擎使用的请求与批次数据结构。

Engine 请求可以携带小型元数据或本地媒体引用，不传递视频字节、GPU 张量或模型 Cache。
Mage-VL Runner 会合并兼容输入；输入不兼容时按请求逐个执行。
"""

from __future__ import annotations


from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ExecutionPhase(str, Enum):
    PREFILL = "prefill"
    DECODE = "decode"


@dataclass(frozen=True)
class PreparedChatInput:
    """由 API 输入处理器创建、可安全跨 Engine IPC 传递的模型输入。"""

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
    """跨 API、Engine 与模型 Adapter 传递的单次推理结果。"""

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


"""Engine Core 与 Scheduler 共用的单请求生成状态。"""


from dataclasses import dataclass, field


@dataclass
class SequenceState:
    """可跨调度轮次更新的请求进度；不持有模型 Cache 或 GPU tensor。"""

    request_id: str
    prompt_tokens_total: int
    max_output_tokens: int
    prefill_min_chunk_tokens: int = 1
    phase: str = "prefill"
    prompt_tokens_processed: int = 0
    generated_tokens: int = 0
    block_table: tuple[int, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("A sequence requires a request ID.")
        if self.prompt_tokens_total < 1 or self.max_output_tokens < 1:
            raise ValueError("A sequence requires positive prompt and output limits.")
        if not 1 <= self.prefill_min_chunk_tokens <= self.prompt_tokens_total:
            raise ValueError("The minimum prefill chunk must fit the prompt.")
        if not 0 <= self.prompt_tokens_processed <= self.prompt_tokens_total:
            raise ValueError("Processed prompt tokens must be within the prompt length.")
        if self.generated_tokens < 0:
            raise ValueError("Generated token count must not be negative.")
        if self.phase not in {"prefill", "decode", "waiting_prefill", "waiting_decode"}:
            raise ValueError(f"Unsupported sequence phase: {self.phase}.")
        self.block_table = tuple(int(block_id) for block_id in self.block_table)

    @property
    def prompt_tokens_remaining(self) -> int:
        return self.prompt_tokens_total - self.prompt_tokens_processed

    @property
    def waiting(self) -> bool:
        return self.phase.startswith("waiting_")


class InferenceError(Exception):
    """可映射至 API 错误响应的推理错误。"""

    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail
