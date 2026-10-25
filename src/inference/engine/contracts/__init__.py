"""API、Engine、执行层及模型 Adapter 共用的进程间契约。"""

from .errors import InferenceError
from .models import (
    ExecutionPhase,
    InferenceResult,
    InferenceStreamEvent,
    PreparedChatInput,
    ScheduledBatch,
    ScheduledWorkItem,
    WorkerResult,
)

__all__ = [
    "ExecutionPhase",
    "InferenceError",
    "InferenceResult",
    "InferenceStreamEvent",
    "PreparedChatInput",
    "ScheduledBatch",
    "ScheduledWorkItem",
    "WorkerResult",
]
