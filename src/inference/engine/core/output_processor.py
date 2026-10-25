"""将 Worker 批次结果转为逐请求 Engine stream event。"""

from __future__ import annotations

from dataclasses import dataclass

from inference.engine.contracts import WorkerResult


@dataclass(frozen=True)
class EngineOutputEvent:
    """不携带 HTTP 格式的单笔输出增量。"""

    kind: str
    text: str | None = None
    token_id: int | None = None


class EngineOutputProcessor:
    """保留 request ID 对应并统一 Token 与文字 Runner 的输出形状。"""

    def process(self, result: WorkerResult) -> tuple[EngineOutputEvent, ...]:
        if result.token_id is not None and result.text_delta:
            raise ValueError("A Worker result cannot contain both token_id and text_delta.")
        if result.token_id is not None:
            return (EngineOutputEvent("token", token_id=int(result.token_id)),)
        if result.text_delta:
            return (EngineOutputEvent("text", text=result.text_delta),)
        return ()
