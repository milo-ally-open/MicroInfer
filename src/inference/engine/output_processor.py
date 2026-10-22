"""將 Worker 批次結果轉為逐請求 Engine stream event。"""

from __future__ import annotations

from dataclasses import dataclass

from engine.contracts import WorkerResult


@dataclass(frozen=True)
class EngineOutputEvent:
    """不攜帶 HTTP 格式的單筆輸出增量。"""

    kind: str
    text: str | None = None
    token_id: int | None = None


class EngineOutputProcessor:
    """保留 request ID 對應並統一 Token 與文字 Runner 的輸出形狀。"""

    def process(self, result: WorkerResult) -> tuple[EngineOutputEvent, ...]:
        if result.token_id is not None and result.text_delta:
            raise ValueError("A Worker result cannot contain both token_id and text_delta.")
        if result.token_id is not None:
            return (EngineOutputEvent("token", token_id=int(result.token_id)),)
        if result.text_delta:
            return (EngineOutputEvent("text", text=result.text_delta),)
        return ()
