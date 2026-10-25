"""为批次 Model Runner 建立可验证的 token 执行计划。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


Phase = Literal["prefill", "decode"]


@dataclass(frozen=True)
class SequenceProgress:
    """Scheduler 在一次规划前取得的序列进度快照。"""

    request_id: str
    phase: Phase
    prompt_tokens_total: int
    prompt_tokens_processed: int
    generated_tokens: int = 0
    prefill_min_chunk_tokens: int = 1

    @property
    def prompt_tokens_remaining(self) -> int:
        return self.prompt_tokens_total - self.prompt_tokens_processed


@dataclass(frozen=True)
class ScheduledSequence:
    """批次中单一序列要执行的工作。"""

    request_id: str
    phase: Phase
    token_budget: int
    prompt_offset: int = 0
    final_prefill_chunk: bool = False


@dataclass(frozen=True)
class BatchPlan:
    """供 Engine 交给批次 Model Runner 的单轮计划。"""

    sequences: tuple[ScheduledSequence, ...]
    scheduled_tokens: int
    next_cursor: int


def plan_batch(
    progress: list[SequenceProgress],
    *,
    max_num_seqs: int,
    max_num_scheduled_tokens: int,
    prefill_chunk_size: int | None = None,
    cursor: int = 0,
) -> BatchPlan:
    """混合安排 decode 与 prefill，并依首段视觉 token 边界切分 prompt。

    每个 request 每轮最多出现一次。Prefill 可切成多轮；Model Runner 必须回报实际处理的
    prompt token 数，Engine 才能更新下一轮进度。`prefill_min_chunk_tokens` 用于要求首段涵盖
    完整视觉 token 区段，避免将图像／影片标记和视觉特征分开。此纯规划函式不执行模型或管理 KV 张量。
    """
    if max_num_seqs < 1 or max_num_scheduled_tokens < 1:
        raise ValueError("Batch limits must be positive.")
    if prefill_chunk_size is not None and prefill_chunk_size < 1:
        raise ValueError("prefill_chunk_size must be positive when configured.")
    if not progress:
        return BatchPlan((), 0, 0)

    ids: set[str] = set()
    for item in progress:
        if not item.request_id or item.request_id in ids:
            raise ValueError("Batch progress must contain unique, non-empty request ids.")
        ids.add(item.request_id)
        if item.phase not in {"prefill", "decode"}:
            raise ValueError(f"Invalid scheduling phase for request {item.request_id}.")
        if item.prompt_tokens_total < 1 or not 0 <= item.prompt_tokens_processed <= item.prompt_tokens_total:
            raise ValueError(f"Invalid prompt progress for request {item.request_id}.")
        if item.generated_tokens < 0:
            raise ValueError(f"Invalid generated token count for request {item.request_id}.")
        if item.prefill_min_chunk_tokens < 1 or item.prefill_min_chunk_tokens > item.prompt_tokens_total:
            raise ValueError(f"Invalid first prefill chunk size for request {item.request_id}.")
        if item.phase == "prefill" and item.prompt_tokens_remaining == 0:
            raise ValueError(f"Prefill request {item.request_id} has no remaining prompt tokens.")
        if item.phase == "decode" and item.prompt_tokens_remaining != 0:
            raise ValueError(f"Decode request {item.request_id} has an incomplete prompt.")

    start = cursor % len(progress)
    rotated = progress[start:] + progress[:start]
    index_by_id = {item.request_id: (start + index) % len(progress) for index, item in enumerate(rotated)}
    chunk_limit = prefill_chunk_size or max_num_scheduled_tokens
    indivisible_visual_prefills = [
        item for item in rotated
        if item.phase == "prefill"
        and item.prompt_tokens_processed == 0
        and item.prefill_min_chunk_tokens > chunk_limit
    ]
    ordered = indivisible_visual_prefills + [
        item for item in rotated if item.phase == "decode"
    ] + [
        item for item in rotated
        if item.phase == "prefill" and item not in indivisible_visual_prefills
    ]

    remaining_budget = max_num_scheduled_tokens
    scheduled: list[ScheduledSequence] = []
    for item in ordered:
        if len(scheduled) >= max_num_seqs or remaining_budget == 0:
            break
        if item.phase == "decode":
            scheduled.append(ScheduledSequence(item.request_id, "decode", 1))
            remaining_budget -= 1
            continue

        first_chunk_min = (
            item.prefill_min_chunk_tokens if item.prompt_tokens_processed == 0 else 1
        )
        request_chunk_limit = max(chunk_limit, first_chunk_min)
        if remaining_budget < first_chunk_min:
            continue
        token_budget = min(item.prompt_tokens_remaining, remaining_budget, request_chunk_limit)
        scheduled.append(ScheduledSequence(
            request_id=item.request_id,
            phase="prefill",
            token_budget=token_budget,
            prompt_offset=item.prompt_tokens_processed,
            final_prefill_chunk=token_budget == item.prompt_tokens_remaining,
        ))
        remaining_budget -= token_budget

    scheduled_tokens = max_num_scheduled_tokens - remaining_budget
    if not scheduled:
        next_cursor = start
    else:
        next_cursor = (index_by_id[scheduled[-1].request_id] + 1) % len(progress)
    return BatchPlan(tuple(scheduled), scheduled_tokens, next_cursor)
