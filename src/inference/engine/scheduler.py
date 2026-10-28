"""创建 prefill/decode 批次并管理请求调度状态。"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Literal

from inference.engine.block_manager import BlockManager
from inference.engine.sequence import (
    ExecutionPhase, ScheduledBatch, ScheduledWorkItem, SequenceState, WorkerResult,
)


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
    完整视觉 token 区段，避免将图像／视频标记和视觉特征分开。此纯规划函数不执行模型或管理 KV 张量。
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




class SchedulerFullError(RuntimeError):
    """等待队列及执行序列均已达上限。"""


class TokenScheduler:
    """以序列上限、轮次 token 预算和 prefill chunk 规划混合 Runner 工作。"""

    def __init__(
        self,
        *,
        max_num_seqs: int,
        max_waiting_requests: int,
        max_num_scheduled_tokens: int,
        batch_wait_seconds: float = 0.01,
        prefill_chunk_size_tokens: int | None = None,
    ) -> None:
        if (
            max_num_seqs < 1 or max_waiting_requests < 0 or max_num_scheduled_tokens < 1
            or batch_wait_seconds < 0
            or (prefill_chunk_size_tokens is not None and prefill_chunk_size_tokens < 1)
        ):
            raise ValueError("Scheduler limits must be positive (waiting requests may be zero).")
        self.max_num_seqs = max_num_seqs
        self.max_waiting_requests = max_waiting_requests
        self.max_num_scheduled_tokens = max_num_scheduled_tokens
        self.batch_wait_seconds = batch_wait_seconds
        self.prefill_chunk_size_tokens = prefill_chunk_size_tokens
        self._condition = threading.Condition()
        self._requests: dict[str, SequenceState] = {}
        self._ready: deque[str] = deque()
        self._ready_since: float | None = None
        self._inflight: set[str] = set()
        self._inflight_items: dict[str, ScheduledWorkItem] = {}
        self._running = False
        self._rounds = 0

    def add_request(
        self,
        request_id: str,
        *,
        prompt_tokens: int,
        max_output_tokens: int,
        prefill_min_chunk_tokens: int = 1,
        cache_block_ids: tuple[int, ...] = (),
    ) -> None:
        if prompt_tokens < 1 or max_output_tokens < 1:
            raise ValueError("Prompt and output token counts must be positive.")
        if not 1 <= prefill_min_chunk_tokens <= prompt_tokens:
            raise ValueError("The first prefill chunk requirement must fit the prompt.")
        if prefill_min_chunk_tokens > self.max_num_scheduled_tokens:
            raise ValueError("The indivisible visual prompt prefix exceeds the per-turn token budget.")
        if prompt_tokens > self.max_num_scheduled_tokens and self.prefill_chunk_size_tokens is None:
            raise ValueError(
                f"Prompt has {prompt_tokens} tokens, exceeding the per-turn budget "
                f"{self.max_num_scheduled_tokens}; chunked prefill is disabled."
            )
        with self._condition:
            if request_id in self._requests:
                raise ValueError(f"Duplicate scheduler request id: {request_id}")
            running_count = self._running_count()
            waiting_count = self._waiting_count()
            if running_count >= self.max_num_seqs and waiting_count >= self.max_waiting_requests:
                raise SchedulerFullError("The inference scheduler is full.")
            phase = "prefill" if running_count < self.max_num_seqs else "waiting_prefill"
            self._requests[request_id] = SequenceState(
                request_id=request_id,
                prompt_tokens_total=prompt_tokens,
                max_output_tokens=max_output_tokens,
                prefill_min_chunk_tokens=prefill_min_chunk_tokens,
                phase=phase,
                block_table=cache_block_ids,
            )
            if phase == "prefill":
                if not self._ready:
                    self._ready_since = time.monotonic()
                self._ready.append(request_id)
            self._condition.notify_all()

    def schedule_batch(self, block_manager: BlockManager | None = None) -> ScheduledBatch | None:
        """以一轮 token 预算创建可同时包含 prefill 与 decode 的批次。"""
        with self._condition:
            if self._running or not self._ready:
                return None
            if self.batch_wait_seconds > 0 and self._ready_since is not None:
                remaining = self.batch_wait_seconds - (time.monotonic() - self._ready_since)
                if remaining > 0:
                    self._condition.wait(timeout=remaining)
                    if self._running or not self._ready:
                        return None
            eligible = [request_id for request_id in self._ready if request_id in self._requests]
            if not eligible:
                return None
            progress = [
                SequenceProgress(
                    request_id=request_id,
                    phase=self._requests[request_id].phase,
                    prompt_tokens_total=self._requests[request_id].prompt_tokens_total,
                    prompt_tokens_processed=self._requests[request_id].prompt_tokens_processed,
                    generated_tokens=self._requests[request_id].generated_tokens,
                    prefill_min_chunk_tokens=self._requests[request_id].prefill_min_chunk_tokens,
                )
                for request_id in eligible
            ]
            plan = plan_batch(
                progress,
                max_num_seqs=self.max_num_seqs,
                max_num_scheduled_tokens=self.max_num_scheduled_tokens,
                prefill_chunk_size=self.prefill_chunk_size_tokens,
            )
            scheduled_items: list[ScheduledWorkItem] = []
            for item in plan.sequences:
                sequence = self._requests[item.request_id]
                if block_manager is not None:
                    required_tokens = (
                        item.prompt_offset + item.token_budget
                        if item.phase == "prefill"
                        else sequence.prompt_tokens_total + sequence.generated_tokens
                    )
                    block_table = block_manager.ensure_capacity(item.request_id, required_tokens)
                    if block_table is None:
                        # KV-pressure requests stay ready; later items that fit may still run.
                        continue
                    sequence.block_table = block_table
                scheduled_items.append(ScheduledWorkItem(
                    request_id=item.request_id,
                    phase=ExecutionPhase(item.phase),
                    token_budget=item.token_budget,
                    prompt_offset=item.prompt_offset,
                    cache_block_ids=sequence.block_table,
                ))
            if not scheduled_items:
                return None
            scheduled = tuple(scheduled_items)
            selected_ids = {item.request_id for item in scheduled}
            total_tokens = sum(item.token_budget for item in scheduled)
            batch = ScheduledBatch(scheduled, total_tokens)
            self._ready = deque(request_id for request_id in self._ready if request_id not in selected_ids)
            self._ready_since = time.monotonic() if self._ready else None
            self._inflight = set(selected_ids)
            self._inflight_items = {item.request_id: item for item in scheduled}
            self._running = True
            self._rounds += 1
            return batch

    def complete_batch(self, results: tuple[WorkerResult, ...] | list[WorkerResult]) -> None:
        """应用批次结果，将未完成 request 放回 decode 队列。"""
        result_ids = [result.request_id for result in results]
        with self._condition:
            if not self._running or len(result_ids) != len(set(result_ids)) or set(result_ids) != self._inflight:
                raise RuntimeError("Worker batch results do not match the in-flight scheduler batch.")
            work_items = self._inflight_items
            for result in results:
                state = self._requests.get(result.request_id)
                work_item = work_items[result.request_id]
                if state is None or work_item.phase is not ExecutionPhase.PREFILL:
                    continue
                expected_processed = work_item.prompt_offset + work_item.token_budget
                if result.prompt_tokens_processed != expected_processed:
                    raise RuntimeError(
                        f"Worker prefill progress for {result.request_id} must be "
                        f"{expected_processed}, got {result.prompt_tokens_processed}."
                    )
                if expected_processed < state.prompt_tokens_total and result.generated_tokens:
                    raise RuntimeError("A partial prefill must not publish a generated token.")
            self._running = False
            self._inflight.clear()
            self._inflight_items = {}
            for result in results:
                state = self._requests.get(result.request_id)
                if state is None:
                    continue
                work_item = work_items[result.request_id]
                if work_item.phase is ExecutionPhase.PREFILL:
                    expected_processed = work_item.prompt_offset + work_item.token_budget
                    state.prompt_tokens_processed = expected_processed
                state.generated_tokens += result.generated_tokens
                if result.finished or state.generated_tokens >= state.max_output_tokens:
                    del self._requests[result.request_id]
                elif (
                    work_item.phase is ExecutionPhase.PREFILL
                    and state.prompt_tokens_processed < state.prompt_tokens_total
                ):
                    state.phase = "prefill"
                    if not self._ready:
                        self._ready_since = time.monotonic()
                    self._ready.append(result.request_id)
                else:
                    state.phase = "decode"
                    if not self._ready:
                        self._ready_since = time.monotonic()
                    self._ready.append(result.request_id)
            self._promote_waiting()
            self._condition.notify_all()

    def abort_batch(self, request_ids: set[str]) -> None:
        """Worker 批次失败时清除本轮调度状态，等待 Engine 统一释放各请求资源。"""
        with self._condition:
            if not self._running or request_ids != self._inflight:
                raise RuntimeError("Only the active Worker batch can be aborted.")
            self._running = False
            self._inflight.clear()
            self._inflight_items = {}
            for request_id in request_ids:
                self._requests.pop(request_id, None)
            self._ready = deque(request_id for request_id in self._ready if request_id not in request_ids)
            self._ready_since = time.monotonic() if self._ready else None
            self._promote_waiting()
            self._condition.notify_all()

    def cancel_request(self, request_id: str) -> None:
        with self._condition:
            self._requests.pop(request_id, None)
            self._ready = deque(item for item in self._ready if item != request_id)
            self._ready_since = time.monotonic() if self._ready else None
            # In-flight work remains registered until complete_batch/abort_batch releases the batch lock.
            self._promote_waiting()
            self._condition.notify_all()

    def wait_idle(self) -> None:
        with self._condition:
            self._condition.wait_for(lambda: not self._requests and not self._running)

    def snapshot(self) -> dict[str, int]:
        with self._condition:
            return {
                "running": self._running_count(),
                "waiting": self._waiting_count(),
                "scheduled_rounds": self._rounds,
            }

    def _running_count(self) -> int:
        return sum(not state.waiting for state in self._requests.values())

    def _waiting_count(self) -> int:
        return sum(state.waiting for state in self._requests.values())

    def _promote_waiting(self) -> None:
        slots = self.max_num_seqs - self._running_count()
        for request_id, state in self._requests.items():
            if slots <= 0:
                break
            if state.waiting:
                state.phase = "prefill"
                if not self._ready:
                    self._ready_since = time.monotonic()
                self._ready.append(request_id)
                slots -= 1
