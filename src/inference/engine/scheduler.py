"""以等待／執行佇列和 token 預算規劃線上推論批次。"""

from __future__ import annotations

import threading
import time
from collections import deque

from engine.batch_planner import SequenceProgress, plan_batch
from engine.contracts import ExecutionPhase, ScheduledBatch, ScheduledWorkItem, WorkerResult


class SchedulerFullError(RuntimeError):
    """等待佇列及執行序列均已達上限。"""


class _RequestState:
    def __init__(self, phase: str, prompt_tokens: int, max_output_tokens: int) -> None:
        self.phase = phase
        self.prompt_tokens = prompt_tokens
        self.max_output_tokens = max_output_tokens
        self.generated_tokens = 0


class TokenScheduler:
    """以 batch planner 從等待／執行佇列選取同階段的 Runner 工作。"""

    def __init__(
        self,
        *,
        max_num_seqs: int,
        max_waiting_requests: int,
        max_num_scheduled_tokens: int,
        batch_wait_seconds: float = 0.01,
    ) -> None:
        if (
            max_num_seqs < 1 or max_waiting_requests < 0 or max_num_scheduled_tokens < 1
            or batch_wait_seconds < 0
        ):
            raise ValueError("Scheduler limits must be positive (waiting requests may be zero).")
        self.max_num_seqs = max_num_seqs
        self.max_waiting_requests = max_waiting_requests
        self.max_num_scheduled_tokens = max_num_scheduled_tokens
        self.batch_wait_seconds = batch_wait_seconds
        self._condition = threading.Condition()
        self._requests: dict[str, _RequestState] = {}
        self._ready: deque[str] = deque()
        self._ready_since: float | None = None
        self._inflight: set[str] = set()
        self._running = False
        self._rounds = 0

    def add_request(self, request_id: str, *, prompt_tokens: int, max_output_tokens: int) -> None:
        if prompt_tokens < 1 or max_output_tokens < 1:
            raise ValueError("Prompt and output token counts must be positive.")
        if prompt_tokens > self.max_num_scheduled_tokens:
            raise ValueError(
                f"Prompt has {prompt_tokens} tokens, exceeding the per-turn budget "
                f"{self.max_num_scheduled_tokens}; chunked prefill is not enabled."
            )
        with self._condition:
            if request_id in self._requests:
                raise ValueError(f"Duplicate scheduler request id: {request_id}")
            running_count = self._running_count()
            waiting_count = self._waiting_count()
            if running_count >= self.max_num_seqs and waiting_count >= self.max_waiting_requests:
                raise SchedulerFullError("The inference scheduler is full.")
            phase = "prefill" if running_count < self.max_num_seqs else "waiting_prefill"
            self._requests[request_id] = _RequestState(phase, prompt_tokens, max_output_tokens)
            if phase == "prefill":
                if not self._ready:
                    self._ready_since = time.monotonic()
                self._ready.append(request_id)
            self._condition.notify_all()

    def add_atomic_request(self, request_id: str, *, token_budget: int) -> None:
        """登記尚未拆分為逐 token 呼叫的 Live 工作。"""
        if token_budget < 1 or token_budget > self.max_num_scheduled_tokens:
            raise ValueError("Atomic request token_budget must fit the per-turn scheduler budget.")
        with self._condition:
            if request_id in self._requests:
                raise ValueError(f"Duplicate scheduler request id: {request_id}")
            running_count = self._running_count()
            if running_count >= self.max_num_seqs and self._waiting_count() >= self.max_waiting_requests:
                raise SchedulerFullError("The inference scheduler is full.")
            phase = "atomic" if running_count < self.max_num_seqs else "waiting_atomic"
            self._requests[request_id] = _RequestState(phase, token_budget, token_budget)
            if phase == "atomic":
                if not self._ready:
                    self._ready_since = time.monotonic()
                self._ready.append(request_id)
            self._condition.notify_all()

    def schedule_batch(self) -> ScheduledBatch | None:
        """以 batch planner 建立一個同階段批次，暫不切分 prompt prefill。"""
        with self._condition:
            if self._running or not self._ready:
                return None
            if self.batch_wait_seconds > 0 and self._ready_since is not None:
                remaining = self.batch_wait_seconds - (time.monotonic() - self._ready_since)
                if remaining > 0:
                    self._condition.wait(timeout=remaining)
                    if self._running or not self._ready:
                        return None
            first = self._requests.get(self._ready[0])
            if first is None:
                self._ready.popleft()
                return None
            phase = first.phase
            if phase not in {"prefill", "decode"}:
                return None
            eligible = [
                request_id for request_id in self._ready
                if request_id in self._requests and self._requests[request_id].phase == phase
            ]
            progress = [
                SequenceProgress(
                    request_id=request_id,
                    phase=phase,
                    prompt_tokens_total=self._requests[request_id].prompt_tokens,
                    prompt_tokens_processed=(
                        0 if phase == "prefill" else self._requests[request_id].prompt_tokens
                    ),
                    generated_tokens=self._requests[request_id].generated_tokens,
                )
                for request_id in eligible
            ]
            plan = plan_batch(
                progress,
                max_num_seqs=self.max_num_seqs,
                max_num_scheduled_tokens=self.max_num_scheduled_tokens,
            )
            # 若 token 預算只容許部分 prefill，延後該 request，避免假裝 Runner 已支援 chunked prefill。
            selected_ids = {
                item.request_id
                for item in plan.sequences
                if item.phase == "decode"
                or item.token_budget == self._requests[item.request_id].prompt_tokens
            }
            if not selected_ids:
                return None
            selected = [item for item in plan.sequences if item.request_id in selected_ids]
            scheduled = tuple(
                ScheduledWorkItem(
                    request_id=item.request_id,
                    phase=ExecutionPhase(item.phase),
                    token_budget=item.token_budget,
                    prompt_offset=item.prompt_offset,
                )
                for item in selected
            )
            total_tokens = sum(item.token_budget for item in scheduled)
            batch = ScheduledBatch(scheduled, total_tokens)
            self._ready = deque(request_id for request_id in self._ready if request_id not in selected_ids)
            self._ready_since = time.monotonic() if self._ready else None
            self._inflight = set(selected_ids)
            self._running = True
            self._rounds += 1
            return batch

    def complete_batch(self, results: tuple[WorkerResult, ...] | list[WorkerResult]) -> None:
        """套用批次結果，將未完成 request 放回 decode 佇列。"""
        result_ids = [result.request_id for result in results]
        with self._condition:
            if not self._running or len(result_ids) != len(set(result_ids)) or set(result_ids) != self._inflight:
                raise RuntimeError("Worker batch results do not match the in-flight scheduler batch.")
            self._running = False
            self._inflight.clear()
            for result in results:
                state = self._requests.get(result.request_id)
                if state is None:
                    continue
                state.generated_tokens += result.generated_tokens
                if result.finished or state.generated_tokens >= state.max_output_tokens:
                    del self._requests[result.request_id]
                else:
                    state.phase = "decode"
                    if not self._ready:
                        self._ready_since = time.monotonic()
                    self._ready.append(result.request_id)
            self._promote_waiting()
            self._condition.notify_all()

    def abort_batch(self, request_ids: set[str]) -> None:
        """Worker 批次失敗時清除本輪排程狀態，等待 Engine 統一釋放各請求資源。"""
        with self._condition:
            if not self._running or request_ids != self._inflight:
                raise RuntimeError("Only the active Worker batch can be aborted.")
            self._running = False
            self._inflight.clear()
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
        return sum(not state.phase.startswith("waiting") for state in self._requests.values())

    def _waiting_count(self) -> int:
        return sum(state.phase.startswith("waiting") for state in self._requests.values())

    def _promote_waiting(self) -> None:
        slots = self.max_num_seqs - self._running_count()
        for request_id, state in self._requests.items():
            if slots <= 0:
                break
            if state.phase.startswith("waiting_"):
                state.phase = "atomic" if state.phase == "waiting_atomic" else "prefill"
                if not self._ready:
                    self._ready_since = time.monotonic()
                self._ready.append(request_id)
                slots -= 1
