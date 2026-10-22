"""以公平輪轉方式分配推論請求的 prefill 與 decode token 預算。"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass


class SchedulerFullError(RuntimeError):
    """等待佇列及執行序列均已達上限。"""


@dataclass(frozen=True)
class ScheduledTurn:
    """Scheduler 本輪授予模型執行器的工作與 token 預算。"""

    request_id: str
    phase: str
    token_budget: int


@dataclass
class _RequestState:
    phase: str
    prompt_tokens: int
    max_output_tokens: int
    generated_tokens: int = 0


class TokenScheduler:
    """限制同時序列數，並逐輪輪轉 prefill 與單 token decode 工作。"""

    def __init__(self, *, max_num_seqs: int, max_waiting_requests: int, max_num_scheduled_tokens: int) -> None:
        if max_num_seqs < 1 or max_waiting_requests < 0 or max_num_scheduled_tokens < 1:
            raise ValueError("Scheduler limits must be positive (waiting requests may be zero).")
        self.max_num_seqs = max_num_seqs
        self.max_waiting_requests = max_waiting_requests
        self.max_num_scheduled_tokens = max_num_scheduled_tokens
        self._condition = threading.Condition()
        self._requests: dict[str, _RequestState] = {}
        self._ready: deque[str] = deque()
        self._running = False
        self._current: str | None = None
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
            running_count = sum(state.phase != "waiting" for state in self._requests.values())
            waiting_count = sum(state.phase.startswith("waiting") for state in self._requests.values())
            if running_count >= self.max_num_seqs and waiting_count >= self.max_waiting_requests:
                raise SchedulerFullError("The inference scheduler is full.")
            phase = "prefill" if running_count < self.max_num_seqs else "waiting_prefill"
            self._requests[request_id] = _RequestState(phase, prompt_tokens, max_output_tokens)
            if phase == "prefill":
                self._ready.append(request_id)
            self._condition.notify_all()

    def add_atomic_request(self, request_id: str, *, token_budget: int) -> None:
        """登記尚未拆分為逐 token 呼叫的 Live 工作；仍遵守同一公平執行序列。"""
        if token_budget < 1 or token_budget > self.max_num_scheduled_tokens:
            raise ValueError("Atomic request token_budget must fit the per-turn scheduler budget.")
        with self._condition:
            if request_id in self._requests:
                raise ValueError(f"Duplicate scheduler request id: {request_id}")
            running_count = self._running_count()
            waiting_count = sum(state.phase.startswith("waiting") for state in self._requests.values())
            if running_count >= self.max_num_seqs and waiting_count >= self.max_waiting_requests:
                raise SchedulerFullError("The inference scheduler is full.")
            phase = "atomic" if running_count < self.max_num_seqs else "waiting_atomic"
            self._requests[request_id] = _RequestState(phase, token_budget, token_budget)
            if phase == "atomic":
                self._ready.append(request_id)
            self._condition.notify_all()

    def acquire_turn(self, request_id: str) -> ScheduledTurn:
        """等待公平輪轉名額，回傳本輪要處理的 token 數。"""
        with self._condition:
            while True:
                state = self._requests.get(request_id)
                if state is None:
                    raise RuntimeError(f"Scheduler request is no longer active: {request_id}")
                if state.phase.startswith("waiting_") and self._running_count() < self.max_num_seqs:
                    state.phase = "atomic" if state.phase == "waiting_atomic" else "prefill"
                    self._ready.append(request_id)
                if not self._running and self._ready and self._ready[0] == request_id:
                    self._ready.popleft()
                    self._running = True
                    self._current = request_id
                    self._rounds += 1
                    tokens = state.prompt_tokens if state.phase == "prefill" else (
                        state.max_output_tokens if state.phase == "atomic" else 1
                    )
                    return ScheduledTurn(request_id, state.phase, tokens)
                self._condition.wait()

    def complete_turn(self, request_id: str, *, generated_tokens: int = 0, finished: bool = False) -> None:
        with self._condition:
            if not self._running or self._current != request_id:
                raise RuntimeError("Only the request holding the current scheduler turn can complete it.")
            state = self._requests[request_id]
            state.generated_tokens += generated_tokens
            self._running = False
            self._current = None
            if finished or state.generated_tokens >= state.max_output_tokens:
                del self._requests[request_id]
                self._promote_waiting()
            else:
                state.phase = "decode"
                self._ready.append(request_id)
            self._condition.notify_all()

    def cancel_request(self, request_id: str) -> None:
        with self._condition:
            if self._current == request_id:
                self._running = False
                self._current = None
            self._requests.pop(request_id, None)
            self._ready = deque(item for item in self._ready if item != request_id)
            self._promote_waiting()
            self._condition.notify_all()

    def wait_idle(self) -> None:
        with self._condition:
            self._condition.wait_for(lambda: not self._requests and not self._running)

    def snapshot(self) -> dict[str, int]:
        with self._condition:
            return {
                "running": self._running_count(),
                "waiting": sum(state.phase.startswith("waiting") for state in self._requests.values()),
                "scheduled_rounds": self._rounds,
            }

    def _running_count(self) -> int:
        return sum(not state.phase.startswith("waiting") for state in self._requests.values())

    def _promote_waiting(self) -> None:
        slots = self.max_num_seqs - self._running_count()
        for request_id, state in self._requests.items():
            if slots <= 0:
                break
            if state.phase.startswith("waiting_"):
                state.phase = "atomic" if state.phase == "waiting_atomic" else "prefill"
                self._ready.append(request_id)
                slots -= 1
