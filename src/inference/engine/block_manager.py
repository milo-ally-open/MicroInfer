"""管理 Engine 侧 KV block 所有权与容量。"""

from __future__ import annotations

import threading
from dataclasses import dataclass


@dataclass(frozen=True)
class KVCacheLease:
    """请求当前持有的 block table 与容量估算。"""

    request_id: str
    block_ids: tuple[int, ...]
    reserved_tokens: int
    estimated_bytes: int


@dataclass
class _LeaseState:
    max_tokens: int
    block_ids: list[int]


class BlockManager:
    """分配逻辑物理 block ID；实体 KV tensor 由 Worker 的 page pool 持有。

    `reserve()` 默认保留完整请求上限。API 也支持较小的 `initial_tokens`
    与 `ensure_capacity()`，但当前线上 Engine 仍全量预留，避免 KV 不足时
    多个请求持有部分租约并互相等待；按需扩容策略尚未接入 Scheduler 准入。
    """

    def __init__(self, *, block_size_tokens: int, capacity_tokens: int, bytes_per_token: int = 0) -> None:
        if block_size_tokens < 1 or capacity_tokens < block_size_tokens:
            raise ValueError("KV cache capacity must contain at least one full block.")
        if capacity_tokens % block_size_tokens:
            raise ValueError("KV cache capacity_tokens must be divisible by block_size_tokens.")
        if bytes_per_token < 0:
            raise ValueError("bytes_per_token must not be negative.")
        self.block_size_tokens = block_size_tokens
        self.capacity_tokens = capacity_tokens
        self.bytes_per_token = bytes_per_token
        self._free = list(range(capacity_tokens // block_size_tokens))
        self._leases: dict[str, _LeaseState] = {}
        self._lock = threading.Lock()

    def reserve(
        self,
        request_id: str,
        *,
        prompt_tokens: int,
        max_output_tokens: int,
        initial_tokens: int | None = None,
    ) -> KVCacheLease | None:
        if not request_id:
            raise ValueError("Cache request id must not be empty.")
        if prompt_tokens < 1 or max_output_tokens < 1:
            raise ValueError("Prompt and output token counts must be positive.")
        max_tokens = prompt_tokens + max_output_tokens
        initial = max_tokens if initial_tokens is None else int(initial_tokens)
        if not prompt_tokens <= initial <= max_tokens:
            raise ValueError("Initial KV reservation must cover the prompt and fit the request limit.")
        block_count = self._blocks_for(initial)
        with self._lock:
            if request_id in self._leases:
                raise ValueError(f"KV cache request already exists: {request_id}")
            if block_count > len(self._free):
                return None
            block_ids = self._free[:block_count]
            del self._free[:block_count]
            self._leases[request_id] = _LeaseState(max_tokens, list(block_ids))
            return self._lease(request_id, self._leases[request_id])

    def ensure_capacity(self, request_id: str, required_tokens: int) -> tuple[int, ...] | None:
        """确保序列最多可写入 required_tokens 个 KV；不足时不修改租约。"""
        if required_tokens < 1:
            raise ValueError("Required KV token capacity must be positive.")
        with self._lock:
            state = self._leases.get(request_id)
            if state is None:
                raise RuntimeError(f"No KV block lease exists for request {request_id}.")
            if required_tokens > state.max_tokens:
                raise ValueError(
                    f"KV request {request_id} needs {required_tokens} tokens, "
                    f"above its configured maximum {state.max_tokens}."
                )
            required_blocks = self._blocks_for(required_tokens)
            additional = required_blocks - len(state.block_ids)
            if additional <= 0:
                return tuple(state.block_ids)
            if additional > len(self._free):
                return None
            state.block_ids.extend(self._free[:additional])
            del self._free[:additional]
            return tuple(state.block_ids)

    def get(self, request_id: str) -> KVCacheLease:
        with self._lock:
            state = self._leases.get(request_id)
            if state is None:
                raise KeyError(request_id)
            return self._lease(request_id, state)

    def release(self, request_id: str) -> None:
        with self._lock:
            state = self._leases.pop(request_id, None)
            if state is not None:
                self._free.extend(state.block_ids)
                self._free.sort()

    def set_bytes_per_token(self, value: int) -> None:
        if value < 0:
            raise ValueError("bytes_per_token must not be negative.")
        with self._lock:
            if self._leases:
                raise RuntimeError("KV cache profile cannot change while requests hold reservations.")
            self.bytes_per_token = value

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            used_blocks = sum(len(state.block_ids) for state in self._leases.values())
            reserved_tokens = used_blocks * self.block_size_tokens
            return {
                "capacity_tokens": self.capacity_tokens,
                "reserved_tokens": reserved_tokens,
                "used_blocks": used_blocks,
                "capacity_blocks": len(self._free) + used_blocks,
                "estimated_bytes": reserved_tokens * self.bytes_per_token,
            }

    def _blocks_for(self, token_count: int) -> int:
        return (token_count + self.block_size_tokens - 1) // self.block_size_tokens

    def _lease(self, request_id: str, state: _LeaseState) -> KVCacheLease:
        reserved_tokens = len(state.block_ids) * self.block_size_tokens
        return KVCacheLease(
            request_id=request_id,
            block_ids=tuple(state.block_ids),
            reserved_tokens=reserved_tokens,
            estimated_bytes=reserved_tokens * self.bytes_per_token,
        )
