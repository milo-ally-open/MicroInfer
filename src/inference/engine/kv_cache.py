"""管理请求 KV 快取的逻辑 token block 预留与释放。"""

from __future__ import annotations

import threading
from dataclasses import dataclass


@dataclass(frozen=True)
class KVCacheLease:
    """单一请求持有的 block table 与 KV 记忆体估算。"""

    request_id: str
    block_ids: tuple[int, ...]
    reserved_tokens: int
    estimated_bytes: int


class KVCacheManager:
    """分配固定 block ID；Qwen Worker 将同一 ID 对应至预先配置的实体 GPU pages。

    Manager 负责容量准入与租约生命周期，实体 tensor pool 由 GPU Worker 持有；
    attention 会收集 page 组成批次 K/V 张量，不代表使用 PagedAttention kernel。
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
        self._leases: dict[str, KVCacheLease] = {}
        self._lock = threading.Lock()

    def reserve(self, request_id: str, *, prompt_tokens: int, max_output_tokens: int) -> KVCacheLease | None:
        if not request_id:
            raise ValueError("Cache request id must not be empty.")
        if prompt_tokens < 1 or max_output_tokens < 1:
            raise ValueError("Prompt and output token counts must be positive.")
        token_count = prompt_tokens + max_output_tokens
        block_count = (token_count + self.block_size_tokens - 1) // self.block_size_tokens
        with self._lock:
            if request_id in self._leases:
                raise ValueError(f"KV cache request already exists: {request_id}")
            if block_count > len(self._free):
                return None
            block_ids = tuple(self._free[:block_count])
            del self._free[:block_count]
            lease = KVCacheLease(
                request_id=request_id,
                block_ids=block_ids,
                reserved_tokens=block_count * self.block_size_tokens,
                # 实际容量以整个 block table 预留量估算，不以未对齐的 token 需求低估。
                estimated_bytes=block_count * self.block_size_tokens * self.bytes_per_token,
            )
            self._leases[request_id] = lease
            return lease

    def release(self, request_id: str) -> None:
        with self._lock:
            lease = self._leases.pop(request_id, None)
            if lease is not None:
                self._free.extend(lease.block_ids)
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
            used_blocks = sum(len(lease.block_ids) for lease in self._leases.values())
            reserved_tokens = sum(lease.reserved_tokens for lease in self._leases.values())
            return {
                "capacity_tokens": self.capacity_tokens,
                "reserved_tokens": reserved_tokens,
                "used_blocks": used_blocks,
                "capacity_blocks": len(self._free) + used_blocks,
                "estimated_bytes": sum(lease.estimated_bytes for lease in self._leases.values()),
            }
