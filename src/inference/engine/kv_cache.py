"""管理請求 KV 快取的邏輯 token block 預留與釋放。"""

from __future__ import annotations

from collections.abc import Iterable
import threading
from dataclasses import dataclass


@dataclass(frozen=True)
class KVCacheLease:
    """單一請求持有的邏輯 KV block 對照與記憶體估算。"""

    request_id: str
    block_ids: tuple[int, ...]
    reserved_tokens: int
    estimated_bytes: int


@dataclass(frozen=True)
class KVCacheRequest:
    """一次排程批次中單一序列所需的 cache 容量。"""

    request_id: str
    prompt_tokens: int
    max_output_tokens: int


class KVCacheManager:
    """預留固定容量的 KV token blocks，並在請求結束後回收供後續請求使用。

    此管理器負責 Engine 端容量核算；底層 Transformers Cache 張量仍由 Model Runner 配置，
    不代表已使用 PagedAttention 或支援跨請求 prefix cache。
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
        batch = self.reserve_batch([
            KVCacheRequest(request_id, prompt_tokens, max_output_tokens)
        ])
        return None if batch is None else batch[request_id]

    def reserve_batch(
        self, requests: Iterable[KVCacheRequest]
    ) -> dict[str, KVCacheLease] | None:
        """以全有或全無方式預留一個 Scheduler batch 的 block table。"""
        items = list(requests)
        if not items:
            return {}
        request_ids = [item.request_id for item in items]
        if any(not request_id for request_id in request_ids) or len(set(request_ids)) != len(request_ids):
            raise ValueError("KV cache batch request ids must be unique and non-empty.")
        blocks_by_request: list[tuple[KVCacheRequest, int, int]] = []
        total_blocks = 0
        for item in items:
            if item.prompt_tokens < 1 or item.max_output_tokens < 1:
                raise ValueError("Prompt and output token counts must be positive.")
            token_count = item.prompt_tokens + item.max_output_tokens
            block_count = (token_count + self.block_size_tokens - 1) // self.block_size_tokens
            blocks_by_request.append((item, token_count, block_count))
            total_blocks += block_count
        with self._lock:
            existing = set(request_ids).intersection(self._leases)
            if existing:
                raise ValueError(f"KV cache request already exists: {sorted(existing)[0]}")
            if total_blocks > len(self._free):
                return None
            allocated = iter(self._free[:total_blocks])
            del self._free[:total_blocks]
            leases: dict[str, KVCacheLease] = {}
            for item, token_count, block_count in blocks_by_request:
                block_ids = tuple(next(allocated) for _ in range(block_count))
                lease = KVCacheLease(
                    request_id=item.request_id,
                    block_ids=block_ids,
                    reserved_tokens=block_count * self.block_size_tokens,
                    estimated_bytes=token_count * self.bytes_per_token,
                )
                leases[item.request_id] = lease
                self._leases[item.request_id] = lease
            return leases

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
