"""追踪 Worker 内每个请求实际持有的模型 Cache 物件与序列长度。"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RequestCacheEntry:
    cache: Any
    sequence_length: int
    logical_block_ids: tuple[int, ...] = ()


class RequestCacheManager:
    """管理 Worker 进程内的 Cache 物件；Engine 端容量 lease 仍由 KVCacheManager 管理。"""

    def __init__(self) -> None:
        self._entries: dict[str, RequestCacheEntry] = {}
        self._lock = threading.RLock()

    def register(self, request_id: str) -> None:
        if not request_id:
            raise ValueError("A cache entry requires a request ID.")
        with self._lock:
            if request_id in self._entries:
                raise ValueError(f"Cache request already exists: {request_id}")
            self._entries[request_id] = RequestCacheEntry(cache=None, sequence_length=0)

    def get(self, request_id: str) -> Any:
        with self._lock:
            try:
                return self._entries[request_id].cache
            except KeyError as exc:
                raise RuntimeError(f"Unknown cache request: {request_id}") from exc

    def update(self, request_id: str, cache: Any, *, expected_length: int | None = None) -> int:
        if cache is None:
            raise ValueError("A populated cache entry cannot hold None.")
        length = self.sequence_length(cache)
        if length < 0:
            raise ValueError("Model Cache sequence length must not be negative.")
        if expected_length is not None and length != expected_length:
            raise RuntimeError(
                f"Model Cache length mismatch for {request_id}: expected {expected_length}, received {length}."
            )
        with self._lock:
            if request_id not in self._entries:
                raise RuntimeError(f"Unknown cache request: {request_id}")
            current = self._entries[request_id]
            if length < current.sequence_length:
                raise RuntimeError(
                    f"Model Cache sequence length moved backwards for {request_id}: "
                    f"{current.sequence_length} to {length}."
                )
            self._entries[request_id] = RequestCacheEntry(
                cache=cache,
                sequence_length=length,
                logical_block_ids=current.logical_block_ids,
            )
        return length

    def bind_blocks(self, request_id: str, block_ids: tuple[int, ...]) -> None:
        blocks = tuple(int(block_id) for block_id in block_ids)
        if not blocks or any(block_id < 0 for block_id in blocks) or len(blocks) != len(set(blocks)):
            raise ValueError("Logical KV block IDs must be non-empty, non-negative and unique.")
        with self._lock:
            try:
                current = self._entries[request_id]
            except KeyError as exc:
                raise RuntimeError(f"Unknown cache request: {request_id}") from exc
            if current.logical_block_ids and current.logical_block_ids != blocks:
                raise RuntimeError(f"Logical KV block table cannot be rebound for {request_id}.")
            self._entries[request_id] = RequestCacheEntry(
                cache=current.cache,
                sequence_length=current.sequence_length,
                logical_block_ids=blocks,
            )

    def length(self, request_id: str) -> int:
        with self._lock:
            try:
                return self._entries[request_id].sequence_length
            except KeyError as exc:
                raise RuntimeError(f"Unknown cache request: {request_id}") from exc

    def release(self, request_id: str) -> None:
        with self._lock:
            self._entries.pop(request_id, None)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            populated = [entry for entry in self._entries.values() if entry.cache is not None]
            return {
                "active_requests": len(self._entries),
                "populated_caches": len(populated),
                "cached_tokens": sum(entry.sequence_length for entry in populated),
                "logical_blocks": sum(len(entry.logical_block_ids) for entry in self._entries.values()),
            }

    @staticmethod
    def sequence_length(cache: Any) -> int:
        getter = getattr(cache, "get_seq_length", None)
        if callable(getter):
            return int(getter())
        try:
            return int(cache[0][0].shape[-2])
        except (IndexError, KeyError, TypeError) as exc:
            raise ValueError("The model Cache does not expose its sequence length.") from exc
