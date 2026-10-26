"""以固定尺寸 GPU tensor pool 保存 request 专属实体 KV pages。"""

from __future__ import annotations

import threading
from collections.abc import Sequence

import torch


class PagedKVCache:
    """固定大小 KV page pool；Attention 以 request block table 写入并读取页面。"""

    def __init__(
        self,
        *,
        num_layers: int,
        num_blocks: int,
        block_size_tokens: int,
        num_key_value_heads: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        dimensions = (num_layers, num_blocks, block_size_tokens, num_key_value_heads, head_dim)
        if any(value < 1 for value in dimensions):
            raise ValueError("Paged KV cache dimensions must be positive.")
        self.num_layers = num_layers
        self.num_blocks = num_blocks
        self.block_size_tokens = block_size_tokens
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        device = torch.device(device)
        self.dtype = dtype
        self.storage = torch.empty(
            (num_layers, 2, num_blocks, block_size_tokens, num_key_value_heads, head_dim),
            device=device,
            dtype=dtype,
        )
        # 统一 `cuda` 与实际解析后的 `cuda:0`，避免模型与 cache 将同一装置误判为不同。
        self.device = self.storage.device
        self._owners: list[str | None] = [None] * num_blocks
        self._request_blocks: dict[str, tuple[int, ...]] = {}
        # 将实际写入长度和租约分开保存，避免错误的 block table 读取未初始化 GPU tensor。
        self._written_lengths: dict[tuple[str, int], int] = {}
        self._lock = threading.Lock()

    @property
    def bytes_allocated(self) -> int:
        return self.storage.numel() * self.storage.element_size()

    def bind(self, request_id: str, block_ids: Sequence[int]) -> None:
        blocks = tuple(int(block_id) for block_id in block_ids)
        if not request_id or not blocks or len(blocks) != len(set(blocks)):
            raise ValueError("A KV lease must have a request id and unique physical block ids.")
        if any(block_id < 0 or block_id >= self.num_blocks for block_id in blocks):
            raise ValueError("A KV lease contains a block id outside the physical pool.")
        with self._lock:
            if request_id in self._request_blocks:
                raise ValueError(f"KV pages are already bound to request {request_id}.")
            occupied = [block_id for block_id in blocks if self._owners[block_id] is not None]
            if occupied:
                raise RuntimeError(f"KV page {occupied[0]} is already owned by another request.")
            for block_id in blocks:
                self._owners[block_id] = request_id
            self._request_blocks[request_id] = blocks
            for layer_index in range(self.num_layers):
                self._written_lengths[(request_id, layer_index)] = 0

    def extend(self, request_id: str, block_ids: Sequence[int]) -> None:
        """将 Engine 新分配的 page 追加到现有请求，不移动已写入的 KV。"""
        blocks = tuple(int(block_id) for block_id in block_ids)
        if not blocks or len(blocks) != len(set(blocks)):
            raise ValueError("An extended KV block table must contain unique block IDs.")
        if any(block_id < 0 or block_id >= self.num_blocks for block_id in blocks):
            raise ValueError("An extended KV block table contains an ID outside the physical pool.")
        with self._lock:
            current = self._request_blocks.get(request_id)
            if current is None:
                raise RuntimeError(f"KV pages are not bound for request {request_id}.")
            if len(blocks) < len(current) or blocks[: len(current)] != current:
                raise RuntimeError("KV block table updates must append pages without reordering existing pages.")
            occupied = [block_id for block_id in blocks[len(current) :] if self._owners[block_id] is not None]
            if occupied:
                raise RuntimeError(f"KV page {occupied[0]} is already owned by another request.")
            for block_id in blocks[len(current) :]:
                self._owners[block_id] = request_id
            self._request_blocks[request_id] = blocks

    def release(self, request_id: str) -> None:
        with self._lock:
            blocks = self._request_blocks.pop(request_id, ())
            for block_id in blocks:
                if self._owners[block_id] == request_id:
                    self._owners[block_id] = None
            for key in tuple(self._written_lengths):
                if key[0] == request_id:
                    del self._written_lengths[key]

    def write(
        self,
        *,
        request_id: str,
        layer_index: int,
        block_table: Sequence[int],
        position_offset: int,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> int:
        """将一个 request 的 `[heads, tokens, head_dim]` K/V 写至已保留 page。"""
        if not 0 <= layer_index < self.num_layers:
            raise ValueError(f"KV layer index is outside the cache: {layer_index}.")
        if key.shape != value.shape or key.ndim != 3:
            raise ValueError("KV writes require matching [heads, tokens, head_dim] tensors.")
        if key.shape[0] != self.num_key_value_heads or key.shape[2] != self.head_dim:
            raise ValueError("KV write head dimensions do not match the allocated cache.")
        if key.device != self.device or value.device != self.device:
            raise ValueError("KV write tensors must be on the cache device.")
        if key.dtype != self.dtype or value.dtype != self.dtype:
            raise ValueError("KV write tensors must use the cache dtype.")
        token_count = int(key.shape[1])
        if position_offset < 0 or token_count < 1:
            raise ValueError("KV writes require a non-negative offset and at least one token.")
        previous_length = self._written_lengths.get((request_id, layer_index))
        if previous_length is None:
            raise RuntimeError(f"KV pages are not bound for request {request_id}.")
        if position_offset != previous_length:
            raise RuntimeError(
                f"KV writes must be contiguous for {request_id}: expected offset "
                f"{previous_length}, got {position_offset}."
            )
        positions = torch.arange(
            position_offset,
            position_offset + token_count,
            device=self.device,
            dtype=torch.long,
        )
        table = tuple(int(block_id) for block_id in block_table)
        required_blocks = (position_offset + token_count + self.block_size_tokens - 1) // self.block_size_tokens
        if required_blocks > len(table):
            raise RuntimeError("KV write exceeds the request's reserved block table.")
        self._validate_owner(request_id, table)
        block_ids = torch.tensor(table, device=self.device, dtype=torch.long)[
            positions // self.block_size_tokens
        ]
        offsets = positions % self.block_size_tokens
        self.storage[layer_index, 0, block_ids, offsets] = key.transpose(0, 1)
        self.storage[layer_index, 1, block_ids, offsets] = value.transpose(0, 1)
        written_length = position_offset + token_count
        self._written_lengths[(request_id, layer_index)] = written_length
        return written_length

    def read_batch(
        self,
        *,
        layer_index: int,
        request_ids: Sequence[str],
        block_tables: Sequence[Sequence[int]],
        lengths: Sequence[int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """依各 request 的 block table 收集并补齐批次 K/V 张量。"""
        if not (len(request_ids) == len(block_tables) == len(lengths)) or not lengths:
            raise ValueError("KV read batch metadata must have matching non-empty lengths.")
        if len(request_ids) != len(set(request_ids)):
            raise ValueError("KV read batch request IDs must be unique.")
        if not 0 <= layer_index < self.num_layers:
            raise ValueError(f"KV layer index is outside the cache: {layer_index}.")
        max_length = max(int(length) for length in lengths)
        keys = torch.zeros(
            (len(lengths), self.num_key_value_heads, max_length, self.head_dim),
            device=self.device,
            dtype=self.dtype,
        )
        values = torch.zeros_like(keys)
        for batch_index, (request_id, raw_table, raw_length) in enumerate(
            zip(request_ids, block_tables, lengths, strict=True)
        ):
            length = int(raw_length)
            table = tuple(int(block_id) for block_id in raw_table)
            required_blocks = (length + self.block_size_tokens - 1) // self.block_size_tokens
            if length < 1 or required_blocks > len(table):
                raise RuntimeError("KV read exceeds the request's reserved block table.")
            self._validate_owner(request_id, table)
            written_length = self._written_lengths.get((request_id, layer_index), 0)
            if length > written_length:
                raise RuntimeError(
                    f"KV read for {request_id} requested {length} tokens, "
                    f"but only {written_length} are initialized."
                )
            page_ids = torch.tensor(table[:required_blocks], device=self.device, dtype=torch.long)
            keys[batch_index, :, :length, :] = self.storage[layer_index, 0, page_ids].reshape(
                -1, self.num_key_value_heads, self.head_dim
            )[:length].transpose(0, 1)
            values[batch_index, :, :length, :] = self.storage[layer_index, 1, page_ids].reshape(
                -1, self.num_key_value_heads, self.head_dim
            )[:length].transpose(0, 1)
        return keys, values

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            used = sum(owner is not None for owner in self._owners)
            return {
                "physical_blocks": self.num_blocks,
                "used_blocks": used,
                "free_blocks": self.num_blocks - used,
                "block_size_tokens": self.block_size_tokens,
                "allocated_bytes": self.bytes_allocated,
            }

    def _validate_owner(self, request_id: str, block_table: Sequence[int]) -> None:
        with self._lock:
            if self._request_blocks.get(request_id) != tuple(block_table):
                raise RuntimeError(f"KV block table does not match the lease for request {request_id}.")


if __name__ == "__main__":
    test_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_device(test_device)
    cache = PagedKVCache(
        num_layers=1,
        num_blocks=3,
        block_size_tokens=2,
        num_key_value_heads=2,
        head_dim=3,
        device=test_device,
        dtype=torch.float32,
    )
    request_id = "cache-forward-check"
    block_table = (2, 0, 1)
    first_key = torch.arange(18, dtype=torch.float32).reshape(2, 3, 3)
    first_value = first_key + 100
    next_key = torch.arange(12, dtype=torch.float32).reshape(2, 2, 3) + 30
    next_value = next_key + 100
    cache.bind(request_id, block_table)
    try:
        cache.write(
            request_id=request_id,
            layer_index=0,
            block_table=block_table,
            position_offset=0,
            key=first_key,
            value=first_value,
        )
        cache.write(
            request_id=request_id,
            layer_index=0,
            block_table=block_table,
            position_offset=3,
            key=next_key,
            value=next_value,
        )
        engine_key, engine_value = cache.read_batch(
            layer_index=0,
            request_ids=(request_id,),
            block_tables=(block_table,),
            lengths=(5,),
        )
        baseline_key = torch.cat((first_key, next_key), dim=1).unsqueeze(0)
        baseline_value = torch.cat((first_value, next_value), dim=1).unsqueeze(0)
        print("==========engine===========")
        print(f"KV={engine_key[0, 0, :, 0].tolist()}")
        print("==========baseline==========")
        print(f"dense append={baseline_key[0, 0, :, 0].tolist()}")
        torch.testing.assert_close(engine_key, baseline_key, atol=0, rtol=0)
        torch.testing.assert_close(engine_value, baseline_value, atol=0, rtol=0)
    finally:
        cache.release(request_id)
    print(f"Qwen3 paged KV cache matches the dense append baseline on {test_device}.")

