"""以固定尺寸 GPU tensor pool 保存 request 專屬實體 KV pages。"""

from __future__ import annotations

import threading
from collections.abc import Sequence

import torch


class PagedKVCache:
    """固定大小 KV page pool；Attention 以 request block table 寫入並讀取頁面。"""

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
        self.device = device
        self.dtype = dtype
        self.storage = torch.empty(
            (num_layers, 2, num_blocks, block_size_tokens, num_key_value_heads, head_dim),
            device=device,
            dtype=dtype,
        )
        self._owners: list[str | None] = [None] * num_blocks
        self._request_blocks: dict[str, tuple[int, ...]] = {}
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

    def release(self, request_id: str) -> None:
        with self._lock:
            blocks = self._request_blocks.pop(request_id, ())
            for block_id in blocks:
                if self._owners[block_id] == request_id:
                    self._owners[block_id] = None

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
        """將一個 request 的 `[heads, tokens, head_dim]` K/V 寫至已保留 page。"""
        if key.shape != value.shape or key.ndim != 3:
            raise ValueError("KV writes require matching [heads, tokens, head_dim] tensors.")
        token_count = int(key.shape[1])
        positions = torch.arange(
            position_offset,
            position_offset + token_count,
            device=self.device,
            dtype=torch.long,
        )
        table = tuple(int(block_id) for block_id in block_table)
        required_blocks = (position_offset + token_count + self.block_size_tokens - 1) // self.block_size_tokens
        if position_offset < 0 or required_blocks > len(table):
            raise RuntimeError("KV write exceeds the request's reserved block table.")
        self._validate_owner(request_id, table)
        block_ids = torch.tensor(table, device=self.device, dtype=torch.long)[
            positions // self.block_size_tokens
        ]
        offsets = positions % self.block_size_tokens
        self.storage[layer_index, 0, block_ids, offsets] = key.transpose(0, 1)
        self.storage[layer_index, 1, block_ids, offsets] = value.transpose(0, 1)
        return position_offset + token_count

    def read_batch(
        self,
        *,
        layer_index: int,
        request_ids: Sequence[str],
        block_tables: Sequence[Sequence[int]],
        lengths: Sequence[int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """依各 request 的 block table 組成 SDPA 使用的 padded K/V batch。"""
        if not (len(request_ids) == len(block_tables) == len(lengths)) or not lengths:
            raise ValueError("KV read batch metadata must have matching non-empty lengths.")
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

