"""組裝 Model Runner 每輪使用的輸入張量與 KV block table。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

from engine.contracts import ScheduledWorkItem


@dataclass(frozen=True)
class InputBatch:
    """對齊單輪工作項目、輸入 token、position 與 request KV block table。"""

    request_ids: tuple[str, ...]
    input_ids: torch.Tensor
    token_lengths: tuple[int, ...]
    position_offsets: tuple[int, ...]
    block_tables: tuple[tuple[int, ...], ...]

    @classmethod
    def build(
        cls,
        work_items: Sequence[ScheduledWorkItem],
        input_sequences: Sequence[torch.Tensor],
        token_lengths: Sequence[int],
        position_offsets: Sequence[int],
        block_tables: Sequence[Sequence[int]],
        *,
        device: Any,
    ) -> InputBatch:
        item_count = len(work_items)
        sizes = (len(input_sequences), len(token_lengths), len(position_offsets), len(block_tables))
        if not item_count or any(size != item_count for size in sizes):
            raise ValueError("InputBatch fields must contain the same positive number of requests.")
        request_ids = tuple(item.request_id for item in work_items)
        if len(request_ids) != len(set(request_ids)):
            raise ValueError("InputBatch cannot contain duplicate request IDs.")
        lengths = tuple(int(length) for length in token_lengths)
        offsets = tuple(int(offset) for offset in position_offsets)
        if any(length < 1 for length in lengths) or any(offset < 0 for offset in offsets):
            raise ValueError("InputBatch token lengths must be positive and offsets non-negative.")
        if any(item.token_budget != lengths[index] for index, item in enumerate(work_items)):
            raise ValueError("InputBatch token lengths must match the scheduled token budgets.")
        if any(torch.as_tensor(tokens).numel() != lengths[index] for index, tokens in enumerate(input_sequences)):
            raise ValueError("InputBatch token sequence length does not match token_lengths.")
        tables = tuple(tuple(int(block_id) for block_id in table) for table in block_tables)
        if any(not table or any(block_id < 0 for block_id in table) for table in tables):
            raise ValueError("InputBatch block tables must contain non-negative block IDs.")

        input_ids = torch.zeros((item_count, max(lengths)), device=device, dtype=torch.long)
        for index, tokens in enumerate(input_sequences):
            input_ids[index, : lengths[index]] = torch.as_tensor(
                tokens, device=device, dtype=torch.long
            ).reshape(-1)
        return cls(
            request_ids=request_ids,
            input_ids=input_ids,
            token_lengths=lengths,
            position_offsets=offsets,
            block_tables=tables,
        )
