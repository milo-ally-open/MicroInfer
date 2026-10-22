"""Qwen3 預設 RoPE 頻率與旋轉位置編碼。"""

from __future__ import annotations

import torch


def rotary_embeddings(
    position_ids: torch.Tensor,
    *,
    head_dim: int,
    theta: float,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """為一個 batch 建立 Qwen3 split-half RoPE 的 cosine 與 sine。"""
    frequency_indices = torch.arange(0, head_dim, 2, device=position_ids.device, dtype=torch.float32)
    inverse_frequency = 1.0 / (float(theta) ** (frequency_indices / head_dim))
    frequencies = position_ids.float().unsqueeze(-1) * inverse_frequency
    angles = torch.cat((frequencies, frequencies), dim=-1)
    return angles.cos().to(dtype), angles.sin().to(dtype)


def apply_rotary(
    query: torch.Tensor,
    key: torch.Tensor,
    cosine: torch.Tensor,
    sine: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """將 RoPE 套用於 `[batch, heads, sequence, head_dim]` 張量。"""

    def rotate_half(values: torch.Tensor) -> torch.Tensor:
        first, second = values.chunk(2, dim=-1)
        return torch.cat((-second, first), dim=-1)

    cosine = cosine.unsqueeze(1)
    sine = sine.unsqueeze(1)
    return (query * cosine + rotate_half(query) * sine, key * cosine + rotate_half(key) * sine)

