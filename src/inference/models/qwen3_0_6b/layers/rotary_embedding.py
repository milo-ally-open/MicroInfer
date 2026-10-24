"""Qwen3 预设 RoPE 频率与旋转位置编码。"""

from __future__ import annotations

import torch


def rotary_embeddings(
    position_ids: torch.Tensor,
    *,
    head_dim: int,
    theta: float,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """为一个 batch 建立 Qwen3 split-half RoPE 的 cosine 与 sine。"""
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
    """将 RoPE 套用于 `[batch, heads, sequence, head_dim]` 张量。"""

    def rotate_half(values: torch.Tensor) -> torch.Tensor:
        first, second = values.chunk(2, dim=-1)
        return torch.cat((-second, first), dim=-1)

    cosine = cosine.unsqueeze(1)
    sine = sine.unsqueeze(1)
    return (query * cosine + rotate_half(query) * sine, key * cosine + rotate_half(key) * sine)


if __name__ == "__main__":
    from transformers import Qwen3Config
    from transformers.models.qwen3.modeling_qwen3 import (
        Qwen3RotaryEmbedding,
        apply_rotary_pos_emb,
    )

    torch.manual_seed(43)
    config = Qwen3Config(hidden_size=64, num_attention_heads=4, head_dim=16)
    rope_theta = float(config.rope_parameters["rope_theta"])
    positions = torch.arange(6).expand(2, -1)
    query = torch.randn(2, 4, 6, 16)
    key = torch.randn(2, 2, 6, 16)
    cosine, sine = rotary_embeddings(
        positions, head_dim=16, theta=rope_theta, dtype=query.dtype
    )
    engine_output = apply_rotary(query, key, cosine, sine)
    baseline_cosine, baseline_sine = Qwen3RotaryEmbedding(config)(torch.zeros(2, 6, 64), positions)
    baseline_output = apply_rotary_pos_emb(query, key, baseline_cosine, baseline_sine)
    print("==========engine===========")
    print(f"query={engine_output[0][0, 0, 0, :4].tolist()}")
    print(f"key={engine_output[1][0, 0, 0, :4].tolist()}")
    print("==========baseline==========")
    print(f"query={baseline_output[0][0, 0, 0, :4].tolist()}")
    print(f"key={baseline_output[1][0, 0, 0, :4].tolist()}")
    torch.testing.assert_close(engine_output[0], baseline_output[0], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(engine_output[1], baseline_output[1], atol=1e-6, rtol=1e-6)
    print("Qwen3 RoPE matches the Transformers baseline on CPU.")
