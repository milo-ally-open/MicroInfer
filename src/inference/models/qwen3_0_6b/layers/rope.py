"""Qwen3 使用的 split-half 旋转位置编码（RoPE）。"""

from __future__ import annotations

import math

import torch
from torch import nn


class RoPE(nn.Module):
    """对 `[batch, heads, sequence, head_size]` 的 Query 和 Key 应用 RoPE。

    位置缓存只保存半维度的 cos 和 sin。模型若先在 `meta` device 创建空权重，
    缓存会等到第一次真实前向时再按输入设备创建。
    """

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_pos_emb: int,
        base: float = 10_000.0,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        if head_size < 1 or rotary_dim < 1:
            raise ValueError("head_size 和 rotary_dim 必须是正整数。")
        if rotary_dim % 2:
            raise ValueError("rotary_dim 必须是偶数。")
        if rotary_dim > head_size:
            raise ValueError("rotary_dim 不能大于 head_size。")
        if max_pos_emb < 1:
            raise ValueError("max_pos_emb 必须是正整数。")
        if not math.isfinite(base) or base <= 0:
            raise ValueError("RoPE base 必须是有限的正数。")
        if dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            raise ValueError("RoPE dtype 必须是浮点类型。")

        self.head_size = int(head_size)
        self.rotary_dim = int(rotary_dim)
        self.half = self.rotary_dim // 2
        self.max_pos_emb = int(max_pos_emb)
        self.base = float(base)
        self.cache_dtype = dtype

        # 非持久化缓存不会写进模型 checkpoint；meta 模型则延迟到实际前向再建缓存。
        self.register_buffer("cos_sin_cache", None, persistent=False)
        if device is not None and torch.device(device).type != "meta":
            self._build_cache(torch.device(device))

    def _build_cache(self, device: torch.device) -> None:
        """以 FP32 计算角度，再转换为模型精度，减少低精度频率误差。"""
        positions = torch.arange(self.max_pos_emb, device=device, dtype=torch.float32)
        frequency_indices = torch.arange(0, self.rotary_dim, 2, device=device, dtype=torch.float32)
        inverse_frequencies = self.base ** (-frequency_indices / self.rotary_dim)
        angles = torch.outer(positions, inverse_frequencies)
        self.cos_sin_cache = torch.cat((angles.cos(), angles.sin()), dim=-1).to(self.cache_dtype)

    def forward(
        self,
        pos: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """按位置旋转 Query/Key，并保留 `rotary_dim` 以外的 head 通道。"""
        if query.ndim != 4 or key.ndim != 4:
            raise ValueError("query 和 key 必须是 [batch, heads, sequence, head_size] 四维张量。")
        if query.shape[0] != key.shape[0] or query.shape[2] != key.shape[2]:
            raise ValueError("query 与 key 的 batch 和 sequence 维度必须一致。")
        if query.shape[-1] != self.head_size or key.shape[-1] != self.head_size:
            raise ValueError("query/key 的最后一维必须等于初始化时的 head_size。")
        if pos.ndim not in (1, 2):
            raise ValueError("pos 必须是一维 [sequence] 或二维 [batch, sequence] 张量。")
        if pos.shape[-1] != query.shape[2]:
            raise ValueError("pos 的序列长度必须与 query/key 的序列长度一致。")
        if pos.ndim == 2 and pos.shape[0] != query.shape[0]:
            raise ValueError("二维 pos 的 batch 大小必须与 query/key 一致。")

        # `.to()` 在位置张量已位于模型设备时不会复制数据；第一次调用时惰性创建缓存。
        positions = pos.to(device=query.device, dtype=torch.long)
        if self.cos_sin_cache is None or self.cos_sin_cache.device != query.device:
            self._build_cache(query.device)
        cos_sin = self.cos_sin_cache[positions]
        if positions.ndim == 1:
            cos_sin = cos_sin.unsqueeze(0)

        # [batch, sequence, half] -> [batch, 1, sequence, half]，广播到不同数量的 Q/K heads。
        cosine = cos_sin[..., : self.half].unsqueeze(1)
        sine = cos_sin[..., self.half :].unsqueeze(1)

        def rotate(values: torch.Tensor) -> torch.Tensor:
            values_rot = values[..., : self.rotary_dim]
            first_half = values_rot[..., : self.half]
            second_half = values_rot[..., self.half :]
            cos = cosine.to(dtype=values.dtype)
            sin = sine.to(dtype=values.dtype)
            rotated = torch.cat(
                (first_half * cos - second_half * sin, first_half * sin + second_half * cos),
                dim=-1,
            )
            if self.rotary_dim == self.head_size:
                return rotated
            return torch.cat((rotated, values[..., self.rotary_dim :]), dim=-1)

        return rotate(query), rotate(key)


if __name__ == "__main__":
    # 与 Transformers 官方 Qwen3 RoPE 对照，默认使用 CUDA，有 CUDA 时不降级到 CPU。
    from transformers import Qwen3Config
    from transformers.models.qwen3.modeling_qwen3 import (
        Qwen3RotaryEmbedding,
        apply_rotary_pos_emb,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(43)
    config = Qwen3Config(
        vocab_size=97,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=64,
    )
    rope_theta = float(config.rope_parameters["rope_theta"])
    positions = torch.arange(6, device=device).expand(2, -1)
    query = torch.randn(2, 4, 6, 16, device=device)
    key = torch.randn(2, 2, 6, 16, device=device)
    rope = RoPE(
        head_size=16,
        rotary_dim=16,
        max_pos_emb=config.max_position_embeddings,
        base=rope_theta,
        dtype=query.dtype,
        device=device,
    )

    actual_query, actual_key = rope(positions, query, key)
    baseline_cosine, baseline_sine = Qwen3RotaryEmbedding(config).to(device)(
        torch.zeros(2, 6, config.hidden_size, device=device), positions
    )
    expected_query, expected_key = apply_rotary_pos_emb(
        query, key, baseline_cosine, baseline_sine
    )
    torch.testing.assert_close(actual_query, expected_query, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(actual_key, expected_key, atol=1e-6, rtol=1e-6)
    print(f"Qwen3 RoPE 前向与 Transformers 基线一致：device={device}")
