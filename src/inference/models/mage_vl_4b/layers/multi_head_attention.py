"""Mage-VL 可检视的标准多头注意力投影与矩阵计算。"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class MultiHeadAttention(nn.Module):
    """以 `nn.MultiheadAttention` 相容的权重名称计算多头注意力。"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        *,
        batch_first: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if embed_dim < 1 or num_heads < 1 or embed_dim % num_heads:
            raise ValueError("embed_dim must be divisible by num_heads.")
        if not math.isfinite(dropout) or not 0 <= dropout <= 1:
            raise ValueError("dropout must be finite and between zero and one.")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.batch_first = batch_first
        self.dropout = float(dropout)
        self.in_proj_weight = nn.Parameter(torch.empty(3 * embed_dim, embed_dim))
        self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim))
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.zeros_(self.in_proj_bias)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        key_padding_mask: torch.Tensor | None = None,
        need_weights: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if not self.batch_first:
            query, key, value = (item.transpose(0, 1) for item in (query, key, value))
        if query.ndim != 3 or key.ndim != 3 or value.ndim != 3:
            raise ValueError("Attention inputs must be rank-3 [batch, sequence, embedding] tensors.")
        if query.shape[0] != key.shape[0] or key.shape[:2] != value.shape[:2]:
            raise ValueError("Attention inputs must have matching batch and key/value sequence dimensions.")
        if query.shape[1] < 1 or key.shape[1] < 1:
            raise ValueError("Attention query and key sequences must not be empty.")
        if any(item.shape[-1] != self.embed_dim for item in (query, key, value)):
            raise ValueError(f"Attention input embedding dimensions must equal {self.embed_dim}.")
        batch_size, query_length, _ = query.shape
        key_length = key.shape[1]
        if key_padding_mask is not None and key_padding_mask.shape != (batch_size, key_length):
            raise ValueError("key_padding_mask must have shape [batch, key sequence].")
        if key_padding_mask is not None and key_padding_mask.dtype != torch.bool and not key_padding_mask.is_floating_point():
            raise ValueError("key_padding_mask must use boolean or floating-point values.")
        q_weight, k_weight, v_weight = self.in_proj_weight.chunk(3, dim=0)
        q_bias, k_bias, v_bias = self.in_proj_bias.chunk(3, dim=0)
        q = F.linear(query, q_weight, q_bias)
        k = F.linear(key, k_weight, k_bias)
        v = F.linear(value, v_weight, v_bias)
        q = q.view(batch_size, query_length, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, key_length, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, key_length, self.num_heads, self.head_dim).transpose(1, 2)

        # QKᵀ 先按 head_dim 缩放；Softmax 以 FP32 累积降低半精度溢位风险。
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if key_padding_mask is not None:
            mask = key_padding_mask.to(device=scores.device)
            if mask.dtype == torch.bool:
                scores = scores.masked_fill(mask[:, None, None, :], torch.finfo(scores.dtype).min)
            else:
                scores = scores + mask[:, None, None, :].to(dtype=scores.dtype)
        probabilities = torch.softmax(scores.float(), dim=-1).to(dtype=q.dtype)
        probabilities = F.dropout(probabilities, p=self.dropout, training=self.training)
        attended = torch.matmul(probabilities, v)
        attended = attended.transpose(1, 2).contiguous().view(batch_size, query_length, self.embed_dim)
        attended = self.out_proj(attended)
        if not self.batch_first:
            attended = attended.transpose(0, 1)
        weights = probabilities.mean(dim=1) if need_weights else None
        return attended, weights


if __name__ == "__main__":
    torch.manual_seed(7)
    reference = nn.MultiheadAttention(24, 4, batch_first=True).eval()
    implementation = MultiHeadAttention(24, 4).eval()
    implementation.load_state_dict(reference.state_dict(), strict=True)
    query = torch.randn(2, 3, 24)
    key = torch.randn(2, 7, 24)
    value = torch.randn(2, 7, 24)
    padding_mask = torch.tensor(
        [[False, False, False, False, False, True, True], [False, False, False, False, False, False, True]]
    )
    expected, expected_weights = reference(
        query, key, value, key_padding_mask=padding_mask, need_weights=True
    )
    actual, actual_weights = implementation(
        query, key, value, key_padding_mask=padding_mask, need_weights=True
    )
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(actual_weights, expected_weights, atol=1e-6, rtol=1e-5)
    additive_mask = torch.zeros_like(padding_mask, dtype=query.dtype).masked_fill(padding_mask, -torch.inf)
    expected_additive, _ = reference(
        query, key, value, key_padding_mask=additive_mask, need_weights=False
    )
    actual_additive, _ = implementation(
        query, key, value, key_padding_mask=additive_mask, need_weights=False
    )
    torch.testing.assert_close(actual_additive, expected_additive, atol=1e-6, rtol=1e-5)
    print("==========engine===========")
    print(f"shape={tuple(actual.shape)}, sample={actual[0, 0, :4].tolist()}")
    print("==========baseline==========")
    print(f"shape={tuple(expected.shape)}, sample={expected[0, 0, :4].tolist()}")
    print("MultiHeadAttention matches torch.nn.MultiheadAttention on CPU.")
