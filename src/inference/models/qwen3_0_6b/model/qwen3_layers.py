"""Qwen3 decoder layer 所需的正規化與 gated MLP 元件。"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class Qwen3RMSNorm(nn.Module):
    """以 FP32 累積平方平均，再轉回輸入 dtype 的 RMSNorm。"""

    def __init__(self, hidden_size: int, eps: float, *, device: torch.device, dtype: torch.dtype) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(hidden_size, device=device, dtype=dtype))
        self.variance_epsilon = float(eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        values = hidden_states.float()
        values = values * torch.rsqrt(values.square().mean(dim=-1, keepdim=True) + self.variance_epsilon)
        return self.weight * values.to(input_dtype)


class Qwen3MLP(nn.Module):
    """以 SiLU(gate) * up 執行 Qwen3 gated MLP。"""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False, device=device, dtype=dtype)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False, device=device, dtype=dtype)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False, device=device, dtype=dtype)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))

