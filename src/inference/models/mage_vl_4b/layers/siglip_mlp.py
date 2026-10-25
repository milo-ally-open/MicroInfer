"""Mage-VL 视觉塔使用的 SigLIP 前馈层。"""

from __future__ import annotations

import torch
from torch import nn
from transformers.activations import ACT2FN


class SiglipMLP(nn.Module):
    """保留 SigLIP checkpoint 权重名称的两层 MLP。"""

    def __init__(self, config) -> None:
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.activation = ACT2FN[str(config.hidden_act)]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.activation(self.fc1(hidden_states)))


if __name__ == "__main__":
    test_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_device(test_device)
    from transformers.models.siglip2.configuration_siglip2 import Siglip2VisionConfig
    from transformers.models.siglip2.modeling_siglip2 import Siglip2MLP

    torch.manual_seed(47)
    config = Siglip2VisionConfig(hidden_size=32, intermediate_size=64, hidden_act="gelu")
    baseline = Siglip2MLP(config).eval()
    engine = SiglipMLP(config).eval()
    engine.load_state_dict(baseline.state_dict(), strict=True)
    hidden_states = torch.randn(2, 5, config.hidden_size)
    with torch.inference_mode():
        engine_output = engine(hidden_states)
        baseline_output = baseline(hidden_states)
    print("==========engine===========")
    print(f"shape={tuple(engine_output.shape)}, sample={engine_output[0, 0, :4].tolist()}")
    print("==========baseline==========")
    print(f"shape={tuple(baseline_output.shape)}, sample={baseline_output[0, 0, :4].tolist()}")
    torch.testing.assert_close(engine_output, baseline_output, atol=1e-6, rtol=1e-6)
    print(f"Mage-VL SigLIP MLP matches the Transformers baseline on {test_device}.")
