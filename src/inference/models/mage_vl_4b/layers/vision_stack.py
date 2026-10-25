"""Mage-VL 视觉 Transformer Encoder block 与层堆叠。"""

# 本模块沿用 microsoft/Mage-VL revision d88b153285f1633a61b2f693c59c8576693af185（Apache-2.0）实现。

from typing import Optional, Union

import torch
import torch.nn as nn


from transformers.modeling_outputs import BaseModelOutput
from ..model.configuration_mage_vl import MageVLVisionConfig
from .siglip_mlp import SiglipMLP
from .vision_attention import MageVLVisionAttention, get_norm_layer


class MageVLVisionEncoderLayer(nn.Module):
    """Mage-VL 视觉 Encoder 层，采前置正规化与明确 eager 注意力。"""

    def __init__(self, config: MageVLVisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = MageVLVisionAttention(config)
        self.layer_norm1 = get_norm_layer(config)
        self.mlp = SiglipMLP(config)
        self.layer_norm2 = get_norm_layer(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)

        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            rotary_pos_emb=rotary_pos_emb,
            output_attentions=output_attentions,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states, attn_weights) if output_attentions else (hidden_states,)
        return outputs


class MageVLVisionEncoder(nn.Module):
    def __init__(self, config: MageVLVisionConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([MageVLVisionEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        # Gradient checkpointing support
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
    ) -> Union[tuple, BaseModelOutput]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer.__call__,
                    hidden_states,
                    attention_mask,
                    rotary_pos_emb,
                    output_attentions,
                    cu_seqlens,
                    max_seqlen,
                )
            else:
                layer_outputs = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    rotary_pos_emb=rotary_pos_emb,
                    output_attentions=output_attentions,
                    cu_seqlens=cu_seqlens,
                    max_seqlen=max_seqlen,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


if __name__ == "__main__":
    test_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_device(test_device)
    import importlib.util
    import sys
    import types
    from pathlib import Path

    official_snapshot = Path(__file__).resolve().parents[5] / "checkpoints" / "microsoft-Mage-VL-d88b153"
    official_package_name = "_mage_vl_official_d88b153"
    official_package = types.ModuleType(official_package_name)
    official_package.__path__ = [str(official_snapshot)]
    sys.modules[official_package_name] = official_package
    official_name = f"{official_package_name}.modeling_mage_vl"
    official_spec = importlib.util.spec_from_file_location(official_name, official_snapshot / "modeling_mage_vl.py")
    if official_spec is None or official_spec.loader is None:
        raise RuntimeError(f"Unable to load the official Mage-VL source at {official_snapshot}.")
    official = importlib.util.module_from_spec(official_spec)
    sys.modules[official_name] = official
    official_spec.loader.exec_module(official)

    torch.manual_seed(67)
    config = MageVLVisionConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        attention_dropout=0.0,
    )
    reference_config = official.MageVLVisionConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        attention_dropout=0.0,
    )
    config._attn_implementation = "eager"
    reference_config._attn_implementation = "eager"
    engine = MageVLVisionEncoder(config).eval()
    baseline = official.MageVLVisionEncoder(reference_config).eval()
    engine.load_state_dict(baseline.state_dict(), strict=True)

    hidden_states = torch.randn(1, 8, config.hidden_size)
    grid_thw = torch.tensor([[2, 2, 2]])
    frequencies = official.VisionRotaryEmbedding(reference_config)(grid_thw)
    rotary_pos_emb = torch.cat((frequencies, frequencies), dim=-1).unsqueeze(0)
    cu_seqlens = torch.tensor([0, 4, 8], dtype=torch.int32)
    with torch.inference_mode():
        engine_output = engine(
            hidden_states,
            rotary_pos_emb=rotary_pos_emb,
            output_hidden_states=True,
            return_dict=True,
            cu_seqlens=cu_seqlens,
            max_seqlen=4,
        )
        baseline_output = baseline(
            hidden_states,
            rotary_pos_emb=rotary_pos_emb,
            output_hidden_states=True,
            return_dict=True,
            cu_seqlens=cu_seqlens,
            max_seqlen=4,
        )
    torch.testing.assert_close(
        engine_output.last_hidden_state, baseline_output.last_hidden_state, atol=1e-6, rtol=1e-6
    )
    for engine_hidden, baseline_hidden in zip(engine_output.hidden_states, baseline_output.hidden_states):
        torch.testing.assert_close(engine_hidden, baseline_hidden, atol=1e-6, rtol=1e-6)
    print("==========engine===========")
    print(f"shape={tuple(engine_output.last_hidden_state.shape)}, sample={engine_output.last_hidden_state[0, 0, :4].tolist()}")
    print("==========baseline==========")
    print(f"official Mage-VL sample={baseline_output.last_hidden_state[0, 0, :4].tolist()}")
    print(f"Mage-VL vision encoder stack and intermediate hidden states match the official source on {test_device}.")
