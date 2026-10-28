#!/usr/bin/env python3
"""比较 Mage-VL 本地模型计算图与 Transformers 原生模型。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--config", type=Path, required=True, help="明确指定测试 YAML。")
args = parser.parse_args()

config_path = args.config.resolve()
settings = yaml.safe_load(config_path.read_text(encoding="utf-8"))
if not isinstance(settings, dict):
    raise ValueError("测试 YAML 最上层必须是 mapping。")
snapshot = (config_path.parent / settings["snapshot"]).resolve()
if not (snapshot / "config.json").is_file():
    raise FileNotFoundError(f"本地模型配置不存在：{snapshot / 'config.json'}")

from inference.models.mage_vl_4b.model.configuration_mage_vl import MageVLConfig
from inference.models.mage_vl_4b.model.conditional_generation import MageVLForConditionalGeneration

config = MageVLConfig.from_pretrained(snapshot, local_files_only=True)
baseline_config = MageVLConfig.from_pretrained(snapshot, local_files_only=True)
baseline_config._name_or_path = str(snapshot)
with torch.device("meta"):
    engine_model = MageVLForConditionalGeneration(config)
    baseline_model = AutoModelForCausalLM.from_config(baseline_config, trust_remote_code=True)
expected_keys = set(json.loads((snapshot / "model.safetensors.index.json").read_text(encoding="utf-8"))["weight_map"])
engine_keys = set(engine_model.state_dict())
if expected_keys != engine_keys:
    raise AssertionError(
        f"Engine checkpoint key mismatch: missing={sorted(expected_keys - engine_keys)[:5]}, "
        f"unexpected={sorted(engine_keys - expected_keys)[:5]}"
    )
if any(isinstance(module, torch.nn.MultiheadAttention) for module in engine_model.modules()):
    raise AssertionError("Mage-VL engine must not instantiate torch.nn.MultiheadAttention.")

print("==========engine===========")
print(f"Mage-VL 本地模型图（meta device，不加载权重）: {type(engine_model).__module__}.{type(engine_model).__name__}")
for name, module in engine_model.named_modules():
    if not name or name.count(".") <= 2:
        print(f"{name or '<root>'}: {type(module).__module__}.{type(module).__name__}")
print("==========baseline==========")
print(f"Transformers 原生模型（meta device，不加载权重）: {type(baseline_model).__module__}.{type(baseline_model).__name__}")
for name, module in baseline_model.named_modules():
    if not name or name.count(".") <= 2:
        print(f"{name or '<root>'}: {type(module).__module__}.{type(module).__name__}")
print(f"checkpoint tensors: {len(engine_keys)} keys matched")

from torch import nn
from transformers import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3Model as TransformersQwen3Model
from inference.models.mage_vl_4b.layers.multi_head_attention import MultiHeadAttention
from inference.models.mage_vl_4b.model.text_backbone import Qwen3Model

torch.manual_seed(7)
reference_attention = nn.MultiheadAttention(24, 4, batch_first=True).eval()
attention_layer = MultiHeadAttention(24, 4).eval()
attention_layer.load_state_dict(reference_attention.state_dict(), strict=True)
query = torch.randn(1, 3, 24)
key = torch.randn(1, 5, 24)
value = torch.randn(1, 5, 24)
padding = torch.tensor([[False, False, False, False, True]])
print("==========engine===========")
actual_output = attention_layer(query, key, value, key_padding_mask=padding)[0]
print(f"MultiHeadAttention output shape: {tuple(actual_output.shape)}")
print("==========baseline==========")
reference_output = reference_attention(
    query, key, value, key_padding_mask=padding, need_weights=False
)[0]
torch.testing.assert_close(actual_output, reference_output, atol=1e-6, rtol=1e-5)

config = Qwen3Config(
    vocab_size=97,
    hidden_size=64,
    intermediate_size=128,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=16,
    max_position_embeddings=128,
)
config._attn_implementation = "eager"
config.attention_query_chunk_size = 2
reference_model = TransformersQwen3Model(config).eval()
local_model = Qwen3Model(config).eval()
local_model.load_state_dict(reference_model.state_dict(), strict=True)
input_ids = torch.tensor([[4, 8, 15, 16, 23]])
attention_mask = torch.ones_like(input_ids)
print("==========engine===========")
local_prefill = local_model(
    input_ids=input_ids, attention_mask=attention_mask, use_cache=True, return_dict=True
)
print(f"Qwen3 text prefill output shape: {tuple(local_prefill.last_hidden_state.shape)}")
print("==========baseline==========")
reference_prefill = reference_model(
    input_ids=input_ids, attention_mask=attention_mask, use_cache=True, return_dict=True
)
torch.testing.assert_close(
    local_prefill.last_hidden_state,
    reference_prefill.last_hidden_state,
    atol=1e-5,
    rtol=1e-5,
)

next_ids = torch.tensor([[42]])
next_mask = torch.ones((1, 6), dtype=torch.long)
position_ids = torch.tensor([[5]])
print("==========engine===========")
local_decode = local_model(
    input_ids=next_ids,
    attention_mask=next_mask,
    position_ids=position_ids,
    past_key_values=local_prefill.past_key_values,
    use_cache=True,
    return_dict=True,
)
print(f"Qwen3 text decode output shape: {tuple(local_decode.last_hidden_state.shape)}")
print("==========baseline==========")
reference_decode = reference_model(
    input_ids=next_ids,
    attention_mask=next_mask,
    position_ids=position_ids,
    past_key_values=reference_prefill.past_key_values,
    use_cache=True,
    return_dict=True,
)
torch.testing.assert_close(
    local_decode.last_hidden_state,
    reference_decode.last_hidden_state,
    atol=1e-5,
    rtol=1e-5,
)
print("Mage-VL module tree, attention, Qwen3 prefill and decode comparisons passed.")
