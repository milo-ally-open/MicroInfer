#!/usr/bin/env python3
"""比较 Mage-VL Inference API 与 Transformers 原生图片生成延迟。"""

from __future__ import annotations

import argparse
import base64
import json
import statistics
import time
from pathlib import Path
from urllib.request import Request, urlopen

import torch
import yaml
from PIL import Image
from transformers import AutoProcessor, BitsAndBytesConfig

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--config", type=Path, required=True, help="明确指定测试 YAML。")
args = parser.parse_args()

config_path = args.config.resolve()
settings = yaml.safe_load(config_path.read_text(encoding="utf-8"))
if not isinstance(settings, dict):
    raise ValueError("测试 YAML 最上层必须是 mapping。")
snapshot = (config_path.parent / settings["snapshot"]).resolve()
inference_config_path = (config_path.parent / settings["inference_config"]).resolve()
inference_settings = yaml.safe_load(inference_config_path.read_text(encoding="utf-8"))
image_path = (config_path.parent / settings["image"]).resolve()
if not (snapshot / "config.json").is_file() or not image_path.is_file():
    raise FileNotFoundError("请确认本地模型快照和 YAML 指定的图片均存在。")

from inference.models.mage_vl_4b.model.configuration_mage_vl import MageVLConfig
from inference.models.mage_vl_4b.model.conditional_generation import MageVLForConditionalGeneration

model_config = MageVLConfig.from_pretrained(snapshot, local_files_only=True)
processor = AutoProcessor.from_pretrained(snapshot, trust_remote_code=True, local_files_only=True)
dtype = getattr(torch, settings["component"]["dtype"])
cuda_available = torch.cuda.is_available()
quantization = inference_settings["model"]["quantization"] if cuda_available else {"method": "none"}
load_options = {
    "device_map": inference_settings["model"]["device_map"] if cuda_available else "cpu",
    "torch_dtype": dtype,
}
method = quantization["method"]
if method == "nf4":
    load_options["quantization_config"] = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=quantization.get("quant_type", "nf4"),
        bnb_4bit_use_double_quant=bool(quantization["double_quant"]),
        bnb_4bit_compute_dtype=getattr(torch, quantization["compute_dtype"]),
    )
elif method == "int8":
    load_options["quantization_config"] = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_enable_fp32_cpu_offload=bool(quantization["llm_int8_enable_fp32_cpu_offload"]),
        llm_int8_threshold=float(quantization["llm_int8_threshold"]),
    )
elif method not in {"none", "prequantized"}:
    raise ValueError(f"不支持的量化模式：{method}")
max_memory = quantization.get("gpu_max_memory")
cpu_memory = quantization.get("cpu_max_memory")
if cuda_available and (max_memory or cpu_memory):
    load_options["max_memory"] = {}
    if max_memory:
        load_options["max_memory"][0] = str(max_memory)
    if cpu_memory:
        load_options["max_memory"]["cpu"] = str(cpu_memory)
model = MageVLForConditionalGeneration.from_pretrained(
    snapshot,
    config=model_config,
    local_files_only=True,
    attn_implementation="eager",
    **load_options,
).eval()

with Image.open(image_path) as opened:
    image = opened.convert("RGB")
    max_side = int(inference_settings["api"].get("max_image_side", 896))
    image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
messages = [{
    "role": "user",
    "content": [{"type": "text", "text": settings["prompt"]}, {"type": "image"}],
}]
prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=[prompt], images=[image], return_tensors="pt")
input_device = model.get_input_embeddings().weight.device
inputs = {
    key: value.to(input_device) if isinstance(value, torch.Tensor) else value
    for key, value in inputs.items()
}
for key in ("pixel_values", "pixel_values_videos"):
    if key in inputs and inputs[key].is_floating_point():
        inputs[key] = inputs[key].to(dtype=dtype)

encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
request_body = {
    "model": inference_settings["model"]["id"],
    "stream": False,
    "messages": [{
        "role": "user",
        "content": [
            {"type": "text", "text": settings["prompt"]},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}},
        ],
    }],
    "max_tokens": int(settings["max_new_tokens"]),
    "temperature": 0,
}

warmup_requests = int(settings["warmup_requests"])
iterations = int(settings["iterations"])
if warmup_requests < 0 or iterations < 1:
    raise ValueError("warmup_requests 不能小于 0，iterations 必须大于 0。")
measurements = {"engine_api": [], "transformers_baseline": []}
for iteration in range(warmup_requests + iterations):
    request = Request(
        settings["inference_url"].rstrip("/") + "/v1/chat/completions",
        data=json.dumps(request_body, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    api_started = time.perf_counter()
    with urlopen(request, timeout=float(settings["timeout_seconds"])) as response:
        api_result = json.loads(response.read())
    api_elapsed = time.perf_counter() - api_started
    api_tokens = int(api_result.get("usage", {}).get("completion_tokens", 0))
    if iteration >= warmup_requests:
        measurements["engine_api"].append((api_elapsed, api_tokens))

    baseline_started = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=int(settings["max_new_tokens"]),
            do_sample=False,
        )
    if cuda_available:
        torch.cuda.synchronize()
    baseline_elapsed = time.perf_counter() - baseline_started
    baseline_tokens = int(generated.shape[-1] - inputs["input_ids"].shape[-1])
    if iteration >= warmup_requests:
        measurements["transformers_baseline"].append((baseline_elapsed, baseline_tokens))

report = {
    "model_alias": settings["model_alias"],
    "execution_device": "cuda" if cuda_available else "cpu",
    "warmup_requests": warmup_requests,
    "iterations": iterations,
}
for name, values in measurements.items():
    latencies = [item[0] for item in values]
    token_rates = [item[1] / item[0] if item[0] else 0 for item in values]
    report[name] = {
        "latency_seconds_mean": statistics.fmean(latencies),
        "latency_seconds_median": statistics.median(latencies),
        "completion_tokens_per_second_mean": statistics.fmean(token_rates),
        "completion_tokens": [item[1] for item in values],
    }
print(json.dumps(report, ensure_ascii=False, indent=2))
