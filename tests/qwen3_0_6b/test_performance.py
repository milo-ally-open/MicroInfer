#!/usr/bin/env python3
"""比较 Qwen3 Inference API 与 Transformers 原生生成延迟。"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path
from urllib.request import Request, urlopen

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
inference_config_path = (config_path.parent / settings["inference_config"]).resolve()
inference_settings = yaml.safe_load(inference_config_path.read_text(encoding="utf-8"))
if not (snapshot / "config.json").is_file():
    raise FileNotFoundError(f"本机模型快照不存在：{snapshot}")
os.environ["INFERENCE_CONFIG"] = str(inference_config_path)
os.environ["INFERENCE_NETWORK_CONFIG"] = str(inference_config_path.with_name("network.yml"))

from inference.models.qwen3_0_6b.utils.loader import load_model_config, load_tokenizer

revision = str(inference_settings["model"]["revision"])
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_config = load_model_config(snapshot, revision)
tokenizer = load_tokenizer(snapshot, revision)
model = AutoModelForCausalLM.from_pretrained(
    snapshot,
    config=model_config,
    revision=revision,
    local_files_only=True,
    torch_dtype=getattr(torch, settings["component"]["dtype"]),
    attn_implementation="eager",
).to(device).eval()

messages = [{"role": "user", "content": settings["prompt"]}]
tokenized = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
    enable_thinking=False,
)
input_ids = tokenized.input_ids if hasattr(tokenized, "input_ids") else tokenized
input_ids = input_ids.to(model.device)
max_new_tokens = int(settings["max_new_tokens"])
request_body = {
    "model": settings["model_alias"],
    "messages": messages,
    "stream": False,
    "max_tokens": max_new_tokens,
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
        generated = model.generate(input_ids, max_new_tokens=max_new_tokens, do_sample=False)
    if model.device.type == "cuda":
        torch.cuda.synchronize(model.device)
    baseline_elapsed = time.perf_counter() - baseline_started
    baseline_tokens = int(generated.shape[-1] - input_ids.shape[-1])
    if iteration >= warmup_requests:
        measurements["transformers_baseline"].append((baseline_elapsed, baseline_tokens))

report = {
    "model_alias": settings["model_alias"],
    "revision": revision,
    "execution_device": str(device),
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
