#!/usr/bin/env python3
"""Benchmark the pinned Qwen3-VL-2B Hugging Face image-chat baseline."""

from pathlib import Path
import json
import statistics
import time

import torch
import yaml
from PIL import Image
import transformers
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

TEST_CONFIG = Path(__file__).resolve().with_name("test.yml")
ROOT = TEST_CONFIG.parents[2]
OUTPUT_PATH = ROOT / "artifacts/qwen3-vl-2b-transformers-benchmark.json"
settings = yaml.safe_load(TEST_CONFIG.read_text(encoding="utf-8"))
snapshot = (TEST_CONFIG.parent / settings["snapshot"]).resolve()
image_path = (TEST_CONFIG.parent / settings["image"]).resolve()
revision = str(settings["revision"])
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dtype = getattr(torch, settings["dtype"])
processor = AutoProcessor.from_pretrained(snapshot, revision=revision, local_files_only=True)
model = Qwen3VLForConditionalGeneration.from_pretrained(
    snapshot,
    revision=revision,
    local_files_only=True,
    use_safetensors=True,
    dtype=dtype,
    device_map=0 if device.type == "cuda" else "cpu",
    attn_implementation=settings["attention"],
).eval()
messages = [{
    "role": "user",
    "content": [
        {"type": "image", "image": Image.open(image_path).convert("RGB")},
        {"type": "text", "text": settings["prompt"]},
    ],
}]
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt",
)
model_inputs = {}
for name, value in inputs.items():
    if isinstance(value, torch.Tensor):
        if name in {"pixel_values", "pixel_values_videos"} and value.is_floating_point():
            model_inputs[name] = value.to(device=model.device, dtype=model.dtype)
        else:
            model_inputs[name] = value.to(model.device)
    else:
        model_inputs[name] = value
model_inputs.pop("token_type_ids", None)
model_inputs.setdefault("attention_mask", torch.ones_like(model_inputs["input_ids"]))
max_new_tokens = int(settings["max_new_tokens"])
warmups = int(settings["warmup_requests"])
iterations = int(settings["iterations"])
for _ in range(warmups):
    with torch.inference_mode():
        model.generate(**model_inputs, max_new_tokens=max_new_tokens, do_sample=False)
if device.type == "cuda":
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)

latencies = []
completion_tokens = []
for _ in range(iterations):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode():
        output_ids = model.generate(**model_inputs, max_new_tokens=max_new_tokens, do_sample=False)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    latencies.append(time.perf_counter() - started)
    completion_tokens.append(int(output_ids.shape[-1] - model_inputs["input_ids"].shape[-1]))

assert completion_tokens and min(completion_tokens) > 0
report = {
    "model_alias": settings["model_alias"],
    "repo_id": settings["repo_id"],
    "revision": revision,
    "implementation": "transformers",
    "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
    "torch_version": torch.__version__,
    "transformers_version": transformers.__version__,
    "dtype": str(model.dtype),
    "image_path": str(image_path),
    "input_tokens": int(model_inputs["input_ids"].shape[-1]),
    "max_new_tokens": max_new_tokens,
    "warmup_requests": warmups,
    "iterations": iterations,
    "completion_tokens_per_run": completion_tokens,
    "latency_seconds_median": statistics.median(latencies),
    "tokens_per_second_median": statistics.median(
        tokens / elapsed for tokens, elapsed in zip(completion_tokens, latencies, strict=True)
    ),
    "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
    "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0,
}
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(report, ensure_ascii=False, indent=2))
print("benchmark_json:", OUTPUT_PATH)
