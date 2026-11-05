"""Transformers 基线 Prefill：采集整段图片 Prompt 的算子耗时与形状。"""

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(root / "src"))

import torch
import yaml
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig, KernelConfig

from inference.models.mage_vl_4b.architecture.image_encoder import _resize_image

config_path = root / "tests/mage_vl_4b/baseline/profiles/inference.yml"
config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
model_settings = config["model"]
model_path = Path(os.path.expandvars(os.path.expanduser(config["runtime"]["local_model_path"])))
model_path = (root / model_path).resolve() if not model_path.is_absolute() else model_path.resolve()
revision = str(model_settings["revision"])
alias, prompt = "mage-vl-4b", "请简要描述图片中的主要内容。"
image_path = root / "test.jpg"
cuda = torch.cuda.is_available()
dtype = getattr(torch, str(model_settings["dtype"]))
profile_mode = os.environ.get("INFERENCE_PROFILE_MODE", "torch").strip().lower()
if profile_mode not in {"torch", "external"}: raise ValueError("INFERENCE_PROFILE_MODE 只支持 torch 或 external。")
activities = [torch.profiler.ProfilerActivity.CPU] + ([torch.profiler.ProfilerActivity.CUDA] if cuda and profile_mode == "torch" else [])
trace_suffix = "-external" if profile_mode == "external" else ""

print("================= baseline =================")
quant = model_settings["quantization"] if cuda else {"method": "none"}
model_kwargs = {"trust_remote_code": True, "device_map": model_settings["device_map"] if cuda else "cpu", "revision": revision, "dtype": dtype, "local_files_only": True, "attn_implementation": model_settings["attn_implementation"], "use_safetensors": model_settings["use_safetensors"], "weights_only": model_settings["weights_only"], "offload_buffers": model_settings["offload_buffers"]}
for key in ("experts_implementation", "use_kernels", "disable_mmap"):
    if model_settings.get(key) is not None: model_kwargs[key] = model_settings[key]
if model_settings.get("kernel_config"): model_kwargs["kernel_config"] = KernelConfig(**model_settings["kernel_config"])
for key, target in (("gpu_max_memory", 0), ("cpu_max_memory", "cpu")):
    if cuda and quant.get(key): model_kwargs.setdefault("max_memory", {})[target] = str(quant[key])
if quant.get("method") == "nf4":
    model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type=str(quant.get("quant_type", "nf4")), bnb_4bit_use_double_quant=bool(quant.get("double_quant", True)), bnb_4bit_compute_dtype=getattr(torch, str(quant.get("compute_dtype", model_settings["dtype"]))))
elif quant.get("method") == "int8":
    model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True, llm_int8_enable_fp32_cpu_offload=bool(quant.get("llm_int8_enable_fp32_cpu_offload", False)), llm_int8_threshold=float(quant.get("llm_int8_threshold", 6.0)), llm_int8_has_fp16_weight=bool(quant.get("llm_int8_has_fp16_weight", False)), llm_int8_skip_modules=quant.get("llm_int8_skip_modules"))
elif quant.get("method", "none") not in {"none", "prequantized"}: raise ValueError(f"不支持的量化方法：{quant['method']}")
processor_kwargs = {"trust_remote_code": True, "revision": revision, "local_files_only": True}
if config.get("processor", {}).get("use_fast") is not None: processor_kwargs["use_fast"] = bool(config["processor"]["use_fast"])
processor = AutoProcessor.from_pretrained(model_path, **processor_kwargs)
model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs).eval()
image = Image.open(image_path).convert("RGB")
image = _resize_image(image, int(config["api"]["max_image_side"]))
prompt_text = processor.apply_chat_template([{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image"}]}], tokenize=False, add_generation_prompt=True)
encoded = processor(text=[prompt_text], images=[image], return_tensors="pt")
inputs = {key: value.to(device=model.get_input_embeddings().weight.device, dtype=model.dtype) if key in {"pixel_values", "pixel_values_videos"} and isinstance(value, torch.Tensor) and value.is_floating_point() else value.to(model.get_input_embeddings().weight.device) if isinstance(value, torch.Tensor) else value for key, value in encoded.items()}
inputs.setdefault("attention_mask", torch.ones_like(inputs["input_ids"]))
if cuda:
    for index in range(torch.cuda.device_count()): torch.cuda.reset_peak_memory_stats(index)
trace_path = root / f"artifacts/mage_vl_4b/traces/baseline-prefill{trace_suffix}.json"
trace_path.parent.mkdir(parents=True, exist_ok=True)
start_event = torch.cuda.Event(enable_timing=True) if cuda else None
end_event = torch.cuda.Event(enable_timing=True) if cuda else None
started = time.perf_counter()
with torch.profiler.profile(activities=activities, record_shapes=True, profile_memory=True, with_stack=True) as profile:
    if start_event: start_event.record()
    with torch.inference_mode(): output = model(**inputs, use_cache=True, return_dict=True)
    if end_event:
        end_event.record()
        end_event.synchronize()
prefill_ms = float(start_event.elapsed_time(end_event)) if cuda else (time.perf_counter() - started) * 1000
if output.past_key_values is None: raise RuntimeError("Transformers Prefill 没有返回 KV Cache。")
cache_tokens = int(output.past_key_values.get_seq_length())
prompt_tokens = int(inputs["input_ids"].shape[-1])
if cache_tokens != prompt_tokens:
    raise AssertionError(f"Prefill Cache 长度 {cache_tokens} 与输入长度 {prompt_tokens} 不一致。")
profile.export_chrome_trace(str(trace_path))
operators = [{"phase": "prefill", "operator": str(event.key), "input_shapes": event.input_shapes, "call_count": int(event.count), "self_device_time_total_us": float(getattr(event, "self_device_time_total", getattr(event, "self_cuda_time_total", 0)) or 0), "self_cpu_time_total_us": float(event.self_cpu_time_total), "self_device_memory_usage_bytes": int(getattr(event, "self_device_memory_usage", getattr(event, "self_cuda_memory_usage", 0)) or 0), "self_cpu_memory_usage_bytes": int(event.self_cpu_memory_usage)} for event in profile.key_averages(group_by_input_shape=True)]
operators.sort(key=lambda row: row["self_device_time_total_us"], reverse=True)
print(profile.key_averages(group_by_input_shape=True).table(sort_by="self_device_time_total" if cuda else "self_cpu_time_total", row_limit=20))
report = {
    "schema_version": 1, "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
    "model": {"name": "mage_vl_4b", "alias": alias, "repo_id": str(model_settings["id"]), "revision": revision},
    "implementation": "baseline", "task": "prefill",
    "environment": {"device": "cuda" if cuda else "cpu", "gpu_name": ", ".join(torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())) if cuda else "CPU", "torch_version": torch.__version__, "cuda_version": torch.version.cuda, "dtype": str(model.dtype), "profile_mode": profile_mode},
    "workload": {"input_type": "image", "image_width": image.width, "image_height": image.height, "prompt": prompt, "prompt_tokens": prompt_tokens},
    "result": {"status": "passed", "query_tokens": prompt_tokens, "cache_tokens_after_prefill": cache_tokens, "logits_shape": list(output.logits.shape)},
    "metrics": {"prefill_latency_ms": prefill_ms, "peak_cuda_memory_bytes": max(torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())) if cuda else 0}, "operator_records": operators, "trace_path": str(trace_path),
}
output_path = root / "artifacts/mage_vl_4b/baseline/prefill.json"
output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
print("prefill_ms:", round(prefill_ms, 3), "operator_records:", len(operators), "json:", output_path)
