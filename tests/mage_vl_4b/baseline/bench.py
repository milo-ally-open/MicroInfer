"""Transformers 基线 Bench：测量固定图片请求生成的延迟与吞吐。"""

import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig, KernelConfig

root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(root / "src"))
from inference.models.mage_vl_4b.architecture.image_encoder import _resize_image

config_path = root / "tests/mage_vl_4b/baseline/profiles/inference.yml"
config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
model_settings = config["model"]
model_path = Path(os.path.expandvars(os.path.expanduser(config["runtime"]["local_model_path"])))
model_path = (root / model_path).resolve() if not model_path.is_absolute() else model_path.resolve()
revision, alias = str(model_settings["revision"]), "mage-vl-4b"
prompt, max_new_tokens, warmups, iterations = "请简要描述图片中的主要内容。", 2, 1, 3
image_path = root / "test.jpg"
cuda = torch.cuda.is_available()
dtype = getattr(torch, str(model_settings["dtype"]))

print("================= baseline =================")
reference = json.loads((root / "artifacts/mage_vl_4b/baseline/test.json").read_text(encoding="utf-8"))["result"]["generated_token_ids"]
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
elif quant.get("method", "none") not in {"none", "prequantized"}:
    raise ValueError(f"不支持的量化方法：{quant['method']}")
processor_kwargs = {"trust_remote_code": True, "revision": revision, "local_files_only": True}
if config.get("processor", {}).get("use_fast") is not None: processor_kwargs["use_fast"] = bool(config["processor"]["use_fast"])
processor = AutoProcessor.from_pretrained(model_path, **processor_kwargs)
model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs).eval()
image = Image.open(image_path).convert("RGB")
image = _resize_image(image, int(config["api"]["max_image_side"]))
messages = [{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image"}]}]
prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
encoded = processor(text=[prompt_text], images=[image], return_tensors="pt")
inputs = {key: value.to(device=model.get_input_embeddings().weight.device, dtype=model.dtype) if key in {"pixel_values", "pixel_values_videos"} and isinstance(value, torch.Tensor) and value.is_floating_point() else value.to(model.get_input_embeddings().weight.device) if isinstance(value, torch.Tensor) else value for key, value in encoded.items()}
inputs.setdefault("attention_mask", torch.ones_like(inputs["input_ids"]))
if cuda:
    for index in range(torch.cuda.device_count()): torch.cuda.reset_peak_memory_stats(index)
latencies_ms, token_counts, matches = [], [], []
for index in range(warmups + iterations):
    if cuda: torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode(): output = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens)
    if cuda: torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - started) * 1000
    token_ids = output[:, inputs["input_ids"].shape[-1]:].detach().cpu().reshape(-1).tolist()
    matches.append(token_ids == reference)
    if index >= warmups: latencies_ms.append(elapsed_ms); token_counts.append(len(token_ids))
metrics = {
    "warmup_iterations": warmups, "measured_iterations": iterations, "latency_ms_samples": latencies_ms,
    "latency_ms_median": statistics.median(latencies_ms),
    "tokens_per_second_median": statistics.median(count / (ms / 1000) for count, ms in zip(token_counts, latencies_ms)),
    "baseline_test_token_match_all_runs": all(matches),
    "peak_cuda_memory_bytes": max(torch.cuda.max_memory_allocated(index) for index in range(torch.cuda.device_count())) if cuda else 0,
    "peak_cuda_reserved_bytes": max(torch.cuda.max_memory_reserved(index) for index in range(torch.cuda.device_count())) if cuda else 0,
}
report = {
    "schema_version": 1, "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
    "model": {"name": "mage_vl_4b", "alias": alias, "repo_id": str(model_settings["id"]), "revision": revision},
    "implementation": "baseline", "task": "bench",
    "environment": {"device": "cuda" if cuda else "cpu", "gpu_name": torch.cuda.get_device_name(0) if cuda else "CPU", "torch_version": torch.__version__, "cuda_version": torch.version.cuda, "dtype": str(model.dtype)},
    "workload": {"input_type": "image", "prompt_tokens": int(inputs["input_ids"].shape[-1]), "max_new_tokens": max_new_tokens},
    "result": {"status": "passed" if all(matches) else "failed", "reference_token_match_all_runs": all(matches)}, "metrics": metrics, "operator_records": [], "trace_path": None,
}
output_path = root / "artifacts/mage_vl_4b/baseline/bench.json"
output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
print(f"latency_median_ms: {metrics['latency_ms_median']:.3f}")
print(f"throughput_median_tokens_s: {metrics['tokens_per_second_median']:.3f}")
print("json:", output_path)
if not all(matches): raise AssertionError(f"Bench 输出与 baseline Test token 不一致，记录见 {output_path}")
