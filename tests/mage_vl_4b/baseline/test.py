"""Transformers 基线 Test：验证 Mage-VL 本地图片生成结果。"""

import json
import os
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
revision = str(model_settings["revision"])
alias, prompt, max_new_tokens = "mage-vl-4b", "请简要描述图片中的主要内容。", 2
image_path = root / "test.jpg"
cuda = torch.cuda.is_available()
dtype = getattr(torch, str(model_settings["dtype"]))

print("================= baseline =================")
if not (model_path / "config.json").is_file() or not image_path.is_file():
    raise FileNotFoundError(f"请先准备本地权重和 test.jpg：{model_path}, {image_path}")
quant = model_settings["quantization"] if cuda else {"method": "none"}
model_kwargs = {
    "trust_remote_code": True, "device_map": model_settings["device_map"] if cuda else "cpu", "revision": revision,
    "dtype": dtype, "local_files_only": True, "attn_implementation": model_settings["attn_implementation"],
    "use_safetensors": model_settings["use_safetensors"], "weights_only": model_settings["weights_only"], "offload_buffers": model_settings["offload_buffers"],
}
for key in ("experts_implementation", "use_kernels", "disable_mmap"):
    if model_settings.get(key) is not None:
        model_kwargs[key] = model_settings[key]
if model_settings.get("kernel_config"):
    model_kwargs["kernel_config"] = KernelConfig(**model_settings["kernel_config"])
for key, target in (("gpu_max_memory", 0), ("cpu_max_memory", "cpu")):
    if cuda and quant.get(key):
        model_kwargs.setdefault("max_memory", {})[target] = str(quant[key])
if str(quant.get("method", "none")) == "nf4":
    model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type=str(quant.get("quant_type", "nf4")), bnb_4bit_use_double_quant=bool(quant.get("double_quant", True)), bnb_4bit_compute_dtype=getattr(torch, str(quant.get("compute_dtype", model_settings["dtype"]))))
elif str(quant.get("method", "none")) == "int8":
    model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True, llm_int8_enable_fp32_cpu_offload=bool(quant.get("llm_int8_enable_fp32_cpu_offload", False)), llm_int8_threshold=float(quant.get("llm_int8_threshold", 6.0)), llm_int8_has_fp16_weight=bool(quant.get("llm_int8_has_fp16_weight", False)), llm_int8_skip_modules=quant.get("llm_int8_skip_modules"))
elif quant.get("method", "none") not in {"none", "prequantized"}:
    raise ValueError(f"不支持的量化方法：{quant['method']}")
processor_kwargs = {"trust_remote_code": True, "revision": revision, "local_files_only": True}
if config.get("processor", {}).get("use_fast") is not None:
    processor_kwargs["use_fast"] = bool(config["processor"]["use_fast"])
processor = AutoProcessor.from_pretrained(model_path, **processor_kwargs)
model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs).eval()
image = Image.open(image_path).convert("RGB")
image = _resize_image(image, int(config["api"]["max_image_side"]))
messages = [{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image"}]}]
prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
encoded = processor(text=[prompt_text], images=[image], return_tensors="pt")
inputs = {key: value.to(device=model.get_input_embeddings().weight.device, dtype=model.dtype) if key in {"pixel_values", "pixel_values_videos"} and isinstance(value, torch.Tensor) and value.is_floating_point() else value.to(model.get_input_embeddings().weight.device) if isinstance(value, torch.Tensor) else value for key, value in encoded.items()}
inputs.setdefault("attention_mask", torch.ones_like(inputs["input_ids"]))
started = time.perf_counter()
with torch.inference_mode():
    output = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens)
latency_ms = (time.perf_counter() - started) * 1000
token_ids = output[:, inputs["input_ids"].shape[-1]:].detach().cpu().reshape(-1).tolist()
vocab_size = int(getattr(getattr(model.config, "text_config", model.config), "vocab_size"))
if not 1 <= len(token_ids) <= max_new_tokens or min(token_ids) < 0 or max(token_ids) >= vocab_size:
    raise AssertionError("Transformers 返回了无效 token ID 或 token 数量。")
text = processor.batch_decode([token_ids], skip_special_tokens=True)[0].strip()
report = {
    "schema_version": 1, "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
    "model": {"name": "mage_vl_4b", "alias": alias, "repo_id": str(model_settings["id"]), "revision": revision},
    "implementation": "baseline", "task": "test",
    "environment": {"device": "cuda" if cuda else "cpu", "gpu_name": torch.cuda.get_device_name(0) if cuda else "CPU", "torch_version": torch.__version__, "cuda_version": torch.version.cuda, "dtype": str(model.dtype)},
    "workload": {"input_type": "image", "image_width": image.width, "image_height": image.height, "prompt": prompt, "prompt_tokens": int(inputs["input_ids"].shape[-1]), "max_new_tokens": max_new_tokens},
    "result": {"status": "passed", "generated_token_ids": token_ids, "generated_text": text}, "metrics": {"latency_ms": latency_ms}, "operator_records": [], "trace_path": None,
}
output_path = root / "artifacts/mage_vl_4b/baseline/test.json"
output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
print("status: passed", "generated_token_ids:", token_ids)
print("generated_text:", text, "json:", output_path)
