"""Transformers 基线 Test：校验图片输入和确定性生成结果。"""

import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

root = Path(__file__).resolve().parents[3]
config_path = root / "tests/qwen3_vl_2b/baseline/profiles/inference.yml"
config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
model_path = (root / config["runtime"]["local_model_path"]).resolve()
revision = str(config["model"]["revision"])
alias = "qwen3-vl-2b"
image_path = root / "test.jpg"
prompt = "请简要描述图片中的主要内容。"
max_new_tokens = 2
cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda else "cpu")
dtype = getattr(torch, str(config["model"]["dtype"]))

print("================= baseline =================")
if not (model_path / "model.safetensors").is_file() or not image_path.is_file():
    raise FileNotFoundError(f"请先准备本地权重和 test.jpg：{model_path}, {image_path}")
processor = AutoProcessor.from_pretrained(model_path, revision=revision, local_files_only=True)
model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_path, revision=revision, local_files_only=True, use_safetensors=True,
    dtype=dtype, device_map=0 if cuda else "cpu", attn_implementation=str(config["model"]["attn_implementation"]),
).eval()
image = Image.open(image_path).convert("RGB")
messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt")
inputs = {key: value.to(model.device) if isinstance(value, torch.Tensor) else value for key, value in inputs.items() if key != "token_type_ids"}
inputs.setdefault("attention_mask", torch.ones_like(inputs["input_ids"]))
started = time.perf_counter()
with torch.inference_mode():
    output = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens)
elapsed_ms = (time.perf_counter() - started) * 1000
token_ids = output[:, inputs["input_ids"].shape[-1]:].detach().cpu().reshape(-1).tolist()
vocab_size = int(model.config.text_config.vocab_size)
if not 1 <= len(token_ids) <= max_new_tokens or min(token_ids) < 0 or max(token_ids) >= vocab_size:
    raise AssertionError("Transformers 返回了无效 token ID 或 token 数量。")
text = processor.batch_decode([token_ids], skip_special_tokens=True)[0].strip()
report = {
    "schema_version": 1, "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
    "model": {"name": "qwen3_vl_2b", "alias": alias, "repo_id": "Qwen/Qwen3-VL-2B-Instruct", "revision": revision},
    "implementation": "baseline", "task": "test",
    "environment": {"device": str(device), "gpu_name": torch.cuda.get_device_name(0) if cuda else "CPU", "torch_version": torch.__version__, "cuda_version": torch.version.cuda, "dtype": str(model.dtype)},
    "workload": {"input_type": "image", "image_width": image.width, "image_height": image.height, "prompt": prompt, "prompt_tokens": int(inputs["input_ids"].shape[-1]), "max_new_tokens": max_new_tokens},
    "result": {"status": "passed", "generated_token_ids": token_ids, "generated_text": text},
    "metrics": {"latency_ms": elapsed_ms}, "operator_records": [], "trace_path": None,
}
output_path = root / "artifacts/qwen3_vl_2b/baseline/test.json"
output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print("status: passed")
print("generated_token_ids:", token_ids)
print("generated_text:", text)
print("json:", output_path)
