#!/usr/bin/env python3
"""一次性对照 Mage-VL Transformers 基线与项目 Inference Runtime 的图片 forward。"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# 一、固定路径：模型专属配置副本与权重都相对于仓库根目录解析。
ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "scripts/mage_vl_4b/inference.yml"
NETWORK_CONFIG_PATH = ROOT / "scripts/mage_vl_4b/network.yml"
MANIFEST_PATH = ROOT / "scripts/mage_vl_4b/model.yml"
MODEL_ALIAS = "mage-vl-4b"
os.environ["INFERENCE_PROJECT_ROOT"] = str(ROOT)
os.environ["INFERENCE_MODEL_ALIAS"] = MODEL_ALIAS
os.environ["INFERENCE_MODEL_MANIFEST"] = str(MANIFEST_PATH)
os.environ["INFERENCE_CONFIG"] = str(CONFIG_PATH)
os.environ["INFERENCE_NETWORK_CONFIG"] = str(NETWORK_CONFIG_PATH)
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# 二、所有第三方库与项目模块统一在文件顶部导入；Inference 依赖上面的配置环境。
import torch
import yaml
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig, KernelConfig
import inference
from inference.engine.sequence import ExecutionPhase, ScheduledWorkItem
from inference.models.mage_vl_4b import processor as inference_processor
from inference.models.mage_vl_4b.runtime import create_runtime


CUDA_AVAILABLE = torch.cuda.is_available()
PROFILER_ACTIVITIES = [torch.profiler.ProfilerActivity.CPU] + ([torch.profiler.ProfilerActivity.CUDA] if CUDA_AVAILABLE else [])

# 三、读取模型副本中的权重路径和 revision；实验输入与输出参数各占一行。
CONFIG = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
MODEL_SETTINGS = CONFIG["model"]
MODEL_PATH = (ROOT / Path(os.path.expandvars(os.path.expanduser(CONFIG["runtime"]["local_model_path"])))).resolve()
REVISION = str(MODEL_SETTINGS["revision"])
DTYPE = getattr(torch, str(MODEL_SETTINGS["dtype"]))
INPUT_IMAGE_PATH = ROOT / "test.jpg"
QUESTION = "请简要描述图片中的主要内容。"
MAX_NEW_TOKENS = 2  # Prefill 生成首 token，再执行一次 decode。
MAX_OPERATORS = 20  # 终端表格显示前 20 项；JSON 保存全部算子。
OUTPUT_PATH = ROOT / "artifacts/mage-vl-image-forward.json"

# 四、两条路径共用同一份图片与编码结果；状态只在本脚本主流程与两个前向函数间传递。
IMAGE: Image.Image | None = None
ENCODED_INPUTS_CPU: dict[str, Any] = {}
INPUT_IDS_CPU: torch.Tensor | None = None
BASELINE_GENERATED_IDS: torch.Tensor | None = None
INFERENCE_GENERATED_IDS: torch.Tensor | None = None
BASELINE_TEXT = ""
INFERENCE_TEXT = ""


def forward_transformer():
    """第一段：按正式 YAML 用 Transformers 载入 Mage-VL，执行图片 prefill 和一步 decode。"""
    global ENCODED_INPUTS_CPU, INPUT_IDS_CPU, BASELINE_GENERATED_IDS, BASELINE_TEXT
    if IMAGE is None:
        raise RuntimeError("先在主流程读取测试图片。")

    print("==================== transformers ====================")
    print("input_image:", INPUT_IMAGE_PATH)
    quantization = MODEL_SETTINGS["quantization"] if CUDA_AVAILABLE else {"method": "none"}
    model_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "device_map": MODEL_SETTINGS["device_map"] if CUDA_AVAILABLE else "cpu",
        "revision": REVISION,
        "dtype": DTYPE,
        "local_files_only": True,
        "attn_implementation": MODEL_SETTINGS["attn_implementation"],
        "use_safetensors": MODEL_SETTINGS["use_safetensors"],
        "weights_only": MODEL_SETTINGS["weights_only"],
        "offload_buffers": MODEL_SETTINGS["offload_buffers"],
    }
    for key in ("experts_implementation", "use_kernels", "disable_mmap"):
        if MODEL_SETTINGS.get(key) is not None:
            model_kwargs[key] = MODEL_SETTINGS[key]
    if MODEL_SETTINGS.get("kernel_config"):
        model_kwargs["kernel_config"] = KernelConfig(**MODEL_SETTINGS["kernel_config"])
    offload_folder = MODEL_SETTINGS.get("offload_folder")
    if offload_folder:
        offload_path = Path(os.path.expandvars(os.path.expanduser(str(offload_folder))))
        if not offload_path.is_absolute():
            offload_path = ROOT / offload_path
        offload_path.mkdir(parents=True, exist_ok=True)
        model_kwargs["offload_folder"] = str(offload_path.resolve())
    max_memory = {}
    if CUDA_AVAILABLE and quantization.get("gpu_max_memory"):
        max_memory[0] = str(quantization["gpu_max_memory"])
    if CUDA_AVAILABLE and quantization.get("cpu_max_memory"):
        max_memory["cpu"] = str(quantization["cpu_max_memory"])
    if CUDA_AVAILABLE and max_memory:
        model_kwargs["max_memory"] = max_memory
    method = str(quantization.get("method", "none"))
    if method == "nf4":
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=str(quantization.get("quant_type", "nf4")),
            bnb_4bit_use_double_quant=bool(quantization.get("double_quant", True)),
            bnb_4bit_compute_dtype=getattr(torch, str(quantization.get("compute_dtype", MODEL_SETTINGS["dtype"]))),
            bnb_4bit_quant_storage=(
                getattr(torch, str(quantization["quant_storage"])) if quantization.get("quant_storage") else None
            ),
        )
    elif method == "int8":
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=bool(quantization.get("llm_int8_enable_fp32_cpu_offload", False)),
            llm_int8_threshold=float(quantization.get("llm_int8_threshold", 6.0)),
            llm_int8_has_fp16_weight=bool(quantization.get("llm_int8_has_fp16_weight", False)),
            llm_int8_skip_modules=quantization.get("llm_int8_skip_modules"),
        )
    elif method not in {"none", "prequantized"}:
        raise ValueError(f"不支持的量化方法：{method}")

    processor_kwargs = {"trust_remote_code": True, "revision": REVISION, "local_files_only": True}
    if CONFIG.get("processor", {}).get("use_fast") is not None:
        processor_kwargs["use_fast"] = bool(CONFIG["processor"]["use_fast"])
    processor = AutoProcessor.from_pretrained(MODEL_PATH, **processor_kwargs)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, **model_kwargs).eval()
    messages = [{"role": "user", "content": [
        {"type": "text", "text": QUESTION}, {"type": "image"},
    ]}]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    encoded = processor(text=[prompt], images=[IMAGE], return_tensors="pt")
    ENCODED_INPUTS_CPU = {
        key: value.detach().cpu() if isinstance(value, torch.Tensor) else value
        for key, value in encoded.items()
    }
    INPUT_IDS_CPU = ENCODED_INPUTS_CPU["input_ids"]
    input_device = model.get_input_embeddings().weight.device
    model_inputs = {}
    for key, value in ENCODED_INPUTS_CPU.items():
        if isinstance(value, torch.Tensor):
            if key in {"pixel_values", "pixel_values_videos"} and value.is_floating_point():
                model_inputs[key] = value.to(device=input_device, dtype=model.dtype)
            else:
                model_inputs[key] = value.to(input_device)
        else:
            model_inputs[key] = value
    model_inputs.setdefault("attention_mask", torch.ones_like(model_inputs["input_ids"]))

    # A. Prefill：整段图片提示词进入 Transformers 模型，保留原生 DynamicCache。
    for device_index in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(device_index)
    prefill_start = torch.cuda.Event(enable_timing=True) if CUDA_AVAILABLE else None
    prefill_end = torch.cuda.Event(enable_timing=True) if CUDA_AVAILABLE else None
    prefill_cpu_start = time.perf_counter()
    with torch.profiler.profile(
        activities=PROFILER_ACTIVITIES,
        record_shapes=True,
        profile_memory=True,
    ) as prefill_profile:
        if prefill_start is not None:
            prefill_start.record()
        with torch.inference_mode():
            prefill_output = model(**model_inputs, use_cache=True, return_dict=True)
        if prefill_end is not None:
            prefill_end.record()
            prefill_end.synchronize()
    if prefill_output.past_key_values is None:
        raise RuntimeError("Transformers forward 没有返回 past_key_values，无法执行 decode。")
    first_token = prefill_output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    prefill_logits = prefill_output.logits[:, -1, :].detach().float().cpu()
    prefill_ms = float(prefill_start.elapsed_time(prefill_end)) if CUDA_AVAILABLE else (time.perf_counter() - prefill_cpu_start) * 1000
    prefill_averages = prefill_profile.key_averages(group_by_input_shape=True)
    prefill_records = []
    for event in prefill_averages:
        cuda_time = getattr(event, "self_device_time_total", getattr(event, "self_cuda_time_total", 0.0))
        cuda_memory = getattr(event, "self_device_memory_usage", getattr(event, "self_cuda_memory_usage", 0))
        prefill_records.append({
            "phase": "prefill", "operator": str(event.key), "input_shapes": event.input_shapes,
            "call_count": int(event.count), "self_cuda_time_total_us": float(cuda_time or 0.0),
            "self_cpu_time_total_us": float(event.self_cpu_time_total),
            "self_cuda_memory_usage_bytes": int(cuda_memory or 0),
            "self_cpu_memory_usage_bytes": int(event.self_cpu_memory_usage),
        })
    prefill_records.sort(key=lambda row: row["self_cuda_time_total_us"], reverse=True)
    print(f"prefill_cuda_event_ms: {prefill_ms:.3f}")
    print(prefill_averages.table(sort_by="self_cuda_time_total", row_limit=MAX_OPERATORS))

    # B. Decode：使用 Transformers 产生的首 token，读取 prefill Cache 再前向一次。
    decode_ids = torch.cat((model_inputs["input_ids"], first_token), dim=-1)
    decode_inputs = model.prepare_inputs_for_generation(
        decode_ids,
        past_key_values=prefill_output.past_key_values,
        attention_mask=torch.ones_like(decode_ids),
        use_cache=True,
    )
    decode_start = torch.cuda.Event(enable_timing=True) if CUDA_AVAILABLE else None
    decode_end = torch.cuda.Event(enable_timing=True) if CUDA_AVAILABLE else None
    decode_cpu_start = time.perf_counter()
    with torch.profiler.profile(
        activities=PROFILER_ACTIVITIES,
        record_shapes=True,
        profile_memory=True,
    ) as decode_profile:
        if decode_start is not None:
            decode_start.record()
        with torch.inference_mode():
            decode_output = model(**decode_inputs, return_dict=True)
        if decode_end is not None:
            decode_end.record()
            decode_end.synchronize()
    second_token = decode_output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    decode_logits = decode_output.logits[:, -1, :].detach().float().cpu()
    BASELINE_GENERATED_IDS = torch.cat((first_token, second_token), dim=-1).detach().cpu()
    BASELINE_TEXT = processor.batch_decode(BASELINE_GENERATED_IDS, skip_special_tokens=True)[0].strip()
    prefill_logits = prefill_logits.detach().cpu()
    decode_ms = float(decode_start.elapsed_time(decode_end)) if CUDA_AVAILABLE else (time.perf_counter() - decode_cpu_start) * 1000
    decode_averages = decode_profile.key_averages(group_by_input_shape=True)
    decode_records = []
    for event in decode_averages:
        cuda_time = getattr(event, "self_device_time_total", getattr(event, "self_cuda_time_total", 0.0))
        cuda_memory = getattr(event, "self_device_memory_usage", getattr(event, "self_cuda_memory_usage", 0))
        decode_records.append({
            "phase": "decode", "operator": str(event.key), "input_shapes": event.input_shapes,
            "call_count": int(event.count), "self_cuda_time_total_us": float(cuda_time or 0.0),
            "self_cpu_time_total_us": float(event.self_cpu_time_total),
            "self_cuda_memory_usage_bytes": int(cuda_memory or 0),
            "self_cpu_memory_usage_bytes": int(event.self_cpu_memory_usage),
        })
    decode_records.sort(key=lambda row: row["self_cuda_time_total_us"], reverse=True)
    print(f"decode_cuda_event_ms: {decode_ms:.3f}")
    print(decode_averages.table(sort_by="self_cuda_time_total", row_limit=MAX_OPERATORS))

    result = {
        "execution_path": "AutoModelForCausalLM.forward",
        "attention_implementation": str(getattr(model.config, "_attn_implementation", MODEL_SETTINGS["attn_implementation"])),
        "num_hidden_layers": int(getattr(getattr(model.config, "text_config", model.config), "num_hidden_layers", 0)),
        "cache_layout": "transformers_cache",
        "prefill_cuda_event_ms": prefill_ms,
        "decode_cuda_event_ms": decode_ms,
        "peak_cuda_memory_bytes": max(torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())) if CUDA_AVAILABLE else 0,
        "peak_cuda_reserved_bytes": max(torch.cuda.max_memory_reserved(i) for i in range(torch.cuda.device_count())) if CUDA_AVAILABLE else 0,
        "cache_allocated_bytes": None,
        "block_size_tokens": None,
        "generated_text": BASELINE_TEXT,
        "generated_token_ids": [int(token) for token in BASELINE_GENERATED_IDS[0].tolist()],
        "operator_records": prefill_records + decode_records,
    }
    del model, processor, prefill_output, decode_output, decode_inputs
    if CUDA_AVAILABLE:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    return result


def forward_inference():
    """第二段：通过项目 Mage-VL ModelRunner 执行 prefill/decode，并记录同样的算子数据。"""
    global INFERENCE_GENERATED_IDS, INFERENCE_TEXT
    if not ENCODED_INPUTS_CPU or INPUT_IDS_CPU is None:
        raise RuntimeError("先运行 forward_transformer() 生成本机图片输入张量。")

    print("==================== inference ====================")
    print("package:", inference.__file__)
    runtime = create_runtime()
    try:
        runtime.load()
        runner = runtime.model_runner
        if runner.model is None or runner.processor is None:
            raise RuntimeError("Inference Runtime 没有建立 Mage-VL 模型或 Processor。")
        input_device = runner.model.get_input_embeddings().weight.device
        model_inputs = {}
        for key, value in ENCODED_INPUTS_CPU.items():
            if isinstance(value, torch.Tensor):
                if key in {"pixel_values", "pixel_values_videos"} and value.is_floating_point():
                    model_inputs[key] = value.to(device=input_device, dtype=runner.model.dtype)
                else:
                    model_inputs[key] = value.to(input_device)
            else:
                model_inputs[key] = value
        model_inputs.setdefault("attention_mask", torch.ones_like(model_inputs["input_ids"]))
        prompt_tokens = int(model_inputs["input_ids"].shape[-1])
        generation_options = {
            "max_new_tokens": MAX_NEW_TOKENS,
            "do_sample": False,
            "temperature": 0.0,
            "use_cache": True,
            "eos_token_id": (),  # 保证本次固定测完 prefill 和一次 decode。
        }
        request_id = "mage-vl-forward-analysis"
        block_size = int(CONFIG["engine"]["kv_cache"]["block_size_tokens"])
        block_count = (prompt_tokens + MAX_NEW_TOKENS + block_size - 1) // block_size
        runner.begin_generation(request_id, model_inputs, generation_options)
        runner.bind_generation_cache(request_id, tuple(range(block_count)))
        runner.reset_peak_memory()

        # A. Prefill：通过 ModelRunner.execute_batch 进入项目逐步生成路径。
        prefill_start = torch.cuda.Event(enable_timing=True) if CUDA_AVAILABLE else None
        prefill_end = torch.cuda.Event(enable_timing=True) if CUDA_AVAILABLE else None
        prefill_cpu_start = time.perf_counter()
        with torch.profiler.profile(
            activities=PROFILER_ACTIVITIES,
            record_shapes=True,
            profile_memory=True,
        ) as prefill_profile:
            if prefill_start is not None:
                prefill_start.record()
            with torch.inference_mode():
                prefill_result = runner.execute_batch((
                    ScheduledWorkItem(request_id, ExecutionPhase.PREFILL, prompt_tokens),
                ))[0]
            if prefill_end is not None:
                prefill_end.record()
                prefill_end.synchronize()
        prefill_ms = float(prefill_start.elapsed_time(prefill_end)) if CUDA_AVAILABLE else (time.perf_counter() - prefill_cpu_start) * 1000
        prefill_averages = prefill_profile.key_averages(group_by_input_shape=True)
        prefill_records = []
        for event in prefill_averages:
            cuda_time = getattr(event, "self_device_time_total", getattr(event, "self_cuda_time_total", 0.0))
            cuda_memory = getattr(event, "self_device_memory_usage", getattr(event, "self_cuda_memory_usage", 0))
            prefill_records.append({
                "phase": "prefill", "operator": str(event.key), "input_shapes": event.input_shapes,
                "call_count": int(event.count), "self_cuda_time_total_us": float(cuda_time or 0.0),
                "self_cpu_time_total_us": float(event.self_cpu_time_total),
                "self_cuda_memory_usage_bytes": int(cuda_memory or 0),
                "self_cpu_memory_usage_bytes": int(event.self_cpu_memory_usage),
            })
        prefill_records.sort(key=lambda row: row["self_cuda_time_total_us"], reverse=True)
        print(f"prefill_cuda_event_ms: {prefill_ms:.3f}")
        print(prefill_averages.table(sort_by="self_cuda_time_total", row_limit=MAX_OPERATORS))

        # B. Decode：执行一个 token 的 ModelRunner 工作项，并释放该请求 Cache。
        decode_start = torch.cuda.Event(enable_timing=True) if CUDA_AVAILABLE else None
        decode_end = torch.cuda.Event(enable_timing=True) if CUDA_AVAILABLE else None
        decode_cpu_start = time.perf_counter()
        with torch.profiler.profile(
            activities=PROFILER_ACTIVITIES,
            record_shapes=True,
            profile_memory=True,
        ) as decode_profile:
            if decode_start is not None:
                decode_start.record()
            with torch.inference_mode():
                decode_result = runner.execute_batch((
                    ScheduledWorkItem(request_id, ExecutionPhase.DECODE, 1),
                ))[0]
            if decode_end is not None:
                decode_end.record()
                decode_end.synchronize()
        decode_ms = float(decode_start.elapsed_time(decode_end)) if CUDA_AVAILABLE else (time.perf_counter() - decode_cpu_start) * 1000
        decode_averages = decode_profile.key_averages(group_by_input_shape=True)
        decode_records = []
        for event in decode_averages:
            cuda_time = getattr(event, "self_device_time_total", getattr(event, "self_cuda_time_total", 0.0))
            cuda_memory = getattr(event, "self_device_memory_usage", getattr(event, "self_cuda_memory_usage", 0))
            decode_records.append({
                "phase": "decode", "operator": str(event.key), "input_shapes": event.input_shapes,
                "call_count": int(event.count), "self_cuda_time_total_us": float(cuda_time or 0.0),
                "self_cpu_time_total_us": float(event.self_cpu_time_total),
                "self_cuda_memory_usage_bytes": int(cuda_memory or 0),
                "self_cpu_memory_usage_bytes": int(event.self_cpu_memory_usage),
            })
        decode_records.sort(key=lambda row: row["self_cuda_time_total_us"], reverse=True)
        print(f"decode_cuda_event_ms: {decode_ms:.3f}")
        print(decode_averages.table(sort_by="self_cuda_time_total", row_limit=MAX_OPERATORS))
        generated_ids, finish_reason = runner.finish_generation(request_id)
        INFERENCE_GENERATED_IDS = generated_ids.detach().cpu()
        INFERENCE_TEXT = runner.decode(generated_ids)
        result = {
            "execution_path": "create_runtime().model_runner.execute_batch",
            "attention_implementation": str(getattr(runner.model.config, "_attn_implementation", MODEL_SETTINGS["attn_implementation"])),
            "num_hidden_layers": int(getattr(getattr(runner.model.config, "text_config", runner.model.config), "num_hidden_layers", 0)),
            "cache_layout": "transformers_cache_with_engine_reservation",
            "prefill_cuda_event_ms": prefill_ms,
            "decode_cuda_event_ms": decode_ms,
            "peak_cuda_memory_bytes": runner.peak_memory()["peak_allocated"],
            "peak_cuda_reserved_bytes": runner.peak_memory()["peak_reserved"],
            "cache_allocated_bytes": None,
            "block_size_tokens": block_size,
            "generated_text": INFERENCE_TEXT,
            "generated_token_ids": [int(token) for token in INFERENCE_GENERATED_IDS[0].tolist()],
            "finish_reason": finish_reason,
            "prefill_generated_text": prefill_result.text_delta,
            "decode_generated_text": decode_result.text_delta,
            "operator_records": prefill_records + decode_records,
        }
        return result
    finally:
        runtime.close()


if __name__ == "__main__":
    # 五、检查本机权重和固定图片，再将图片处理成两条路径共用的输入。
    if not MODEL_PATH.is_dir() or not (MODEL_PATH / "config.json").is_file():
        raise FileNotFoundError(f"本机 Mage-VL snapshot 不完整：{MODEL_PATH}")
    if not INPUT_IMAGE_PATH.is_file():
        raise FileNotFoundError(f"测试图片不存在：{INPUT_IMAGE_PATH}；请修改文件顶部的 INPUT_IMAGE_PATH。")
    print("运行设备:", "cuda" if CUDA_AVAILABLE else "cpu（CUDA 不可用，已回退）")
    if INPUT_IMAGE_PATH.stat().st_size > int(CONFIG["api"]["max_image_bytes"]):
        raise ValueError("测试图片超过 inference.yml 的 api.max_image_bytes 限制。")
    with Image.open(INPUT_IMAGE_PATH) as opened:
        if opened.width * opened.height > int(CONFIG["api"]["max_image_pixels"]):
            raise ValueError("测试图片超过 inference.yml 的 api.max_image_pixels 限制。")
        IMAGE = opened.convert("RGB")
    IMAGE = inference_processor._resize_image(IMAGE, int(CONFIG["api"]["max_image_side"]))

    # 六、依次运行两条路径、比较输出，并保存供 analysis.py 生成 XLSX 的原始数据。
    transformers_result = forward_transformer()
    inference_result = forward_inference()
    comparison = {
        "first_token_match": bool(
            BASELINE_GENERATED_IDS is not None
            and INFERENCE_GENERATED_IDS is not None
            and BASELINE_GENERATED_IDS.shape == INFERENCE_GENERATED_IDS.shape
            and int(BASELINE_GENERATED_IDS[0, 0]) == int(INFERENCE_GENERATED_IDS[0, 0])
        ),
        "all_generated_tokens_match": bool(
            BASELINE_GENERATED_IDS is not None
            and INFERENCE_GENERATED_IDS is not None
            and torch.equal(BASELINE_GENERATED_IDS, INFERENCE_GENERATED_IDS)
        ),
        "text_match": BASELINE_TEXT == INFERENCE_TEXT,
    }
    print("==================== comparison ====================")
    print(json.dumps(comparison, ensure_ascii=False, indent=2))
    operator_records = [
        {"implementation": implementation_name, **record}
        for implementation_name, result in (("transformers", transformers_result), ("inference", inference_result))
        for record in result["operator_records"]
    ]
    for result in (transformers_result, inference_result):
        result.pop("operator_records")
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "run": {
            "model_alias": MODEL_ALIAS,
            "revision": REVISION,
            "gpu_name": ", ".join(torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())) if CUDA_AVAILABLE else "CPU fallback",
            "execution_device": "cuda" if CUDA_AVAILABLE else "cpu",
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "dtype": str(DTYPE),
            "batch_size": 1,
            "query_tokens": int(INPUT_IDS_CPU.shape[-1]),
            "kv_tokens": int(INPUT_IDS_CPU.shape[-1]) + 1,
            "warmup_iterations": 0,
            "measured_iterations": 1,
            "num_hidden_layers": transformers_result["num_hidden_layers"],
            "input_type": "image",
        },
        "implementations": {"transformers": transformers_result, "inference": inference_result},
        "comparison": comparison,
        "operator_records": operator_records,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"operator_records: {len(operator_records)}")
    print(f"report: {OUTPUT_PATH}")
