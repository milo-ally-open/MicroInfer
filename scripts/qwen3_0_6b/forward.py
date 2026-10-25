#!/usr/bin/env python3
"""一次性对照 Qwen3 Transformers 与项目 Inference Runtime 的 forward。"""

from __future__ import annotations

import inspect
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

# 一、固定路径：模型专属配置副本与权重都相对于仓库根目录解析。
ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "scripts/qwen3_0_6b/inference.yml"
NETWORK_CONFIG_PATH = ROOT / "scripts/qwen3_0_6b/network.yml"
MANIFEST_PATH = ROOT / "scripts/qwen3_0_6b/model.yml"
MODEL_ALIAS = "qwen3-0.6b"
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
from transformers import AutoModelForCausalLM, AutoTokenizer
import inference
from inference.engine.contracts import PreparedChatInput
from inference.models.qwen3_0_6b.processor import prepare_chat_input
from inference.models.qwen3_0_6b.runtime import create_runtime
from inference.models.qwen3_0_6b.utils import loader as inference_config


CUDA_AVAILABLE = torch.cuda.is_available()
PROFILER_ACTIVITIES = [torch.profiler.ProfilerActivity.CPU] + ([torch.profiler.ProfilerActivity.CUDA] if CUDA_AVAILABLE else [])

# 三、读取模型副本中的权重路径和 revision；实验输入与输出参数各占一行。
CONFIG = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
MODEL_SETTINGS = CONFIG["model"]
MODEL_PATH = (ROOT / Path(os.path.expandvars(os.path.expanduser(CONFIG["runtime"]["local_model_path"])))).resolve()
REVISION = str(MODEL_SETTINGS["revision"])
DTYPE = getattr(torch, str(MODEL_SETTINGS["dtype"]))
ATTENTION = "sdpa"  # 只控制 Transformers 基线；Inference 使用自己的 GQA/einsum 实现。
DEVICE_MAP = 0 if CUDA_AVAILABLE else "cpu"
PROMPT = "用一句话说明大型语言模型推理服务。"
MAX_NEW_TOKENS = 2  # prefill 生成首 token，再执行一次 decode。
MAX_OPERATORS = 20  # 终端表格显示前 20 项；JSON 保存全部算子。
OUTPUT_PATH = ROOT / "artifacts/qwen3-0.6b-forward.json"

# 四、两条路径共用同一份输入；状态只在本脚本的主流程与这两个前向函数间传递。
INPUT_IDS_CPU: torch.Tensor | None = None
TRANSFORMERS_NEXT_TOKEN: torch.Tensor | None = None
TRANSFORMERS_PREFILL_LOGITS: torch.Tensor | None = None
TRANSFORMERS_DECODE_LOGITS: torch.Tensor | None = None
INFERENCE_PREFILL_LOGITS: torch.Tensor | None = None
INFERENCE_DECODE_LOGITS: torch.Tensor | None = None


def forward_transformer():
    """第一段：用 Hugging Face Transformers 对本机 Qwen3 做 prefill 和一步 decode。"""
    global TRANSFORMERS_NEXT_TOKEN, TRANSFORMERS_PREFILL_LOGITS, TRANSFORMERS_DECODE_LOGITS
    if INPUT_IDS_CPU is None:
        raise RuntimeError("先在主流程准备聊天 token。")

    print("==================== transformers ====================")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
        use_safetensors=True,
        revision=REVISION,
        device_map=DEVICE_MAP,
        dtype=DTYPE,
        attn_implementation=ATTENTION,
    ).eval()
    device = model.get_input_embeddings().weight.device
    inputs = {
        "input_ids": INPUT_IDS_CPU.to(device),
        "attention_mask": torch.ones_like(INPUT_IDS_CPU, device=device),
    }
    forward_options = {"use_cache": True, "return_dict": True}
    supports_logits_to_keep = "logits_to_keep" in inspect.signature(model.forward).parameters
    if supports_logits_to_keep:
        forward_options["logits_to_keep"] = 1

    # A. Prefill：对整段提示词做一次模型前向，保留 Transformers 自己的 Cache。
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
            prefill_output = model(**inputs, **forward_options)
        if prefill_end is not None:
            prefill_end.record()
            prefill_end.synchronize()
    if prefill_output.past_key_values is None:
        raise RuntimeError("Transformers forward 没有返回 past_key_values，无法执行 decode。")
    TRANSFORMERS_NEXT_TOKEN = prefill_output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    TRANSFORMERS_PREFILL_LOGITS = prefill_output.logits[:, -1, :].detach().float().cpu()
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

    # B. Decode：把首个生成 token 接到 Cache 后，只前向一个 token。
    decode_ids = torch.cat((inputs["input_ids"], TRANSFORMERS_NEXT_TOKEN), dim=-1)
    decode_inputs = model.prepare_inputs_for_generation(
        decode_ids,
        past_key_values=prefill_output.past_key_values,
        attention_mask=torch.ones_like(decode_ids),
        use_cache=True,
    )
    if supports_logits_to_keep:
        decode_inputs["logits_to_keep"] = 1
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
    TRANSFORMERS_DECODE_LOGITS = decode_output.logits[:, -1, :].detach().float().cpu()
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
        "attention_implementation": str(getattr(model.config, "_attn_implementation", ATTENTION)),
        "num_hidden_layers": int(getattr(model.config, "num_hidden_layers", 0)),
        "cache_layout": "transformers_cache",
        "logits_to_keep": bool(supports_logits_to_keep),
        "prefill_cuda_event_ms": prefill_ms,
        "decode_cuda_event_ms": decode_ms,
        "peak_cuda_memory_bytes": max(torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())) if CUDA_AVAILABLE else 0,
        "peak_cuda_reserved_bytes": max(torch.cuda.max_memory_reserved(i) for i in range(torch.cuda.device_count())) if CUDA_AVAILABLE else 0,
        "cache_allocated_bytes": None,
        "block_size_tokens": None,
        "next_token_id": int(TRANSFORMERS_NEXT_TOKEN.item()),
        "operator_records": prefill_records + decode_records,
    }
    del model, prefill_output, decode_output, decode_inputs
    if CUDA_AVAILABLE:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    return result


def forward_inference():
    """第二段：从 inference 包建立 Qwen3 Runtime，用项目模型与 paged KV 做同样的前向。"""
    global INFERENCE_PREFILL_LOGITS, INFERENCE_DECODE_LOGITS
    if INPUT_IDS_CPU is None or TRANSFORMERS_NEXT_TOKEN is None:
        raise RuntimeError("先运行 forward_transformer() 准备共同输入和 decode token。")

    print("==================== inference ====================")
    print("package:", inference.__file__)
    inference_config.LOCAL_MODEL_PATH = MODEL_PATH
    inference_config.MODEL_REVISION = REVISION
    inference_config.MODEL_DTYPE = str(MODEL_SETTINGS["dtype"])
    runtime = create_runtime()
    try:
        runtime.load()
        runner = runtime.model_runner
        if runner.model is None or runner.cache is None:
            raise RuntimeError("Inference Runtime 没有建立 Qwen3 模型或 paged KV cache。")
        device = runner.input_device
        block_size = runner.cache.block_size_tokens
        token_count = int(INPUT_IDS_CPU.shape[-1])
        block_count = (token_count + 1 + block_size - 1) // block_size
        if block_count > runner.cache.num_blocks:
            raise ValueError("提示词超过 Inference Runtime 的 KV page pool 容量。")
        request_id = "qwen3-forward-analysis"
        block_table = tuple(range(block_count))
        runner.cache.bind(request_id, block_table)
        input_ids = INPUT_IDS_CPU.to(device)

        # A. Prefill：调用 inference Runtime 实际使用的 Qwen3Model.forward_batch。
        if CUDA_AVAILABLE:
            torch.cuda.reset_peak_memory_stats(device)
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
                prefill_logits = runner.model.forward_batch(
                    input_ids, token_lengths=(token_count,), position_offsets=(0,),
                    request_ids=(request_id,), block_tables=(block_table,), cache=runner.cache,
                )
            if prefill_end is not None:
                prefill_end.record()
                prefill_end.synchronize()
        INFERENCE_PREFILL_LOGITS = prefill_logits.detach().float().cpu()
        inference_next_token = int(prefill_logits.argmax(dim=-1).item())
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

        # B. Decode：复用同一个首 token 和 KV page，只执行一个 query token。
        decode_ids = TRANSFORMERS_NEXT_TOKEN.to(device)
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
                decode_logits = runner.model.forward_batch(
                    decode_ids, token_lengths=(1,), position_offsets=(token_count,),
                    request_ids=(request_id,), block_tables=(block_table,), cache=runner.cache,
                )
            if decode_end is not None:
                decode_end.record()
                decode_end.synchronize()
        INFERENCE_DECODE_LOGITS = decode_logits.detach().float().cpu()
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
            "execution_path": "create_runtime().model_runner.model.forward_batch",
            "attention_implementation": "native_eager_gqa_einsum",
            "cache_layout": "inference_paged_kv",
            "logits_to_keep": True,
            "prefill_cuda_event_ms": prefill_ms,
            "decode_cuda_event_ms": decode_ms,
            "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if CUDA_AVAILABLE else 0,
            "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved(device)) if CUDA_AVAILABLE else 0,
            "cache_allocated_bytes": runner.cache.bytes_allocated,
            "block_size_tokens": block_size,
            "next_token_id": inference_next_token,
            "operator_records": prefill_records + decode_records,
        }
        runner.cache.release(request_id)
        return result
    finally:
        runtime.close()


if __name__ == "__main__":
    # 五、准备一次聊天输入，确保 Transformers 与 Inference 使用完全相同的 token。
    if not MODEL_PATH.is_dir() or not (MODEL_PATH / "config.json").is_file():
        raise FileNotFoundError(f"本机 Qwen3 snapshot 不完整：{MODEL_PATH}")
    print("运行设备:", "cuda" if CUDA_AVAILABLE else "cpu（CUDA 不可用，已回退）")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH, revision=REVISION, local_files_only=True, use_fast=True
    )
    prepared: PreparedChatInput = prepare_chat_input(tokenizer, {
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": MAX_NEW_TOKENS,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 0,
    })
    INPUT_IDS_CPU = torch.tensor([prepared.input_ids], dtype=torch.long)
    query_tokens = int(INPUT_IDS_CPU.shape[-1])

    # 六、依次运行两条路径、比较输出，并保存供 analysis.py 生成 XLSX 的原始数据。
    transformers_result = forward_transformer()
    inference_result = forward_inference()
    comparison = {
        "prefill_next_token_match": transformers_result["next_token_id"] == inference_result["next_token_id"],
        "prefill_max_abs_logit_difference": float(
            (TRANSFORMERS_PREFILL_LOGITS - INFERENCE_PREFILL_LOGITS).abs().max().item()
        ),
        "decode_argmax_match": int(TRANSFORMERS_DECODE_LOGITS.argmax(dim=-1).item())
        == int(INFERENCE_DECODE_LOGITS.argmax(dim=-1).item()),
        "decode_max_abs_logit_difference": float(
            (TRANSFORMERS_DECODE_LOGITS - INFERENCE_DECODE_LOGITS).abs().max().item()
        ),
        "decode_input_token_id": transformers_result["next_token_id"],
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
            "query_tokens": query_tokens,
            "kv_tokens": query_tokens + 1,
            "warmup_iterations": 0,
            "measured_iterations": 1,
            "num_hidden_layers": transformers_result["num_hidden_layers"],
            "input_type": "text",
        },
        "implementations": {"transformers": transformers_result, "inference": inference_result},
        "comparison": comparison,
        "operator_records": operator_records,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"operator_records: {len(operator_records)}")
    print(f"report: {OUTPUT_PATH}")
