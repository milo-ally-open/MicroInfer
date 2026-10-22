"""透過 Transformers 載入固定 revision 的 Mage-VL 模型與 Processor。"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
from transformers import AutoProcessor, BitsAndBytesConfig, KernelConfig

from .. import config
from . import MAGE_VL_SOURCE_REVISION
from .conditional_generation import MageVLForConditionalGeneration
from .configuration_mage_vl import MageVLConfig


def load_mage_vl() -> tuple[str, Any, Any]:
    """依模型 YAML 載入模型和 Processor，並回傳快照路徑。"""
    if config.MODEL_REVISION != MAGE_VL_SOURCE_REVISION:
        raise RuntimeError(
            "The local Mage-VL model implementation is pinned to revision "
            f"{MAGE_VL_SOURCE_REVISION}, but inference.yml selects {config.MODEL_REVISION}."
        )
    optimization = config.CONFIG.get("optimization", {})
    pytorch_settings = optimization.get("pytorch", {})
    torch.backends.cuda.matmul.allow_tf32 = bool(pytorch_settings.get("allow_tf32", False))
    torch.backends.cudnn.allow_tf32 = bool(pytorch_settings.get("cudnn_allow_tf32", False))
    torch.backends.cudnn.benchmark = bool(pytorch_settings.get("cudnn_benchmark", False))
    matmul_precision = pytorch_settings.get("float32_matmul_precision")
    if matmul_precision is not None:
        if matmul_precision not in {"highest", "high", "medium"}:
            raise RuntimeError("optimization.pytorch.float32_matmul_precision must be highest, high, or medium.")
        torch.set_float32_matmul_precision(matmul_precision)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; this API is configured for NVIDIA GPU serving.")

    model_path = config.resolve_model_load_path()
    model_config = config.MODEL_CONFIG
    quantization = config.QUANTIZATION_CONFIG
    model_kwargs: dict[str, Any] = {
        "device_map": model_config["device_map"],
        "revision": config.MODEL_REVISION,
        "dtype": getattr(torch, model_config["dtype"]),
        "local_files_only": config.LOCAL_FILES_ONLY,
    }
    for key in (
        "attn_implementation", "experts_implementation", "use_kernels", "use_safetensors",
        "weights_only", "offload_buffers", "disable_mmap",
    ):
        if model_config.get(key) is not None:
            model_kwargs[key] = model_config[key]
    kernel_config = model_config.get("kernel_config")
    if kernel_config is not None:
        if not isinstance(kernel_config, dict):
            raise RuntimeError("model.kernel_config must be a mapping in the API YAML.")
        model_kwargs["kernel_config"] = KernelConfig(**kernel_config)
    if model_config.get("offload_folder"):
        offload_folder = Path(os.path.expandvars(os.path.expanduser(model_config["offload_folder"])))
        if not offload_folder.is_absolute():
            offload_folder = config.PROJECT_ROOT / offload_folder
        offload_folder.mkdir(parents=True, exist_ok=True)
        model_kwargs["offload_folder"] = str(offload_folder)

    method = quantization["method"]
    if method == "nf4":
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=str(quantization.get("quant_type", "nf4")),
            bnb_4bit_use_double_quant=bool(quantization["double_quant"]),
            bnb_4bit_compute_dtype=getattr(torch, quantization["compute_dtype"]),
            bnb_4bit_quant_storage=getattr(torch, quantization["quant_storage"])
            if quantization.get("quant_storage") else None,
        )
    elif method == "int8":
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=bool(quantization["llm_int8_enable_fp32_cpu_offload"]),
            llm_int8_threshold=float(quantization["llm_int8_threshold"]),
            llm_int8_has_fp16_weight=bool(quantization.get("llm_int8_has_fp16_weight", False)),
            llm_int8_skip_modules=quantization.get("llm_int8_skip_modules"),
        )
    elif method not in {"none", "prequantized"}:
        raise RuntimeError(f"Unsupported Transformers quantization method in config: {method}")

    max_memory: dict[int | str, str] = {}
    if quantization.get("gpu_max_memory"):
        max_memory[0] = str(quantization["gpu_max_memory"])
    if quantization.get("cpu_max_memory"):
        max_memory["cpu"] = str(quantization["cpu_max_memory"])
    if max_memory:
        model_kwargs["max_memory"] = max_memory

    processor_config = config.CONFIG.get("processor", {})
    processor_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "revision": config.MODEL_REVISION,
        "local_files_only": config.LOCAL_FILES_ONLY,
    }
    if processor_config.get("use_fast") is not None:
        processor_kwargs["use_fast"] = bool(processor_config["use_fast"])

    model_config_object = MageVLConfig.from_pretrained(
        model_path,
        revision=config.MODEL_REVISION,
        local_files_only=config.LOCAL_FILES_ONLY,
    )
    processor = AutoProcessor.from_pretrained(model_path, **processor_kwargs)
    model = MageVLForConditionalGeneration.from_pretrained(
        model_path,
        config=model_config_object,
        **model_kwargs,
    ).eval()
    return model_path, model, processor
