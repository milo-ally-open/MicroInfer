"""离线撷取 Mage-VL 的模型结构、输入张量与 KV cache 资讯。"""

from __future__ import annotations

import argparse
import importlib.metadata
import inspect
import json
import math
import os
import sys
import time
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="仅使用本机权重，撷取 Mage-VL 推理与 KV cache 能力资讯。"
    )
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--image", type=Path, help="本机图片路径")
    inputs.add_argument("--video", type=Path, help="本机影片路径；以均匀取样影格送入 Processor")
    parser.add_argument("--output", required=True, type=Path, help="JSON 报告输出路径")
    parser.add_argument("--config", required=True, type=Path, help="必须明确指定 Mage-VL 探索工具 YAML 路径")
    parser.add_argument("--model-path", required=True, type=Path, help="必须明确指定已存在的本机模型快照目录")
    parser.add_argument("--question", required=True, help="送入模型的问题文字")
    parser.add_argument("--max-new-tokens", type=int, help="覆写 YAML 中 generation.max_tokens_default 的输出 token 数")
    return parser.parse_args()


def _resolve_path(value: Path) -> Path:
    """解析呼叫者提供的路径；相对路径以目前工作目录为基准。"""
    return value.expanduser().resolve()


def _read_config(path: Path) -> dict[str, Any]:
    import yaml

    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"Inference config must contain a YAML mapping: {path}")
    required_sections = {"model", "api", "generation"}
    missing_sections = sorted(required_sections - config.keys())
    if missing_sections:
        raise ValueError(f"Mage-VL test YAML is missing required sections: {', '.join(missing_sections)}")
    model = config["model"]
    api = config["api"]
    if not isinstance(model, dict) or not isinstance(api, dict) or not isinstance(config["generation"], dict):
        raise ValueError("Mage-VL test YAML sections 'model', 'api', and 'generation' must be mappings.")
    for section in ("runtime", "processor", "optimization", "backend"):
        if section in config and not isinstance(config[section], dict):
            raise ValueError(f"Mage-VL test YAML section '{section}' must be a mapping.")
    required_model_fields = {"id", "revision", "dtype", "device_map", "quantization"}
    missing_model_fields = sorted(required_model_fields - model.keys())
    if missing_model_fields:
        raise ValueError(f"Mage-VL test YAML model section is missing: {', '.join(missing_model_fields)}")
    if not isinstance(model["quantization"], dict):
        raise ValueError("Mage-VL test YAML model.quantization must be a mapping.")
    required_api_fields = {
        "max_new_tokens", "max_image_bytes", "max_image_pixels", "max_image_side",
        "max_video_bytes", "max_video_duration_seconds", "video_max_frames", "video_max_frame_side",
    }
    missing_api_fields = sorted(required_api_fields - api.keys())
    if missing_api_fields:
        raise ValueError(f"Mage-VL test YAML api section is missing: {', '.join(missing_api_fields)}")
    if str(api.get("video_backend", "frames")).lower() != "frames":
        raise ValueError("Mage-VL test YAML api.video_backend must be 'frames'.")
    if int(config["generation"].get("max_tokens_default", 0)) < 1:
        raise ValueError("Mage-VL test YAML generation.max_tokens_default must be a positive integer.")
    return config


def _version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _resolve_model_snapshot(model_path: Path) -> Path:
    resolved = _resolve_path(model_path)
    if not resolved.is_dir():
        raise FileNotFoundError(f"Local model snapshot directory does not exist: {resolved}")
    return resolved


def _load_pil_image(path: Path, config: dict[str, Any]) -> Any:
    from PIL import Image

    api = config["api"]
    if path.stat().st_size > int(api["max_image_bytes"]):
        raise ValueError(f"Image exceeds api.max_image_bytes={api['max_image_bytes']}.")
    with Image.open(path) as opened:
        max_pixels = int(api["max_image_pixels"])
        if opened.width * opened.height > max_pixels:
            raise ValueError(f"Image exceeds api.max_image_pixels={max_pixels}.")
        image = opened.convert("RGB")
    max_side = int(api["max_image_side"])
    resampling = str(api.get("image_resampling", "lanczos")).lower()
    filters = {
        "nearest": Image.Resampling.NEAREST,
        "bilinear": Image.Resampling.BILINEAR,
        "bicubic": Image.Resampling.BICUBIC,
        "lanczos": Image.Resampling.LANCZOS,
    }
    if resampling not in filters:
        raise ValueError(f"Unsupported api.image_resampling value: {resampling}")
    image.thumbnail((max_side, max_side), filters[resampling])
    return image


def _sample_video(path: Path, config: dict[str, Any]) -> tuple[list[Any], dict[str, Any]]:
    import cv2
    from PIL import Image

    api = config["api"]
    if str(api.get("video_backend", "frames")) != "frames":
        raise ValueError("Utils probe video input supports only api.video_backend=frames.")
    if path.stat().st_size > int(api["max_video_bytes"]):
        raise ValueError(f"Video exceeds api.max_video_bytes={api['max_video_bytes']}.")
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError(f"Could not open local video: {path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if not all(math.isfinite(value) for value in (fps, float(frame_count), float(width), float(height))) or fps <= 0 or frame_count <= 0:
            raise ValueError("Could not read finite video FPS, frame count, or dimensions.")
        duration = frame_count / fps
        if duration > float(api["max_video_duration_seconds"]):
            raise ValueError(
                f"Video duration {duration:.3f}s exceeds api.max_video_duration_seconds="
                f"{api['max_video_duration_seconds']}."
            )
        sample_count = min(int(api["video_max_frames"]), frame_count)
        if sample_count < 1 or int(api["video_max_frame_side"]) < 1:
            raise ValueError("api.video_max_frames and api.video_max_frame_side must be positive.")
        indices = (
            [frame_count // 2]
            if sample_count == 1
            else [round(index * (frame_count - 1) / (sample_count - 1)) for index in range(sample_count)]
        )
        max_side = int(api["video_max_frame_side"])
        frames: list[Any] = []
        timestamps: list[float] = []
        for frame_index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok or frame is None or frame.size == 0:
                raise ValueError(f"Could not decode sampled video frame {frame_index}.")
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb)
            image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
            frames.append(image)
            timestamps.append(round(frame_index / fps, 3))
        metadata = {
            "duration_seconds": round(duration, 3),
            "source_fps": round(fps, 3),
            "source_frame_count": frame_count,
            "source_width": width,
            "source_height": height,
            "sampled_frame_count": len(frames),
            "sampled_timestamps_seconds": timestamps,
            "sampling": "uniform",
        }
        return frames, metadata
    finally:
        capture.release()


def _load_model(config: dict[str, Any], snapshot: Path, torch: Any) -> tuple[Any, Any]:
    from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig

    model_config = config["model"]
    pytorch_settings = config.get("optimization", {}).get("pytorch", {})
    torch.backends.cuda.matmul.allow_tf32 = bool(pytorch_settings.get("allow_tf32", False))
    torch.backends.cudnn.allow_tf32 = bool(pytorch_settings.get("cudnn_allow_tf32", False))
    torch.backends.cudnn.benchmark = bool(pytorch_settings.get("cudnn_benchmark", False))
    matmul_precision = pytorch_settings.get("float32_matmul_precision")
    if matmul_precision is not None:
        if matmul_precision not in {"highest", "high", "medium"}:
            raise ValueError("optimization.pytorch.float32_matmul_precision must be highest, high, or medium.")
        torch.set_float32_matmul_precision(matmul_precision)
    kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "device_map": model_config["device_map"],
        "revision": model_config["revision"],
        "dtype": getattr(torch, model_config["dtype"]),
        "local_files_only": True,
    }
    for key in (
        "attn_implementation", "experts_implementation", "use_kernels", "use_safetensors",
        "weights_only", "offload_buffers", "disable_mmap",
    ):
        if model_config.get(key) is not None:
            kwargs[key] = model_config[key]
    if model_config.get("kernel_config") is not None:
        from transformers import KernelConfig

        kwargs["kernel_config"] = KernelConfig(**model_config["kernel_config"])
    if model_config.get("offload_folder"):
        offload_folder = _resolve_path(Path(model_config["offload_folder"]))
        offload_folder.mkdir(parents=True, exist_ok=True)
        kwargs["offload_folder"] = str(offload_folder)

    quantization = model_config["quantization"]
    method = quantization["method"]
    if method == "nf4":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=str(quantization.get("quant_type", "nf4")),
            bnb_4bit_use_double_quant=bool(quantization["double_quant"]),
            bnb_4bit_compute_dtype=getattr(torch, quantization["compute_dtype"]),
            bnb_4bit_quant_storage=(
                getattr(torch, quantization["quant_storage"]) if quantization.get("quant_storage") else None
            ),
        )
    elif method == "int8":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=bool(quantization["llm_int8_enable_fp32_cpu_offload"]),
            llm_int8_threshold=float(quantization["llm_int8_threshold"]),
            llm_int8_has_fp16_weight=bool(quantization.get("llm_int8_has_fp16_weight", False)),
            llm_int8_skip_modules=quantization.get("llm_int8_skip_modules"),
        )
    elif method not in {"none", "prequantized"}:
        raise ValueError(f"Unsupported model.quantization.method: {method}")

    max_memory: dict[int | str, str] = {}
    if quantization.get("gpu_max_memory"):
        max_memory[0] = str(quantization["gpu_max_memory"])
    if quantization.get("cpu_max_memory"):
        max_memory["cpu"] = str(quantization["cpu_max_memory"])
    if max_memory:
        kwargs["max_memory"] = max_memory

    processor_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "revision": model_config["revision"],
        "local_files_only": True,
    }
    use_fast = config.get("processor", {}).get("use_fast")
    if use_fast is not None:
        processor_kwargs["use_fast"] = bool(use_fast)
    processor = AutoProcessor.from_pretrained(snapshot, **processor_kwargs)
    model = AutoModelForCausalLM.from_pretrained(snapshot, **kwargs).eval()
    compile_settings = config.get("optimization", {}).get("torch_compile", {})
    if compile_settings.get("enabled", False):
        model = torch.compile(
            model,
            backend=str(compile_settings.get("backend", "inductor")),
            mode=str(compile_settings.get("mode", "default")),
            fullgraph=bool(compile_settings.get("fullgraph", False)),
            dynamic=bool(compile_settings.get("dynamic", True)),
        )
    return model, processor


def _tensor_info(value: Any, torch: Any) -> dict[str, Any] | None:
    if not isinstance(value, torch.Tensor):
        return None
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "device": str(value.device),
        "bytes": int(value.numel() * value.element_size()),
    }


def _input_snapshot(inputs: dict[str, Any], torch: Any) -> dict[str, Any]:
    result = {}
    for name, value in inputs.items():
        result[name] = _tensor_info(value, torch) or {"type": type(value).__name__}
    return result


def _cache_pairs(cache: Any) -> list[tuple[Any, Any]]:
    if cache is None:
        return []
    keys, values = getattr(cache, "key_cache", None), getattr(cache, "value_cache", None)
    if keys is not None and values is not None:
        return list(zip(keys, values))
    layers = getattr(cache, "layers", None)
    if layers is not None:
        pairs = [(getattr(layer, "keys", None), getattr(layer, "values", None)) for layer in layers]
        if any(key is not None or value is not None for key, value in pairs):
            return pairs
    to_legacy = getattr(cache, "to_legacy_cache", None)
    if callable(to_legacy):
        try:
            cache = to_legacy()
        except Exception:
            return []
    if isinstance(cache, (tuple, list)):
        return [
            (layer[0], layer[1]) if isinstance(layer, (tuple, list)) and len(layer) >= 2 else (None, None)
            for layer in cache
        ]
    return []


def _cache_snapshot(cache: Any, torch: Any, *, include_layers: bool) -> dict[str, Any]:
    pairs = _cache_pairs(cache)
    layers: list[dict[str, Any]] = []
    lengths: list[int] = []
    total_bytes = 0
    for index, (key, value) in enumerate(pairs):
        key_info, value_info = _tensor_info(key, torch), _tensor_info(value, torch)
        total_bytes += int((key_info or {}).get("bytes", 0)) + int((value_info or {}).get("bytes", 0))
        for info in (key_info, value_info):
            shape = info.get("shape") if info else None
            if shape and len(shape) >= 2:
                lengths.append(int(shape[-2]))
        if include_layers:
            layers.append({"index": index, "key": key_info, "value": value_info})
    length = None
    getter = getattr(cache, "get_seq_length", None)
    if callable(getter):
        try:
            length = int(getter())
        except Exception:
            pass
    if length is None and lengths:
        length = max(lengths)
    result: dict[str, Any] = {
        "cache_class": f"{type(cache).__module__}.{type(cache).__name__}" if cache is not None else None,
        "layer_count": len(pairs),
        "sequence_length": length,
        "sequence_length_range": [min(lengths), max(lengths)] if lengths else None,
        "tensor_bytes": total_bytes,
    }
    if include_layers:
        result["layers"] = layers
    return result


def _capture_model_structure(model: Any) -> dict[str, Any]:
    config = model.config
    text_config = getattr(config, "text_config", config)
    fields = (
        "model_type", "num_hidden_layers", "num_attention_heads", "num_key_value_heads",
        "head_dim", "hidden_size", "intermediate_size", "vocab_size", "max_position_embeddings",
    )
    attention_classes: Counter[str] = Counter()
    attention_examples: list[dict[str, str]] = []
    for name, module in model.named_modules():
        class_name = type(module).__name__
        if "attn" in name.lower() or "attention" in class_name.lower():
            attention_classes[class_name] += 1
            if len(attention_examples) < 24:
                attention_examples.append({"path": name, "class": class_name})
    dtypes = Counter(str(parameter.dtype) for parameter in model.parameters())
    return {
        "model_class": f"{type(model).__module__}.{type(model).__name__}",
        "config_class": f"{type(config).__module__}.{type(config).__name__}",
        "text_config": {key: getattr(text_config, key, None) for key in fields},
        "attention_implementation": getattr(config, "_attn_implementation", None),
        "attention_module_classes": dict(attention_classes),
        "attention_module_examples": attention_examples,
        "parameter_dtype_counts": dict(dtypes),
        "hf_device_map": getattr(model, "hf_device_map", None),
        "forward_signature": _safe_signature(model.forward),
        "prepare_inputs_for_generation_signature": _safe_signature(
            getattr(model, "prepare_inputs_for_generation", None)
        ),
    }


def _safe_signature(value: Any) -> str | None:
    try:
        return str(inspect.signature(value))
    except (TypeError, ValueError):
        return None


def _gpu_snapshot(torch: Any) -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {"available": False, "devices": []}
    devices = []
    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        free_bytes, total_bytes = torch.cuda.mem_get_info(index)
        devices.append({
            "index": index,
            "name": properties.name,
            "total_bytes": int(total_bytes),
            "free_bytes": int(free_bytes),
            "allocated_bytes": int(torch.cuda.memory_allocated(index)),
            "reserved_bytes": int(torch.cuda.memory_reserved(index)),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(index)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(index)),
        })
    return {"available": True, "devices": devices}


def _sync_cuda(torch: Any) -> None:
    for index in range(torch.cuda.device_count()):
        torch.cuda.synchronize(index)


def _move_inputs(inputs: Any, model: Any, torch: Any) -> dict[str, Any]:
    device = model.get_input_embeddings().weight.device
    moved = {}
    for name, value in inputs.items():
        if isinstance(value, torch.Tensor):
            if name in {"pixel_values", "pixel_values_videos"} and value.is_floating_point():
                moved[name] = value.to(device=device, dtype=model.dtype)
            else:
                moved[name] = value.to(device=device)
        else:
            moved[name] = value
    return moved


def _generate_with_cache_trace(model: Any, processor: Any, inputs: dict[str, Any], max_tokens: int, torch: Any) -> dict[str, Any]:
    input_ids = inputs.get("input_ids")
    if input_ids is None or input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("Utils probe requires exactly one input sequence.")
    if "attention_mask" not in inputs:
        inputs["attention_mask"] = torch.ones_like(input_ids)
    steps: list[dict[str, Any]] = []
    pending_started: list[float] = []
    cache_after_prefill = None

    def before_forward(_module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        del args, kwargs
        _sync_cuda(torch)
        pending_started.append(time.perf_counter())

    def after_forward(_module: Any, _args: tuple[Any, ...], _kwargs: dict[str, Any], output: Any) -> None:
        nonlocal cache_after_prefill
        _sync_cuda(torch)
        started = pending_started.pop() if pending_started else time.perf_counter()
        cache = getattr(output, "past_key_values", None)
        if cache is None:
            raise RuntimeError("Model forward did not return past_key_values.")
        snapshot = _cache_snapshot(cache, torch, include_layers=not steps)
        steps.append({
            "phase": "prefill" if not steps else "decode",
            "seconds": round(time.perf_counter() - started, 6),
            "generated_tokens": len(steps) + 1,
            "cache": _cache_snapshot(cache, torch, include_layers=False),
        })
        if len(steps) == 1:
            cache_after_prefill = snapshot

    started_all = time.perf_counter()
    pre_hook = model.register_forward_pre_hook(before_forward, with_kwargs=True)
    post_hook = model.register_forward_hook(after_forward, with_kwargs=True)
    try:
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                use_cache=True,
                return_dict_in_generate=True,
            )
    finally:
        pre_hook.remove()
        post_hook.remove()
    completion_ids = output.sequences[:, input_ids.shape[-1]:]
    text = processor.batch_decode(completion_ids, skip_special_tokens=True)[0].strip()
    cache_lengths = [step["cache"]["sequence_length"] for step in steps]
    return {
        "prompt_tokens": int(inputs["attention_mask"].shape[-1]),
        "completion_tokens": int(completion_ids.shape[-1]),
        "finish_reason": "length" if completion_ids.shape[-1] >= max_tokens else "stop",
        "answer_characters": len(text),
        "total_seconds": round(time.perf_counter() - started_all, 6),
        "prefill_seconds": steps[0]["seconds"] if steps else None,
        "decode_seconds": [step["seconds"] for step in steps[1:]],
        "steps": steps,
        "cache_after_prefill": cache_after_prefill,
        "cache_grows_one_token_per_decode": (
            len(cache_lengths) <= 1 or all(right - left == 1 for left, right in zip(cache_lengths, cache_lengths[1:]))
        ),
        "cached_generation_observed": bool(steps and cache_after_prefill and cache_after_prefill.get("layer_count", 0) > 0),
    }


def main() -> int:
    args = _parse_args()
    input_path = _resolve_path(args.image or args.video)
    output_path = _resolve_path(args.output)
    config_path = _resolve_path(args.config)
    report: dict[str, Any] = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "project_alias": "mage-vl-4b",
        "input_kind": "image" if args.image else "video",
        "input_name": input_path.name,
        "offline_mode": True,
        "steps": [],
    }
    exit_code = 0
    model = None
    try:
        if not input_path.is_file():
            raise FileNotFoundError(f"Input file does not exist: {input_path}")
        config = _read_config(config_path)
        model_path = _resolve_model_snapshot(args.model_path)
        model_config = config["model"]
        runtime = config.get("runtime", {})
        hf_home_value = os.environ.get("HF_HOME") or runtime.get("hf_home")
        if hf_home_value:
            hf_home = Path(os.path.expandvars(os.path.expanduser(str(hf_home_value))))
            os.environ["HF_HOME"] = str(_resolve_path(hf_home))
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("The current inference service is configured for CUDA; no model was loaded.")
        max_allowed = int(config["api"].get("max_new_tokens", config["generation"]["max_tokens_default"]))
        max_new_tokens = (
            int(config["generation"]["max_tokens_default"])
            if args.max_new_tokens is None else args.max_new_tokens
        )
        if not 1 <= max_new_tokens <= max_allowed:
            raise ValueError(f"max_new_tokens must be between 1 and api.max_new_tokens={max_allowed}.")

        report["upstream_model_id"] = model_config["id"]
        report["configured_revision"] = model_config["revision"]
        report["generation_config"] = {
            "max_tokens_default": int(config["generation"]["max_tokens_default"]),
            "max_new_tokens": max_new_tokens,
        }
        report["environment"] = {
            "python": sys.version.split()[0],
            "torch": _version("torch"),
            "transformers": _version("transformers"),
            "huggingface_hub": _version("huggingface-hub"),
            "pyyaml": _version("PyYAML"),
            "inference_config": str(config_path),
            "configured_local_files_only": bool(runtime.get("local_files_only", False)),
            "forced_local_files_only": True,
        }
        report["resolved_model_path"] = str(model_path)
        load_started = time.perf_counter()
        model, processor = _load_model(config, model_path, torch)
        report["model_load_seconds"] = round(time.perf_counter() - load_started, 4)
        report["model_structure"] = _capture_model_structure(model)
        report["gpu_after_load"] = _gpu_snapshot(torch)

        prepare_started = time.perf_counter()
        if args.image:
            image = _load_pil_image(input_path, config)
            messages = [{"role": "user", "content": [
                {"type": "text", "text": args.question}, {"type": "image"},
            ]}]
            prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[prompt], images=[image], return_tensors="pt")
            report["input_metadata"] = {"width": image.width, "height": image.height}
        else:
            frames, metadata = _sample_video(input_path, config)
            messages = [{"role": "user", "content": [
                {"type": "video"}, {"type": "text", "text": args.question},
            ]}]
            prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[prompt], videos=[frames], return_tensors="pt", padding=True)
            report["input_metadata"] = metadata

        model_inputs = _move_inputs(inputs, model, torch)
        report["input_preparation_seconds"] = round(time.perf_counter() - prepare_started, 6)
        report["prepared_model_inputs"] = _input_snapshot(model_inputs, torch)
        for index in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(index)
        _sync_cuda(torch)
        report["generation"] = _generate_with_cache_trace(model, processor, model_inputs, max_new_tokens, torch)
        _sync_cuda(torch)
        report["gpu_after_generation"] = _gpu_snapshot(torch)
    except Exception as exc:
        exit_code = 1
        report["stepwise_compatible"] = False
        report["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
    finally:
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
        print(f"Utils report written to {output_path}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
