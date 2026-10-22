"""離線擷取 Qwen3-0.6B 模型結構、文字生成及 GPU 資訊。"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用明確指定的本機權重測試 Qwen3-0.6B。")
    parser.add_argument("--config", required=True, type=Path, help="必須明確指定探索 YAML。")
    parser.add_argument("--model-path", required=True, type=Path, help="必須明確指定本機模型快照。")
    parser.add_argument("--prompt", required=True, help="送入 Qwen3 chat template 的使用者文字。")
    parser.add_argument("--output", required=True, type=Path, help="必須明確指定 JSON 報告路徑。")
    parser.add_argument("--max-new-tokens", type=int, help="覆寫 YAML 中 generation.max_new_tokens。")
    return parser.parse_args()


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _load_config(path: Path) -> dict[str, Any]:
    import yaml

    config = yaml.safe_load(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not isinstance(config.get("model"), dict):
        raise ValueError("Qwen3 test YAML must contain a model mapping.")
    if not isinstance(config.get("generation"), dict):
        raise ValueError("Qwen3 test YAML must contain a generation mapping.")
    if int(config["generation"].get("max_new_tokens", 0)) < 1:
        raise ValueError("generation.max_new_tokens must be positive.")
    return config


def main() -> int:
    args = _arguments()
    settings = _load_config(args.config)
    snapshot = args.model_path.expanduser().resolve()
    if not snapshot.is_dir() or not (snapshot / "config.json").is_file():
        raise FileNotFoundError(f"Local model snapshot is incomplete: {snapshot}")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_settings = settings["model"]
    generation = settings["generation"]
    max_new_tokens = int(args.max_new_tokens or generation["max_new_tokens"])
    if max_new_tokens < 1:
        raise ValueError("max-new-tokens must be positive.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; this probe requires a GPU.")

    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        snapshot,
        revision=str(model_settings["revision"]),
        local_files_only=True,
        use_fast=True,
    )
    kwargs: dict[str, Any] = {
        "revision": str(model_settings["revision"]),
        "local_files_only": True,
        "device_map": str(model_settings.get("device_map", "auto")),
        "dtype": getattr(torch, str(model_settings.get("dtype", "bfloat16"))),
        "use_safetensors": True,
    }
    if model_settings.get("attn_implementation"):
        kwargs["attn_implementation"] = str(model_settings["attn_implementation"])
    model = AutoModelForCausalLM.from_pretrained(snapshot, **kwargs).eval()
    load_seconds = time.perf_counter() - started

    input_data = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}],
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=bool(generation.get("enable_thinking", False)),
    )
    input_device = model.get_input_embeddings().weight.device
    input_data = {key: value.to(input_device) for key, value in input_data.items() if isinstance(value, torch.Tensor)}
    prompt_tokens = int(input_data["input_ids"].shape[-1])
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "use_cache": True,
        "do_sample": float(generation.get("temperature", 0.0)) > 0,
    }
    if generation_kwargs["do_sample"]:
        generation_kwargs["temperature"] = float(generation["temperature"])
        generation_kwargs["top_p"] = float(generation.get("top_p", 1.0))
        generation_kwargs["top_k"] = int(generation.get("top_k", 0))
    generate_started = time.perf_counter()
    with torch.inference_mode():
        output_ids = model.generate(**input_data, **generation_kwargs)
    torch.cuda.synchronize()
    generation_seconds = time.perf_counter() - generate_started
    generated_ids = output_ids[0, prompt_tokens:]
    text = tokenizer.decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    model_config = model.config
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": {
            "repo_id": str(model_settings["id"]),
            "revision": str(model_settings["revision"]),
            "snapshot": str(snapshot),
            "model_class": f"{type(model).__module__}.{type(model).__name__}",
            "model_type": getattr(model_config, "model_type", None),
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "dtype": str(next(model.parameters()).dtype),
            "device_map": getattr(model, "hf_device_map", None),
            "layers": getattr(model_config, "num_hidden_layers", None),
            "attention_heads": getattr(model_config, "num_attention_heads", None),
            "kv_heads": getattr(model_config, "num_key_value_heads", None),
            "hidden_size": getattr(model_config, "hidden_size", None),
            "max_position_embeddings": getattr(model_config, "max_position_embeddings", None),
        },
        "runtime": {
            "python_package_transformers": _package_version("transformers"),
            "python_package_torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(0),
            "vram_total_bytes": int(torch.cuda.get_device_properties(0).total_memory),
            "load_seconds": round(load_seconds, 4),
            "generation_seconds": round(generation_seconds, 4),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": int(generated_ids.numel()),
            "completion_tokens_per_second": round(generated_ids.numel() / max(generation_seconds, 1e-9), 4),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        },
        "input_prompt": args.prompt,
        "generated_text": text,
    }
    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"model: {report['model']['repo_id']}@{report['model']['revision']}")
    print(f"snapshot: {snapshot}")
    print(f"prompt_tokens: {prompt_tokens}")
    print(f"completion_tokens: {report['runtime']['completion_tokens']}")
    print(f"generation_seconds: {generation_seconds:.3f}")
    print(f"completion_tokens_per_second: {report['runtime']['completion_tokens_per_second']}")
    print(f"peak_allocated_bytes: {report['runtime']['peak_allocated_bytes']}")
    print(f"generated_text: {text}")
    print(f"report: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
