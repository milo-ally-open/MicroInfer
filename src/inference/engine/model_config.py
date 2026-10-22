"""載入啟動器選定的模型登錄、YAML 設定與模型工廠。"""

from __future__ import annotations

import importlib
import os
from pathlib import Path
from typing import Any

import yaml


INFERENCE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = INFERENCE_ROOT.parents[1]


def read_yaml(path: str | Path) -> dict[str, Any]:
    """讀取 YAML mapping，拒絕空白或格式錯誤的設定。"""
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"YAML must contain a mapping at the top level: {path}")
    return value


def selected_model() -> tuple[str, dict[str, Any]]:
    """依啟動器提供的模型別名載入登錄項目。"""
    alias = os.environ.get("INFERENCE_MODEL_ALIAS", "").strip()
    if not alias:
        raise RuntimeError("Start the inference service through launcher.py to select a model.")
    registry = read_yaml(INFERENCE_ROOT / "model_registry.yaml")
    models = registry.get("models")
    if not isinstance(models, dict) or not isinstance(models.get(alias), dict):
        raise ValueError(f"Model '{alias}' is missing from model_registry.yaml.")
    return alias, models[alias]


def selected_config(environment_name: str) -> dict[str, Any]:
    """讀取啟動器解析後放入環境的設定檔。"""
    path = os.environ.get(environment_name)
    if not path:
        raise RuntimeError(f"Set {environment_name} by starting the inference service through launcher.py.")
    return read_yaml(path)


def model_source(inference_config: dict[str, Any]) -> str:
    """依 YAML 選擇本機模型快照或上游 repo ID。"""
    model = inference_config.get("model")
    runtime = inference_config.get("runtime", {})
    if not isinstance(model, dict) or not isinstance(runtime, dict):
        raise ValueError("model and runtime settings must be mappings.")
    repo_id = str(model.get("id", "")).strip()
    if not repo_id:
        raise ValueError("model.id must be a non-empty upstream repository ID.")
    local_path = runtime.get("local_model_path")
    if local_path is None:
        return repo_id
    path = Path(os.path.expandvars(os.path.expanduser(str(local_path))))
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return str(path.resolve())


def load_factory(factory_path: str) -> Any:
    """載入登錄表指定的 `module:attribute` 元件工廠。"""
    if not isinstance(factory_path, str):
        raise TypeError("Model factory path must be a string.")
    module_name, separator, attribute = factory_path.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError("Model factory must use 'module:attribute' notation.")
    factory = getattr(importlib.import_module(module_name), attribute)
    if not callable(factory):
        raise TypeError(f"Model factory is not callable: {factory_path}")
    return factory


def model_api_settings(model_spec: dict[str, Any]) -> dict[str, Any]:
    settings = model_spec.get("api", {})
    if not isinstance(settings, dict):
        raise ValueError("model_registry.yaml api entries must be mappings.")
    return settings


def model_component(model_spec: dict[str, Any], name: str, *, required: bool = True) -> str | None:
    components = model_spec.get("components", {})
    if not isinstance(components, dict):
        raise ValueError("model_registry.yaml components entries must be mappings.")
    value = components.get(name)
    if value is None and not required:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Model registry entry must define components.{name} as a factory path.")
    return value


def validate_runtime_components(model_spec: dict[str, Any], runner: Any, adapter: Any) -> None:
    """在 Worker 載入權重前，依宣告能力檢查 Runner／Adapter 方法。"""
    runner_methods = {
        "load",
        "gpu_status",
        "close",
        "begin_generation",
        "bind_generation_cache",
        "execute_batch",
        "finish_generation",
        "cancel_generation",
    }
    adapter_methods = {
        "health_info",
        "generation_options",
        "prepare_chat_generation",
        "bind_generation_cache",
        "execute_batch",
        "finish_chat_generation",
        "cancel_generation",
    }
    api_settings = model_api_settings(model_spec)
    capabilities = api_settings.get("capabilities", {})
    if not isinstance(capabilities, dict):
        raise ValueError("model api.capabilities must be a mapping.")
    if capabilities.get("video_upload"):
        adapter_methods |= {"prepare_video_generation", "finish_video_generation"}
    if capabilities.get("live_video"):
        adapter_methods |= {
            "prepare_live_frames_generation",
            "prepare_live_video_generation",
            "finish_live_generation",
        }
    missing_runner = sorted(name for name in runner_methods if not callable(getattr(runner, name, None)))
    missing_adapter = sorted(name for name in adapter_methods if not callable(getattr(adapter, name, None)))
    if missing_runner or missing_adapter:
        raise TypeError(
            "Model component contract mismatch: "
            f"runner missing={missing_runner}, adapter missing={missing_adapter}."
        )
