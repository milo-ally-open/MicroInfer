"""载入启动器选定的模型登录、YAML 设定与模型工厂。"""

from __future__ import annotations

import importlib
import os
from pathlib import Path
from typing import Any

import yaml


INFERENCE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = INFERENCE_ROOT.parents[1]
MODELS_ROOT = INFERENCE_ROOT / "models"


def read_yaml(path: str | Path) -> dict[str, Any]:
    """读取 YAML mapping，拒绝空白或格式错误的设定。"""
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"YAML must contain a mapping at the top level: {path}")
    return value


def selected_model() -> tuple[str, dict[str, Any]]:
    """载入启动器选定的模型目录清单。"""
    alias = os.environ.get("INFERENCE_MODEL_ALIAS", "").strip()
    if not alias:
        raise RuntimeError("Start the inference service through launcher.py to select a model.")
    manifest_value = os.environ.get("INFERENCE_MODEL_MANIFEST", "").strip()
    manifest_path = Path(manifest_value) if manifest_value else model_manifest_path(alias)
    if not manifest_path.is_absolute():
        manifest_path = PROJECT_ROOT / manifest_path
    model = read_yaml(manifest_path)
    if model.get("alias") != alias:
        raise ValueError(f"Model manifest alias does not match selected model '{alias}': {manifest_path}")
    model["_manifest_path"] = str(manifest_path.resolve())
    return alias, model


def model_manifest_path(alias: str) -> Path:
    """依模型别名寻找模型目录内的唯一 `configs/model.yaml`。"""
    matches = []
    for manifest in MODELS_ROOT.glob("*/configs/model.yaml"):
        try:
            data = read_yaml(manifest)
        except (OSError, ValueError):
            continue
        if data.get("alias") == alias:
            matches.append(manifest.resolve())
    if not matches:
        raise ValueError(f"Model '{alias}' has no configs/model.yaml manifest under {MODELS_ROOT}.")
    if len(matches) > 1:
        raise ValueError(f"Model alias '{alias}' is declared by multiple model manifests: {matches}")
    return matches[0]


def selected_config(environment_name: str) -> dict[str, Any]:
    """读取启动器解析后放入环境的设定档。"""
    path = os.environ.get(environment_name)
    if not path:
        raise RuntimeError(f"Set {environment_name} by starting the inference service through launcher.py.")
    return read_yaml(path)


def model_source(inference_config: dict[str, Any]) -> str:
    """依 YAML 选择本机模型快照或上游 repo ID。"""
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
    """载入登录表指定的 `module:attribute` 元件工厂。"""
    if not isinstance(factory_path, str):
        raise TypeError("Model factory path must be a string.")
    module_name, separator, attribute = factory_path.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError("Model factory must use 'module:attribute' notation.")
    factory = getattr(importlib.import_module(module_name), attribute)
    if not callable(factory):
        raise TypeError(f"Model factory is not callable: {factory_path}")
    return factory


def validate_runtime_components(model_spec: dict[str, Any], runtime: Any) -> None:
    """在 Worker 载入权重前，依宣告能力检查单一模型 Runtime 契约。"""
    required_methods = {
        "load", "gpu_status", "close", "health_info", "generation_options",
        "prepare_chat_generation", "bind_generation_cache", "execute_batch",
        "finish_chat_generation", "cancel_generation",
    }
    api_settings = model_api_settings(model_spec)
    capabilities = api_settings.get("capabilities", {})
    if not isinstance(capabilities, dict):
        raise ValueError("model api.capabilities must be a mapping.")
    if capabilities.get("video_upload"):
        required_methods |= {"prepare_video_generation", "finish_video_generation"}
    if capabilities.get("live_video"):
        required_methods |= {
            "prepare_live_frames_generation", "prepare_live_video_generation", "finish_live_generation",
        }
    missing = sorted(name for name in required_methods if not callable(getattr(runtime, name, None)))
    if missing:
        raise TypeError(f"Model Runtime contract mismatch: missing={missing}.")


def model_api_settings(model_spec: dict[str, Any]) -> dict[str, Any]:
    settings = model_spec.get("api", {})
    if not isinstance(settings, dict):
        raise ValueError("model.yaml api entries must be mappings.")
    return settings


def model_runtime_factory(model_spec: dict[str, Any]) -> str:
    value = model_spec.get("runtime_factory")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("model.yaml must define runtime_factory as a module:attribute path.")
    return value
