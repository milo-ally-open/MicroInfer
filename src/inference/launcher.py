#!/usr/bin/env python3
"""推論服務的中央啟動器與模型設定選擇器。"""

from __future__ import annotations

import argparse
import ast
import importlib
import ipaddress
import math
import os
import re
import sys
from pathlib import Path
from typing import Any

import yaml


INFERENCE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = INFERENCE_ROOT.parents[1]
REGISTRY_PATH = INFERENCE_ROOT / "model_registry.yaml"
if str(INFERENCE_ROOT) not in sys.path:
    sys.path.insert(0, str(INFERENCE_ROOT))
PROTECTED_ENV = {
    "CUDA_VISIBLE_DEVICES",
    "HF_HOME",
    "HF_ENDPOINT",
    "HF_HUB_DISABLE_XET",
    "HF_HUB_DOWNLOAD_TIMEOUT",
    "INFERENCE_CONFIG",
    "INFERENCE_NETWORK_CONFIG",
    "INFERENCE_MODEL_ALIAS",
}


def _read_yaml(path: Path) -> dict[str, Any]:
    """讀取 YAML，並確認最上層內容是對應表。"""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"YAML must contain a mapping at the top level: {path}")
    return data


def _load_registry() -> dict[str, Any]:
    """載入模型登錄表，避免使用空白或格式錯誤的設定。"""
    registry = _read_yaml(REGISTRY_PATH)
    models = registry.get("models")
    if not isinstance(models, dict) or not models:
        raise ValueError(f"No model entries are configured in {REGISTRY_PATH}")
    return registry


def _validate_model_registration(alias: str, model: dict[str, Any], default_backend: str) -> dict[str, Any]:
    """檢查模型登錄與 API 能力宣告，不載入模型工廠或權重。"""
    profile = model.get("integration_profile")
    if profile not in {"standard", "custom"}:
        raise ValueError("integration_profile must be standard or custom.")
    backend = str(model.get("backend", default_backend)).strip()
    if not backend:
        raise ValueError("backend or default_backend must be configured.")
    inference_path = _resolve_path(str(model.get("inference_config", "")), "Inference configuration")
    inference_config = _read_yaml(inference_path)
    network_path = _resolve_path(str(model.get("network_config", "")), "Network configuration")
    network_config = _read_yaml(network_path)
    if not isinstance(network_config.get("api"), dict):
        raise ValueError("network configuration must contain an api mapping.")
    media_value = model.get("media_config")
    if media_value:
        media_path = _resolve_path(str(media_value), "Media configuration")
        _read_yaml(media_path)

    components = model.get("components", {})
    if not isinstance(components, dict):
        raise ValueError("components must be a mapping.")
    component_sources: dict[str, tuple[Path, str]] = {}
    for name in ("runner", "adapter"):
        factory = components.get(name)
        if factory is None and profile == "standard":
            default_source = INFERENCE_ROOT / "engine" / "standard_text.py"
            if not default_source.is_file():
                raise FileNotFoundError(f"Default standard {name} implementation not found: {default_source}")
            class_name = "StandardTextModelRunner" if name == "runner" else "StandardTextModelAdapter"
            component_sources[name] = (default_source, class_name)
        elif factory is None:
            raise ValueError(f"custom profile requires components.{name}.")
        else:
            source = _resolve_factory_source(str(factory), name)
            _, _, class_name = str(factory).partition(":")
            component_sources[name] = (source, class_name)

    api_settings = model.get("api")
    if not isinstance(api_settings, dict) or not str(api_settings.get("model_id", "")).strip():
        raise ValueError("api.model_id must be a non-empty string.")
    capabilities = api_settings.get("capabilities")
    if not isinstance(capabilities, dict):
        raise ValueError("api.capabilities must be a mapping.")
    chat_inputs = capabilities.get("chat_inputs")
    if (
        not isinstance(chat_inputs, list)
        or not chat_inputs
        or any(not isinstance(item, str) or item not in {"text", "image"} for item in chat_inputs)
    ):
        raise ValueError("api.capabilities.chat_inputs must contain text and/or image.")
    if len(set(chat_inputs)) != len(chat_inputs):
        raise ValueError("api.capabilities.chat_inputs must not contain duplicates.")
    for name in ("streaming", "video_upload", "live_video"):
        if not isinstance(capabilities.get(name), bool):
            raise ValueError(f"api.capabilities.{name} must be a boolean.")
    output_mode = capabilities.get("output_mode")
    if output_mode not in {"text_delta", "token_ids"}:
        raise ValueError("api.capabilities.output_mode must be text_delta or token_ids.")

    input_processor = api_settings.get("input_processor")
    if input_processor:
        _resolve_factory_source(str(input_processor), "input_processor")
    if output_mode == "token_ids" and not input_processor:
        if profile != "standard":
            raise ValueError("token_ids output requires api.input_processor for detokenization.")
        if not (INFERENCE_ROOT / "api" / "default_chat_processor.py").is_file():
            raise FileNotFoundError("Default standard chat input processor is missing.")
        processor_source = INFERENCE_ROOT / "api" / "default_chat_processor.py"
        processor_class = "DefaultChatInputProcessor"
    elif input_processor:
        processor_source = _resolve_factory_source(str(input_processor), "input_processor")
        _, _, processor_class = str(input_processor).partition(":")
    else:
        processor_source = None
        processor_class = None

    runner_methods = {
        "load", "gpu_status", "close", "begin_generation", "bind_generation_cache",
        "execute_batch", "finish_generation", "cancel_generation",
    }
    adapter_methods = {
        "health_info", "generation_options", "prepare_chat_generation", "bind_generation_cache",
        "execute_batch", "finish_chat_generation", "cancel_generation",
    }
    if capabilities["video_upload"]:
        adapter_methods |= {"prepare_video_generation", "finish_video_generation"}
    if capabilities["live_video"]:
        adapter_methods |= {
            "prepare_live_frames_generation", "prepare_live_video_generation", "finish_live_generation",
        }
    for name, required_methods in (("runner", runner_methods), ("adapter", adapter_methods)):
        source, class_name = component_sources[name]
        missing = required_methods - _factory_class_methods(source, class_name)
        if missing:
            raise ValueError(f"{name} factory {class_name} is missing methods: {', '.join(sorted(missing))}.")
    if processor_source is not None and processor_class is not None:
        required_processor_methods = {"start", "prepare_chat", "close"}
        if output_mode == "token_ids":
            required_processor_methods |= {"decode_tokens", "create_stream_output_processor"}
        missing = required_processor_methods - _factory_class_methods(processor_source, processor_class)
        if missing:
            raise ValueError(
                f"input_processor factory {processor_class} is missing methods: {', '.join(sorted(missing))}."
            )
    inference_api = inference_config.get("api")
    if not isinstance(inference_api, dict):
        raise ValueError("inference configuration must contain an api mapping.")
    if capabilities["video_upload"] and any(
        type(inference_api.get(key)) is not int or inference_api[key] <= 0
        for key in ("max_video_bytes", "max_video_request_bytes")
    ):
        raise ValueError("video_upload requires positive api.max_video_bytes and api.max_video_request_bytes.")
    if capabilities["live_video"] and (
        type(inference_api.get("max_live_request_bytes")) is not int
        or inference_api["max_live_request_bytes"] <= 0
    ):
        raise ValueError("live_video requires api.max_live_request_bytes.")
    live_manager = api_settings.get("live_session_manager")
    if capabilities["live_video"]:
        if not live_manager:
            raise ValueError("live_video requires api.live_session_manager.")
        _resolve_factory_source(str(live_manager), "live_session_manager")
        if not media_value:
            raise ValueError("live_video requires media_config.")
    elif live_manager:
        _resolve_factory_source(str(live_manager), "live_session_manager")

    if profile == "standard":
        if chat_inputs != ["text"] or output_mode != "token_ids" or capabilities["video_upload"] or capabilities["live_video"]:
            raise ValueError(
                "standard profile supports text chat and token_ids only; video/live or multimodal models require custom profile."
            )
        model_settings = inference_config.get("model")
        if (
            not isinstance(model_settings, dict)
            or not str(model_settings.get("id", "")).strip()
            or not re.fullmatch(r"[0-9a-fA-F]{40}", str(model_settings.get("revision", "")))
        ):
            raise ValueError("standard profile requires model.id and a pinned model.revision.")
        if model_settings.get("device_map") not in (None, ""):
            raise ValueError("standard profile supports one configured device; use custom profile for device_map.")
        if model_settings.get("dtype", "bfloat16") not in {"float16", "bfloat16", "float32"}:
            raise ValueError("standard profile model.dtype must be float16, bfloat16, or float32.")
        runtime_settings = inference_config.get("runtime", {})
        generation_settings = inference_config.get("generation")
        if not isinstance(runtime_settings, dict) or not isinstance(generation_settings, dict):
            raise ValueError("standard profile requires runtime and generation YAML mappings.")
        if runtime_settings.get("platform", "torch_cuda") != "torch_cuda":
            raise ValueError("standard profile currently supports the registered torch_cuda platform only.")
        if not isinstance(runtime_settings.get("local_files_only", True), bool):
            raise ValueError("runtime.local_files_only must be a boolean.")
        if model_settings.get("trust_remote_code", False) not in (True, False):
            raise ValueError("model.trust_remote_code must be a boolean.")

    return {
        "profile": profile,
        "backend": backend,
        "chat_inputs": ",".join(chat_inputs),
        "output_mode": output_mode,
        "video_upload": capabilities["video_upload"],
        "live_video": capabilities["live_video"],
    }


def _resolve_path(value: str, description: str) -> Path:
    """將相對路徑解析為專案內的絕對路徑，並確認檔案存在。"""
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{description} not found: {path}")
    return path


def _list_models(registry: dict[str, Any]) -> None:
    """列出模型狀態、設定路徑與 Runner／Adapter 工廠。"""
    print("MODEL\tPROFILE\tBACKEND\tSTATUS\tINPUTS\tOUTPUT\tVIDEO\tLIVE\tINFERENCE CONFIG\tNETWORK CONFIG\tRUNNER\tADAPTER")
    for alias, model in sorted(registry["models"].items()):
        if not isinstance(model, dict):
            print(f"{alias}\t\tinvalid-registry-entry")
            continue
        display_name = model.get("display_name", alias)
        paths = [model.get("inference_config", ""), model.get("network_config", "")]
        api_settings = model.get("api", {}) if isinstance(model.get("api", {}), dict) else {}
        capabilities = api_settings.get("capabilities", {}) if isinstance(api_settings, dict) else {}
        status = "valid"
        try:
            validated = _validate_model_registration(alias, model, str(registry.get("default_backend", "")))
        except (FileNotFoundError, OSError, ValueError):
            status = "invalid-registry-entry"
        components = model.get("components", {}) if isinstance(model.get("components", {}), dict) else {}
        print(
            f"{alias} ({display_name})\t{validated['profile'] if status == 'valid' else model.get('integration_profile', '')}\t"
            f"{validated['backend'] if status == 'valid' else model.get('backend', registry.get('default_backend', ''))}\t{status}\t"
            f"{validated['chat_inputs'] if status == 'valid' else ''}\t{validated['output_mode'] if status == 'valid' else ''}\t"
            f"{validated['video_upload'] if status == 'valid' else ''}\t{validated['live_video'] if status == 'valid' else ''}\t"
            f"{paths[0]}\t{paths[1]}\t"
            f"{components.get('runner', '')}\t{components.get('adapter', '')}"
        )


def _resolve_factory_source(factory_path: str, name: str) -> Path:
    """確認登錄工廠使用可載入的 Python module:attribute 路徑。"""
    module_name, separator, attribute = factory_path.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError(f"Model {name} factory must use 'module:attribute' notation.")
    module_path = INFERENCE_ROOT.joinpath(*module_name.split("."))
    source = module_path.with_suffix(".py")
    if source.is_file():
        return source
    initializer = module_path / "__init__.py"
    if initializer.is_file():
        return initializer
    raise FileNotFoundError(f"Model {name} factory module not found: {module_name}")


def _factory_class_methods(source: Path, class_name: str) -> set[str]:
    """以 AST 檢查 factory class 方法，不 import 模型程式。"""
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return {
                member.name
                for member in node.body
                if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
    raise ValueError(f"Factory class {class_name} is not declared in {source}.")


def _resolve_selection(
    args: argparse.Namespace, registry: dict[str, Any]
) -> tuple[str, str, Path, Path, Path | None]:
    """依命令列與登錄表選出模型及其設定檔。"""
    model_alias = args.model or str(registry.get("default_model", ""))
    if not model_alias:
        raise ValueError("Pass --model or set default_model in model_registry.yaml.")
    models = registry["models"]
    if model_alias not in models:
        raise ValueError(f"Unknown model config '{model_alias}'. Available: {', '.join(sorted(models))}")

    model_spec = models[model_alias]
    if not isinstance(model_spec, dict):
        raise ValueError(f"Model registry entry '{model_alias}' must be a mapping.")
    _validate_model_registration(model_alias, model_spec, str(registry.get("default_backend", "")))
    backend = str(model_spec.get("backend", registry.get("default_backend", "")))
    if not backend:
        raise ValueError("Set default_backend in model_registry.yaml.")

    inference_config_path = _resolve_path(
        str(args.inference_config or model_spec.get("inference_config", "")), "Inference configuration"
    )
    network_config_path = _resolve_path(
        str(args.network_config or model_spec.get("network_config", "")), "Network configuration"
    )
    api_settings = model_spec.get("api", {})
    media_config_value = args.media_config or model_spec.get("media_config")
    if media_config_value:
        media_config_path = _resolve_path(str(media_config_value), "Media configuration")
    elif api_settings.get("capabilities", {}).get("live_video", False):
        raise ValueError(f"Model registry entry '{model_alias}' enables Live sessions but has no media_config.")
    else:
        media_config_path = None
    return model_alias, backend, inference_config_path, network_config_path, media_config_path


def _set_runtime_environment(
    model_alias: str, config_path: Path, network_config_path: Path, config: dict[str, Any]
) -> None:
    """套用模型設定中的環境變數，讓服務啟動時讀取正確執行環境。"""
    runtime = config.get("runtime", {})
    os.environ["INFERENCE_CONFIG"] = str(config_path)
    os.environ["INFERENCE_NETWORK_CONFIG"] = str(network_config_path)
    os.environ["INFERENCE_MODEL_ALIAS"] = model_alias

    if runtime.get("hf_home"):
        hf_home = Path(os.path.expandvars(os.path.expanduser(str(runtime["hf_home"]))))
        if not hf_home.is_absolute():
            hf_home = PROJECT_ROOT / hf_home
        os.environ["HF_HOME"] = str(hf_home.resolve())
    if runtime.get("hf_endpoint"):
        os.environ["HF_ENDPOINT"] = str(runtime["hf_endpoint"])
    if runtime.get("hf_hub_disable_xet", False):
        os.environ["HF_HUB_DISABLE_XET"] = "1"
    if runtime.get("hf_hub_download_timeout_seconds") is not None:
        os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = str(int(runtime["hf_hub_download_timeout_seconds"]))

    custom_environment = runtime.get("environment", {})
    if not isinstance(custom_environment, dict):
        raise ValueError("runtime.environment must be a mapping of environment variable names to values.")
    for name, value in custom_environment.items():
        if not isinstance(name, str) or not re.fullmatch(r"[A-Z_][A-Z0-9_]*", name):
            raise ValueError(f"Invalid environment variable name: {name!r}.")
        if name in PROTECTED_ENV:
            raise ValueError(f"Set {name} through its dedicated runtime setting, not runtime.environment.")
        if isinstance(value, bool):
            rendered = "1" if value else "0"
        elif isinstance(value, (str, int, float)) and (not isinstance(value, float) or math.isfinite(value)):
            rendered = str(value)
        else:
            raise ValueError(f"runtime.environment.{name} must be a string, number, or boolean.")
        os.environ[name] = os.path.expandvars(rendered)

    visible_devices = runtime.get("cuda_visible_devices")
    if visible_devices is not None:
        visible_devices = str(visible_devices).strip()
        if not re.fullmatch(r"[0-9]+(?:,[0-9]+)*", visible_devices):
            raise ValueError("runtime.cuda_visible_devices must be a comma-separated list of GPU indices.")
        os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices


def _load_service_app() -> Any:
    """載入唯一的共用 API app；模型差異由登錄表提供。"""
    app_module = importlib.import_module("api.service_app")
    if not hasattr(app_module, "app"):
        raise AttributeError("The shared API service entrypoint does not export 'app'.")
    return app_module.app


def _add_runtime_overrides(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    parser.add_argument("--cuda-visible-devices")


def main() -> None:
    """解析啟動參數、載入指定模型設定，並啟動常駐 API 服務。"""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="列出已登錄模型、設定檔及元件工廠狀態。")
    validate_parser = commands.add_parser("validate", help="預檢模型登錄、設定和能力；不載入權重或啟動 GPU。")
    validate_parser.add_argument("--model", action="append", help="指定模型別名；可重複。預設檢查全部模型。")
    start_parser = commands.add_parser("start", help="啟動一組已登錄的模型設定。")
    start_parser.add_argument("--model", help="model_registry.yaml 中的模型別名；預設使用 default_model。")
    start_parser.add_argument("--inference-config", type=Path, help="覆寫登錄表中的推論設定 YAML。")
    start_parser.add_argument("--network-config", type=Path, help="覆寫登錄表中的網路設定 YAML。")
    start_parser.add_argument("--media-config", type=Path, help="覆寫登錄表中的 MediaMTX 設定 YAML。")
    _add_runtime_overrides(start_parser)
    args = parser.parse_args()

    registry = _load_registry()
    if args.command == "list":
        _list_models(registry)
        return
    if args.command == "validate":
        selected_aliases = args.model or sorted(registry["models"])
        for alias in selected_aliases:
            model = registry["models"].get(alias)
            if not isinstance(model, dict):
                raise ValueError(f"Unknown model config '{alias}'.")
            result = _validate_model_registration(alias, model, str(registry.get("default_backend", "")))
            print(
                f"{alias}: valid profile={result['profile']} backend={result['backend']} "
                f"inputs={result['chat_inputs']} output={result['output_mode']} "
                f"video_upload={result['video_upload']} live_video={result['live_video']}"
            )
        print("Preflight completed without loading model weights or initializing a GPU runtime.")
        return

    model_alias, backend, config_path, network_config_path, media_config_path = _resolve_selection(args, registry)
    config = _read_yaml(config_path)
    network_config = _read_yaml(network_config_path)
    api_network = network_config.get("api")
    if not isinstance(api_network, dict):
        raise ValueError("network.api must be a mapping in the selected network YAML.")
    runtime = config.setdefault("runtime", {})
    if args.host is not None:
        api_network["host"] = args.host
    if args.port is not None:
        api_network["port"] = args.port
    if args.cuda_visible_devices is not None:
        runtime["cuda_visible_devices"] = args.cuda_visible_devices
    # 必須在載入模型服務前設定環境變數，確保 CUDA 與 HF 快取設定生效。
    _set_runtime_environment(model_alias, config_path, network_config_path, config)

    host = str(api_network.get("host", "127.0.0.1"))
    if not host or host != host.strip():
        raise ValueError("network.api.host must be an IP address or hostname.")
    try:
        ipaddress.ip_address(host)
    except ValueError:
        if not re.fullmatch(r"[A-Za-z0-9._-]+", host):
            raise ValueError("network.api.host must be an IP address or hostname.")
    port = int(api_network.get("port", 8001))
    if not 1 <= port <= 65535:
        raise ValueError("network.api.port must be between 1 and 65535.")
    api_workers = int(runtime.get("uvicorn_workers", 1))
    if api_workers != 1:
        raise ValueError(
            "runtime.uvicorn_workers must be 1 while each API Server owns one Engine Core and GPU Worker process."
        )

    print(
        f"Starting model '{model_alias}' with backend '{backend}' using inference config {config_path}, "
        f"network config {network_config_path}, and media config {media_config_path}"
    )
    import uvicorn

    # API Server 單程序常駐；Engine Core 與 GPU Worker 各自管理獨立子程序。
    uvicorn.run(
        _load_service_app(),
        host=host,
        port=port,
        workers=api_workers,
        log_level=str(runtime.get("uvicorn_log_level", "info")),
        access_log=bool(runtime.get("uvicorn_access_log", True)),
        timeout_keep_alive=int(runtime.get("uvicorn_timeout_keep_alive_seconds", 5)),
    )


if __name__ == "__main__":
    main()
