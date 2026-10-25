#!/usr/bin/env python3
"""推论服务的中央启动器与模型设定选择器。"""

from __future__ import annotations

import argparse
import ast
import importlib
import ipaddress
import math
import os
import re
from pathlib import Path
from typing import Any

import yaml


INFERENCE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = Path(
    os.environ.get("INFERENCE_PROJECT_ROOT", str(INFERENCE_ROOT.parents[1]))
).expanduser().resolve()
MODELS_ROOT = INFERENCE_ROOT / "models"
PROTECTED_ENV = {
    "CUDA_VISIBLE_DEVICES",
    "HF_HOME",
    "HF_ENDPOINT",
    "HF_HUB_DISABLE_XET",
    "HF_HUB_DOWNLOAD_TIMEOUT",
    "INFERENCE_CONFIG",
    "INFERENCE_NETWORK_CONFIG",
    "INFERENCE_PROJECT_ROOT",
    "INFERENCE_MODEL_ALIAS",
    "INFERENCE_MODEL_MANIFEST",
}


def _read_yaml(path: Path) -> dict[str, Any]:
    """读取 YAML，并确认最上层内容是对应表。"""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"YAML must contain a mapping at the top level: {path}")
    return data


def _discover_models() -> dict[str, dict[str, Any]]:
    """从各模型目录的 manifest 发现模型，不依赖全域登录档。"""
    models: dict[str, dict[str, Any]] = {}
    for manifest_path in sorted(MODELS_ROOT.glob("*/configs/model.yml")):
        model = _read_yaml(manifest_path)
        alias = model.get("alias")
        if not isinstance(alias, str) or not alias.strip():
            raise ValueError(f"Model manifest must define a non-empty alias: {manifest_path}")
        if alias in models:
            raise ValueError(f"Duplicate model alias '{alias}' in {manifest_path}")
        model["_manifest_path"] = str(manifest_path.resolve())
        models[alias] = model
    if not models:
        raise ValueError(f"No models/*/configs/model.yml manifests found under {MODELS_ROOT}")
    return models


def _validate_model_manifest(alias: str, model: dict[str, Any]) -> dict[str, Any]:
    """检查模型 manifest 与 API 能力宣告，不载入模型工厂或权重。"""
    profile = model.get("integration_profile")
    if profile not in {"standard", "custom"}:
        raise ValueError("integration_profile must be standard or custom.")
    backend = str(model.get("backend", "")).strip()
    if not backend:
        raise ValueError("Every model manifest must define backend in configs/model.yml.")
    manifest_path = Path(str(model.get("_manifest_path", ""))).resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Model manifest not found: {manifest_path}")
    inference_path = _resolve_path(
        str(model.get("inference_config", "")), "Inference configuration", base_dir=manifest_path.parent
    )
    inference_config = _read_yaml(inference_path)
    network_path = _resolve_path(
        str(model.get("network_config", "")), "Network configuration", base_dir=manifest_path.parent
    )
    network_config = _read_yaml(network_path)
    if not isinstance(network_config.get("api"), dict):
        raise ValueError("network configuration must contain an api mapping.")
    runtime_factory = model.get("runtime_factory")
    if not isinstance(runtime_factory, str) or not runtime_factory.strip():
        raise ValueError("Every model manifest must define runtime_factory in configs/model.yml.")
    runtime_source = _resolve_factory_source(runtime_factory, "runtime")
    _, _, runtime_symbol = runtime_factory.partition(":")
    if runtime_symbol not in _factory_symbols(runtime_source):
        raise ValueError(f"runtime factory symbol {runtime_symbol!r} is not declared in {runtime_source}.")

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
    for name in ("streaming", "tool_calling", "video_upload", "live_video"):
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
        if not (INFERENCE_ROOT / "api" / "processors" / "default_chat_processor.py").is_file():
            raise FileNotFoundError("Default standard chat input processor is missing.")
        processor_source = INFERENCE_ROOT / "api" / "processors" / "default_chat_processor.py"
        processor_class = "DefaultChatInputProcessor"
    elif input_processor:
        processor_source = _resolve_factory_source(str(input_processor), "input_processor")
        _, _, processor_class = str(input_processor).partition(":")
    else:
        processor_source = None
        processor_class = None

    if processor_source is not None and processor_class is not None:
        required_processor_methods = {"start", "prepare_chat", "close"}
        if output_mode == "token_ids":
            required_processor_methods |= {"decode_tokens", "create_stream_output_processor"}
        missing = required_processor_methods - _factory_class_methods(processor_source, processor_class)
        if missing:
            raise ValueError(
                f"input_processor factory {processor_class} is missing methods: {', '.join(sorted(missing))}."
            )

    tool_adapter_factory = api_settings.get("tool_call_adapter")
    if capabilities["tool_calling"]:
        if not isinstance(tool_adapter_factory, str) or not tool_adapter_factory.strip():
            raise ValueError("tool_calling requires api.tool_call_adapter in configs/model.yml.")
        tool_adapter_source = _resolve_factory_source(tool_adapter_factory, "tool_call_adapter")
        _, _, tool_adapter_class = tool_adapter_factory.partition(":")
        if tool_adapter_class not in _factory_symbols(tool_adapter_source):
            raise ValueError(f"tool-call adapter factory symbol {tool_adapter_class!r} is not declared in {tool_adapter_source}.")
        missing_tool_adapter_methods = {"parse_output", "create_stream_parser"} - _factory_class_methods(
            tool_adapter_source, tool_adapter_class
        )
        if missing_tool_adapter_methods:
            raise ValueError(
                f"tool-call adapter {tool_adapter_class} is missing methods: {', '.join(sorted(missing_tool_adapter_methods))}."
            )
    elif tool_adapter_factory:
        raise ValueError("api.tool_call_adapter must not be set when tool_calling capability is false.")
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
        "tool_calling": capabilities["tool_calling"],
        "video_upload": capabilities["video_upload"],
        "live_video": capabilities["live_video"],
    }


def _resolve_path(value: str, description: str, *, base_dir: Path | None = None) -> Path:
    """将相对路径解析为专案内的绝对路径，并确认档案存在。"""
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (base_dir or PROJECT_ROOT) / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{description} not found: {path}")
    return path


def _list_models(models: dict[str, dict[str, Any]]) -> None:
    """列出自模型 manifest 发现的模型与设定。"""
    print("MODEL\tPROFILE\tBACKEND\tSTATUS\tINPUTS\tOUTPUT\tTOOLS\tVIDEO\tLIVE\tMANIFEST")
    for alias, model in sorted(models.items()):
        if not isinstance(model, dict):
            print(f"{alias}\t\tinvalid-model-manifest")
            continue
        display_name = model.get("display_name", alias)
        manifest = model.get("_manifest_path", "")
        api_settings = model.get("api", {}) if isinstance(model.get("api", {}), dict) else {}
        capabilities = api_settings.get("capabilities", {}) if isinstance(api_settings, dict) else {}
        status = "valid"
        try:
            validated = _validate_model_manifest(alias, model)
        except (FileNotFoundError, OSError, ValueError):
            status = "invalid-model-manifest"
        print(
            f"{alias} ({display_name})\t{validated['profile'] if status == 'valid' else model.get('integration_profile', '')}\t"
            f"{validated['backend'] if status == 'valid' else model.get('backend', '')}\t{status}\t"
            f"{validated['chat_inputs'] if status == 'valid' else ''}\t{validated['output_mode'] if status == 'valid' else ''}\t"
            f"{validated['tool_calling'] if status == 'valid' else ''}\t{validated['video_upload'] if status == 'valid' else ''}\t"
            f"{validated['live_video'] if status == 'valid' else ''}\t"
            f"{manifest}"
        )


def _resolve_factory_source(factory_path: str, name: str) -> Path:
    """确认登录工厂使用可载入的 Python module:attribute 路径。"""
    module_name, separator, attribute = factory_path.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError(f"Model {name} factory must use 'module:attribute' notation.")
    module_parts = module_name.split(".")
    if module_parts[0] != "inference":
        raise ValueError(f"Model {name} factory must use the 'inference.*' package namespace.")
    module_path = INFERENCE_ROOT.joinpath(*module_parts[1:])
    source = module_path.with_suffix(".py")
    if source.is_file():
        return source
    initializer = module_path / "__init__.py"
    if initializer.is_file():
        return initializer
    raise FileNotFoundError(f"Model {name} factory module not found: {module_name}")


def _factory_class_methods(source: Path, class_name: str) -> set[str]:
    """以 AST 检查 factory class 方法，不 import 模型程式。"""
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return {
                member.name
                for member in node.body
                if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
    raise ValueError(f"Factory class {class_name} is not declared in {source}.")


def _factory_symbols(source: Path) -> set[str]:
    """以 AST 取得可由 YAML 指定的顶层工厂函式与类别名称。"""
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    return {
        node.name for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }


def _resolve_selection(
    args: argparse.Namespace, models: dict[str, dict[str, Any]]
) -> tuple[str, str, Path, Path]:
    """依命令列与各模型 manifest 选出单一模型及设定档。"""
    model_alias = args.model
    if not model_alias:
        raise ValueError("Pass --model to select a model manifest.")
    if model_alias not in models:
        raise ValueError(f"Unknown model config '{model_alias}'. Available: {', '.join(sorted(models))}")

    model_spec = models[model_alias]
    if not isinstance(model_spec, dict):
        raise ValueError(f"Model manifest '{model_alias}' must be a mapping.")
    _validate_model_manifest(model_alias, model_spec)
    backend = str(model_spec["backend"])

    manifest_path = Path(model_spec["_manifest_path"])
    inference_config_path = _resolve_path(
        str(args.inference_config or model_spec.get("inference_config", "")),
        "Inference configuration", base_dir=manifest_path.parent,
    )
    network_config_path = _resolve_path(
        str(args.network_config or model_spec.get("network_config", "")),
        "Network configuration", base_dir=manifest_path.parent,
    )
    return model_alias, backend, inference_config_path, network_config_path


def _set_runtime_environment(
    model_alias: str, manifest_path: Path, config_path: Path, network_config_path: Path, config: dict[str, Any]
) -> None:
    """套用模型设定中的环境变数，让服务启动时读取正确执行环境。"""
    runtime = config.get("runtime", {})
    os.environ["INFERENCE_CONFIG"] = str(config_path)
    os.environ["INFERENCE_NETWORK_CONFIG"] = str(network_config_path)
    os.environ["INFERENCE_MODEL_ALIAS"] = model_alias
    os.environ["INFERENCE_MODEL_MANIFEST"] = str(manifest_path.resolve())

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
    """载入唯一的共用 Network app；模型差异由所选 manifest 提供。"""
    app_module = importlib.import_module("inference.network.service")
    if not hasattr(app_module, "app"):
        raise AttributeError("The shared API service entrypoint does not export 'app'.")
    return app_module.app


def _add_runtime_overrides(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    parser.add_argument("--cuda-visible-devices")


def main() -> None:
    """解析启动参数、载入指定模型设定，并启动常驻 API 服务。"""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="列出模型目录中的 manifest 与能力。")
    validate_parser = commands.add_parser("validate", help="预检模型 manifest、设定和能力；不载入权重或启动 GPU。")
    validate_parser.add_argument("--model", action="append", help="指定模型别名；可重复。预设检查全部模型。")
    start_parser = commands.add_parser("start", help="启动指定模型 manifest 的推论 API 服务。")
    start_parser.add_argument("--model", required=True, help="要启动的模型 manifest alias。")
    start_parser.add_argument("--inference-config", type=Path, help="覆写模型 manifest 中的推论设定 YAML。")
    start_parser.add_argument("--network-config", type=Path, help="覆写模型 manifest 中的网路设定 YAML。")
    _add_runtime_overrides(start_parser)
    args = parser.parse_args()

    models = _discover_models()
    if args.command == "list":
        _list_models(models)
        return
    if args.command == "validate":
        selected_aliases = args.model or sorted(models)
        for alias in selected_aliases:
            model = models.get(alias)
            if not isinstance(model, dict):
                raise ValueError(f"Unknown model config '{alias}'.")
            result = _validate_model_manifest(alias, model)
            print(
                f"{alias}: valid profile={result['profile']} backend={result['backend']} "
                f"inputs={result['chat_inputs']} output={result['output_mode']} "
                f"video_upload={result['video_upload']} live_video={result['live_video']}"
            )
        print("Preflight completed without loading model weights or initializing a GPU runtime.")
        return

    model_alias, backend, config_path, network_config_path = _resolve_selection(args, models)
    manifest_path = Path(models[model_alias]["_manifest_path"])
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
    # 必须在载入模型服务前设定环境变数，确保 CUDA 与 HF 快取设定生效。
    _set_runtime_environment(model_alias, manifest_path, config_path, network_config_path, config)

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
        f"and network config {network_config_path}"
    )
    import uvicorn

    # API Server 单程序常驻；Engine Core 与 GPU Worker 各自管理独立子程序。
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
