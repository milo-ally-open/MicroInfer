#!/usr/bin/env python3
"""推論服務的中央啟動器與模型設定選擇器。"""

from __future__ import annotations

import argparse
import importlib.util
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
    """列出模型狀態及登錄的推論、網路、媒體設定路徑。"""
    print("MODEL\tBACKEND\tSTATUS\tINFERENCE CONFIG\tNETWORK CONFIG\tMEDIA CONFIG\tSERVICE")
    for alias, model in sorted(registry["models"].items()):
        if not isinstance(model, dict):
            print(f"{alias}\t\tinvalid-registry-entry")
            continue
        display_name = model.get("display_name", alias)
        backend = registry.get("default_backend", "")
        status = "implemented"
        paths = [
            model.get("inference_config", ""),
            model.get("network_config", ""),
            model.get("media_config", ""),
            model.get("service_entrypoint", ""),
        ]
        try:
            for value, description in zip(
                paths,
                ("Inference configuration", "Network configuration", "Media configuration", "Service application"),
            ):
                _resolve_path(str(value), description)
        except (FileNotFoundError, OSError):
            status = "missing-file"
        print(
            f"{alias} ({display_name})\t{backend}\t{status}\t"
            f"{paths[0]}\t{paths[1]}\t{paths[2]}\t{paths[3]}"
        )


def _resolve_selection(
    args: argparse.Namespace, registry: dict[str, Any]
) -> tuple[str, str, Path, Path, Path, Path]:
    """依命令列與登錄表選出模型及其四個服務資源檔案。"""
    model_alias = args.model or str(registry.get("default_model", ""))
    if not model_alias:
        raise ValueError("Pass --model or set default_model in model_registry.yaml.")
    models = registry["models"]
    if model_alias not in models:
        raise ValueError(f"Unknown model config '{model_alias}'. Available: {', '.join(sorted(models))}")

    model_spec = models[model_alias]
    if not isinstance(model_spec, dict):
        raise ValueError(f"Model registry entry '{model_alias}' must be a mapping.")
    backend = str(registry.get("default_backend", ""))
    if not backend:
        raise ValueError("Set default_backend in model_registry.yaml.")

    inference_config_path = _resolve_path(
        str(args.inference_config or model_spec.get("inference_config", "")), "Inference configuration"
    )
    network_config_path = _resolve_path(
        str(args.network_config or model_spec.get("network_config", "")), "Network configuration"
    )
    media_config_path = _resolve_path(
        str(args.media_config or model_spec.get("media_config", "")), "Media configuration"
    )
    app_path = _resolve_path(str(model_spec.get("service_entrypoint", "")), "Service entrypoint")
    return model_alias, backend, inference_config_path, network_config_path, media_config_path, app_path


def _set_runtime_environment(config_path: Path, network_config_path: Path, config: dict[str, Any]) -> None:
    """套用模型設定中的環境變數，讓服務啟動時讀取正確執行環境。"""
    runtime = config.get("runtime", {})
    os.environ["INFERENCE_CONFIG"] = str(config_path)
    os.environ["INFERENCE_NETWORK_CONFIG"] = str(network_config_path)

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


def _load_service_app(app_path: Path, model_alias: str, backend: str) -> Any:
    """透過臨時套件名稱載入模型套件及其 API app。"""
    backend_dir = app_path.parent
    package_name = "inference_runtime_" + re.sub(r"[^A-Za-z0-9_]", "_", f"{model_alias}_{backend}")
    init_path = backend_dir / "__init__.py"
    if not init_path.is_file():
        raise FileNotFoundError(f"Backend package initializer not found: {init_path}")
    package_spec = importlib.util.spec_from_file_location(
        package_name,
        init_path,
        submodule_search_locations=[str(backend_dir)],
    )
    if package_spec is None or package_spec.loader is None:
        raise RuntimeError(f"Could not create backend package loader for {backend}.")
    package = importlib.util.module_from_spec(package_spec)
    sys.modules[package_name] = package
    package_spec.loader.exec_module(package)

    app_spec = importlib.util.spec_from_file_location(f"{package_name}.{app_path.stem}", app_path)
    if app_spec is None or app_spec.loader is None:
        raise RuntimeError(f"Could not load service application: {app_path}")
    app_module = importlib.util.module_from_spec(app_spec)
    sys.modules[app_spec.name] = app_module
    app_spec.loader.exec_module(app_module)
    if not hasattr(app_module, "app"):
        raise AttributeError(f"Service entrypoint does not export 'app': {app_path}")
    return app_module.app


def _add_runtime_overrides(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    parser.add_argument("--cuda-visible-devices")


def main() -> None:
    """解析啟動參數、載入指定模型設定，並啟動常駐 API 服務。"""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="列出已登錄的模型、設定檔及服務入口狀態。")
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

    model_alias, backend, config_path, network_config_path, media_config_path, app_path = _resolve_selection(args, registry)
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
    _set_runtime_environment(config_path, network_config_path, config)

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
            "runtime.uvicorn_workers must be 1 while each API Server owns one dedicated Model Worker process."
        )

    print(
        f"Starting model '{model_alias}' with backend '{backend}' using inference config {config_path}, "
        f"network config {network_config_path}, and media config {media_config_path}"
    )
    import uvicorn

    # 服務程序常駐提供 HTTP API；工作者數量由模型設定控制。
    uvicorn.run(
        _load_service_app(app_path, model_alias, backend),
        host=host,
        port=port,
        workers=api_workers,
        log_level=str(runtime.get("uvicorn_log_level", "info")),
        access_log=bool(runtime.get("uvicorn_access_log", True)),
        timeout_keep_alive=int(runtime.get("uvicorn_timeout_keep_alive_seconds", 5)),
    )


if __name__ == "__main__":
    main()
