"""Qwen3-VL Runtime 组装与 Engine Client 入口。"""

from importlib import import_module
from pathlib import Path

from inference.engine.engine_client import EngineClient, create_engine_client_for_package
from inference.engine.worker import ModelRuntime


def create_runtime() -> ModelRuntime:
    from .components.adapter import Qwen3VLModelAdapter
    from .components.model_runner import Qwen3VLModelRunner

    runner = Qwen3VLModelRunner()
    return ModelRuntime(runner, Qwen3VLModelAdapter(runner))


def create_engine_client(
    *,
    inference_config_path: str | Path | None = None,
    network_config_path: str | Path | None = None,
) -> EngineClient:
    """创建 Qwen3-VL Engine Client，并使用 profiles/ 中登记的默认配置。"""
    return create_engine_client_for_package(
        import_module(__package__),
        inference_config_path=inference_config_path,
        network_config_path=network_config_path,
    )
