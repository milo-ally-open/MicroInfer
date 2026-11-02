"""Qwen3-VL-2B Inference Engine 运行时与客户端入口。"""

from importlib import import_module
from pathlib import Path

from inference.engine.engine_client import EngineClient, create_engine_client_for_package


def create_engine_client(
    *,
    inference_config_path: str | Path | None = None,
    network_config_path: str | Path | None = None,
) -> EngineClient:
    """创建绑定到 Qwen3-VL-2B 配置的 Engine Client。"""
    return create_engine_client_for_package(
        import_module(__name__),
        inference_config_path=inference_config_path,
        network_config_path=network_config_path,
    )


__all__ = ["create_engine_client"]
