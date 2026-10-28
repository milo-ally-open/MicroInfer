"""Mage-VL-4B 模型插件与推理服务入口。"""

from importlib import import_module
from pathlib import Path

from inference.engine.engine_client import EngineClient, create_engine_client_for_package


def create_engine_client(
    *,
    inference_config_path: str | Path | None = None,
    network_config_path: str | Path | None = None,
) -> EngineClient:
    """创建本模型 Engine client；未传配置路径时使用 model.yml 登记的默认配置。"""
    return create_engine_client_for_package(
        import_module(__name__),
        inference_config_path=inference_config_path,
        network_config_path=network_config_path,
    )


__all__ = ["create_engine_client"]
