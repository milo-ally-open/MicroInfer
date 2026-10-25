"""Qwen3-0.6B 模型插件与推论服务入口。"""

from importlib import import_module

from inference.engine.engine_client import create_engine_client_for_package


def create_engine_client():
    """建立由本模型 manifest/YAML 配置的共用 Engine Core client。"""
    return create_engine_client_for_package(import_module(__name__))


__all__ = ["create_engine_client"]
