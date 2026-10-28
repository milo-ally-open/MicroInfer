"""Mage-VL 模型执行入口；图像、视频与 Live 能力由本模型组件提供。"""

from inference.engine.worker import ModelRuntime

from .components.adapter import MageVLModelAdapter
from .components.model_runner import MageVLModelRunner


def create_runtime() -> ModelRuntime:
    """创建供共用 Worker 使用的单一 Mage-VL Runtime。"""
    runner = MageVLModelRunner()
    return ModelRuntime(runner, MageVLModelAdapter(runner))
