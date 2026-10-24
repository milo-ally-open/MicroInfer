"""Mage-VL 模型执行入口；影像、影片与 Live 能力由本模型元件提供。"""

from engine.worker import ModelRuntime

from .components.adapter import MageVLModelAdapter
from .components.model_runner import MageVLModelRunner


def create_runtime() -> ModelRuntime:
    """建立供共用 Worker 使用的单一 Mage-VL Runtime。"""
    runner = MageVLModelRunner()
    return ModelRuntime(runner, MageVLModelAdapter(runner))
