"""Mage-VL 基础模型的兼容导出；正式实现位于模型包的 ``modeling``。"""

from ..modeling import MageVLModel, build_position_ids

__all__ = ["MageVLModel", "build_position_ids"]
