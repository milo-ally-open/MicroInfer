"""Mage-VL 基础模型的兼容导出；正式实现位于 ``model.mage_vl``。"""

from .mage_vl import MageVLModel, build_position_ids

__all__ = ["MageVLModel", "build_position_ids"]
