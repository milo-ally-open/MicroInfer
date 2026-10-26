"""兼容入口；Mage-VL 输入编码实现位于 components.encoder。"""

from .components.encoder import (
    _check_context,
    _codec_processor_config,
    _generation_args,
    _load_image,
    _move_inputs_to_model,
    _normalize_messages,
    _prepare_inputs,
    _require_codec_runtime,
    _require_processor_codec_support,
    _resize_image,
    _video_inputs,
    MageVLEncoder,
)

__all__ = [
    "MageVLEncoder",
    "_check_context",
    "_codec_processor_config",
    "_generation_args",
    "_load_image",
    "_move_inputs_to_model",
    "_normalize_messages",
    "_prepare_inputs",
    "_require_codec_runtime",
    "_require_processor_codec_support",
    "_resize_image",
    "_video_inputs",
]
