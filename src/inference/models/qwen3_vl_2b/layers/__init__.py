"""Qwen3-VL 底层模块的延迟导出，避免单层 main 测试预先导入其他层。"""

from importlib import import_module

_EXPORTS = {
    "Qwen3VLTextEmbedding": "embedding",
    "Qwen3VLLMHead": "embedding",
    "Qwen3VLTextAttention": "text_attention",
    "Qwen3VLTextDecoderLayer": "text_decoder_layer",
    "Qwen3VLTextMLP": "text_mlp",
    "Qwen3VLTextRMSNorm": "text_norm",
    "Qwen3VLTextRotaryEmbedding": "text_rope",
    "Qwen3VLVisionAttention": "vision_attention",
    "Qwen3VLVisionBlock": "vision_block",
    "Qwen3VLVisionPatchEmbed": "vision_embedding",
    "Qwen3VLVisionPatchMerger": "vision_merger",
    "Qwen3VLVisionMLP": "vision_mlp",
    "Qwen3VLVisionLayerNorm": "vision_norm",
    "Qwen3VLVisionRotaryEmbedding": "vision_rope",
}
__all__ = list(_EXPORTS)


def __getattr__(name: str):
    """首次访问时才加载对应层，减小调试单个组件时的导入影响。"""
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f".{module_name}", __name__), name)
    globals()[name] = value
    return value
