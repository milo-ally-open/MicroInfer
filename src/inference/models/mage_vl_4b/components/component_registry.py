"""列出固定 Mage-VL 架构组件，并应用模型包内登记的替换实现。"""

from __future__ import annotations

from dataclasses import dataclass
from fnmatch import fnmatchcase
from typing import Any, Callable


@dataclass(frozen=True)
class ComponentSlot:
    """固定 revision 中可观察或替换的模型模块群组。"""

    name: str
    path_pattern: str


@dataclass(frozen=True)
class ComponentDescriptor:
    """加载模型后找到的单一模块路径与实际类别。"""

    slot: str
    path: str
    module_type: str


@dataclass(frozen=True)
class _Replacement:
    factory: Callable[[Any, str], Any]
    transfer_weights: Callable[[Any, Any], None] | None


MAGE_VL_COMPONENT_SLOTS = (
    ComponentSlot("vision_patch_embedding", "model.visual.embeddings.patch_embedding"),
    ComponentSlot("vision_attention", "model.visual.encoder.layers.*.self_attn"),
    ComponentSlot("vision_mlp", "model.visual.encoder.layers.*.mlp"),
    ComponentSlot("vision_patch_merger", "model.visual.merger"),
    ComponentSlot("language_attention", "model.language_model.layers.*.self_attn"),
    ComponentSlot("language_mlp", "model.language_model.layers.*.mlp"),
    ComponentSlot("language_embeddings", "model.language_model.embed_tokens"),
    ComponentSlot("output_head", "lm_head"),
)


class MageVLComponentRegistry:
    """以固定模块路径登记本地实现，不从 YAML 动态汇入代码。"""

    def __init__(self) -> None:
        self._slots = {slot.name: slot for slot in MAGE_VL_COMPONENT_SLOTS}
        self._replacements: dict[tuple[str, str], _Replacement] = {}

    def register(
        self,
        slot: str,
        implementation: str,
        factory: Callable[[Any, str], Any],
        *,
        transfer_weights: Callable[[Any, Any], None] | None = None,
    ) -> None:
        """登记模型包中的替换 factory；factory 必须返回完整初始化的模块。"""
        if slot not in self._slots:
            raise ValueError(f"Unknown Mage-VL component slot: {slot}")
        if not implementation or implementation.strip() != implementation:
            raise ValueError("Component implementation name must be non-empty and trimmed.")
        if not callable(factory):
            raise TypeError("Component replacement factory must be callable.")
        key = (slot, implementation)
        if key in self._replacements:
            raise ValueError(f"Mage-VL component replacement is already registered: {slot}={implementation}")
        self._replacements[key] = _Replacement(factory, transfer_weights)

    def validate_configuration(self, configured: Any) -> tuple[tuple[str, str, _Replacement], ...]:
        """在加载权重前确认 YAML 只选取已登记的模型包实现。"""
        if configured is None:
            configured = {}
        if not isinstance(configured, dict):
            raise RuntimeError("model.components.replacements must be a mapping.")

        validated: list[tuple[str, str, _Replacement]] = []
        for slot_name, implementation in configured.items():
            if slot_name not in self._slots:
                raise RuntimeError(f"Unknown model.components.replacements target: {slot_name}")
            if not isinstance(implementation, str) or not implementation.strip() or implementation.strip() != implementation:
                raise RuntimeError(f"Replacement for {slot_name} must name a trimmed local implementation.")
            replacement = self._replacements.get((slot_name, implementation))
            if replacement is None:
                raise RuntimeError(
                    f"No local Mage-VL replacement is registered for {slot_name}={implementation}. "
                    "Register the implementation in the model package before selecting it in YAML."
                )
            validated.append((slot_name, implementation, replacement))
        return tuple(validated)

    def inventory(self, model: Any) -> dict[str, tuple[ComponentDescriptor, ...]]:
        """依固定 revision 的路径样式检查已加载模型的主要组件。"""
        named_modules = dict(model.named_modules())
        inventory: dict[str, tuple[ComponentDescriptor, ...]] = {}
        missing: list[str] = []
        for slot in MAGE_VL_COMPONENT_SLOTS:
            matches = tuple(
                ComponentDescriptor(slot.name, path, type(module).__name__)
                for path, module in named_modules.items()
                if path and fnmatchcase(path, slot.path_pattern)
            )
            inventory[slot.name] = matches
            if not matches:
                missing.append(f"{slot.name} ({slot.path_pattern})")
        if missing:
            raise RuntimeError(
                "Loaded Mage-VL module tree does not match the pinned architecture; missing: "
                + ", ".join(missing)
            )
        return inventory

    def apply(self, model: Any, configured: Any) -> tuple[Any, tuple[str, ...]]:
        """应用 YAML 选取的替换；默认不变更 Transformers 加载的模块。"""
        replacements = self.validate_configuration(configured)
        if not replacements:
            return model, ()

        inventory = self.inventory(model)
        pending: list[tuple[str, Any, Any]] = []
        named_modules = dict(model.named_modules())
        for slot_name, implementation, replacement in replacements:
            for descriptor in inventory[slot_name]:
                original = named_modules[descriptor.path]
                candidate = replacement.factory(original, descriptor.path)
                self._validate_module(candidate, slot_name, implementation)
                if replacement.transfer_weights is None:
                    self._copy_compatible_weights(original, candidate, descriptor.path)
                else:
                    replacement.transfer_weights(original, candidate)
                if hasattr(candidate, "train"):
                    candidate.train(bool(getattr(original, "training", False)))
                pending.append((descriptor.path, original, candidate))

        # 所有模块均创建并完成权重转移后才修改树，避免前段成功、后段失败留下半套模型。
        applied: list[tuple[str, Any]] = []
        try:
            for path, original, candidate in pending:
                parent_path, child_name = path.rsplit(".", 1) if "." in path else ("", path)
                parent = dict(model.named_modules()).get(parent_path) if parent_path else model
                if parent is None or child_name not in getattr(parent, "_modules", {}):
                    raise RuntimeError(f"Could not replace Mage-VL module at {path}.")
                applied.append((path, original))
                setattr(parent, child_name, candidate)
        except BaseException:
            for path, original in reversed(applied):
                parent_path, child_name = path.rsplit(".", 1) if "." in path else ("", path)
                parent = dict(model.named_modules()).get(parent_path) if parent_path else model
                if parent is not None:
                    setattr(parent, child_name, original)
            raise
        return model, tuple(path for path, _, _ in pending)

    @staticmethod
    def _validate_module(module: Any, slot: str, implementation: str) -> None:
        try:
            from torch.nn import Module
        except ImportError as exc:
            raise RuntimeError("PyTorch is required to apply model component replacements.") from exc
        if not isinstance(module, Module):
            raise TypeError(f"Replacement {slot}={implementation} did not return torch.nn.Module.")

    @staticmethod
    def _copy_compatible_weights(original: Any, replacement: Any, path: str) -> None:
        """严格复制相同 state-dict 结构；转换型实现需提供专用 weight loader。"""
        parameters = iter(original.parameters()) if hasattr(original, "parameters") else iter(())
        parameter = next(parameters, None)
        if parameter is not None and hasattr(replacement, "to"):
            kwargs: dict[str, Any] = {"device": parameter.device}
            if parameter.is_floating_point() or parameter.is_complex():
                kwargs["dtype"] = parameter.dtype
            replacement.to(**kwargs)
        try:
            result = replacement.load_state_dict(original.state_dict(), strict=True)
        except Exception as exc:
            raise RuntimeError(
                f"Replacement at {path} is not weight-compatible; provide an explicit weight transfer function."
            ) from exc
        if getattr(result, "missing_keys", ()) or getattr(result, "unexpected_keys", ()):
            raise RuntimeError(f"Replacement at {path} did not accept the complete source state dict.")


MAGE_VL_COMPONENT_REGISTRY = MageVLComponentRegistry()
