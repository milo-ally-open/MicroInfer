"""驗證 Mage-VL 元件目錄、模型路徑和本地替換契約。"""

from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

INFERENCE_ROOT = Path(__file__).resolve().parents[2] / "src" / "inference"
sys.path.insert(0, str(INFERENCE_ROOT))

from models.mage_vl_4b.components.component_registry import MageVLComponentRegistry


class FakeModule:
    def __init__(self, weight: int | None = None) -> None:
        object.__setattr__(self, "_modules", {})
        object.__setattr__(self, "training", False)
        if weight is not None:
            object.__setattr__(self, "weight", weight)

    def __setattr__(self, name: str, value: object) -> None:
        if name in self._modules:
            self._modules[name] = value
        object.__setattr__(self, name, value)

    def add_module(self, name: str, module: FakeModule) -> None:
        self._modules[name] = module
        object.__setattr__(self, name, module)

    def named_modules(self, prefix: str = ""):
        yield prefix, self
        for name, module in self._modules.items():
            path = f"{prefix}.{name}" if prefix else name
            yield from module.named_modules(path)

    def parameters(self, recurse: bool = True):
        return iter(())

    def state_dict(self) -> dict[str, int]:
        return {"weight": self.weight} if hasattr(self, "weight") else {}

    def load_state_dict(self, state: dict[str, int], strict: bool = True):
        if strict and set(state) != set(self.state_dict()):
            raise ValueError("State dictionaries differ.")
        for name, value in state.items():
            object.__setattr__(self, name, value)
        return types.SimpleNamespace(missing_keys=[], unexpected_keys=[])

    def train(self, mode: bool = True):
        object.__setattr__(self, "training", mode)
        return self

    def to(self, **_: object):
        return self


def make_model() -> FakeModule:
    root = FakeModule()
    base = FakeModule()
    visual = FakeModule()
    embeddings = FakeModule()
    embeddings.add_module("patch_embedding", FakeModule(weight=1))
    visual.add_module("embeddings", embeddings)
    encoder = FakeModule()
    layers = FakeModule()
    for index in range(2):
        layer = FakeModule()
        layer.add_module("self_attn", FakeModule(weight=index + 10))
        layer.add_module("mlp", FakeModule(weight=index + 20))
        layers.add_module(str(index), layer)
    encoder.add_module("layers", layers)
    visual.add_module("encoder", encoder)
    visual.add_module("merger", FakeModule(weight=2))
    base.add_module("visual", visual)

    language_model = FakeModule()
    text_layers = FakeModule()
    for index in range(3):
        layer = FakeModule()
        layer.add_module("self_attn", FakeModule(weight=index + 30))
        layer.add_module("mlp", FakeModule(weight=index + 40))
        text_layers.add_module(str(index), layer)
    language_model.add_module("layers", text_layers)
    language_model.add_module("embed_tokens", FakeModule(weight=3))
    base.add_module("language_model", language_model)
    root.add_module("model", base)
    root.add_module("lm_head", FakeModule(weight=4))
    return root


class MageVLComponentRegistryTests(unittest.TestCase):
    def test_inventory_finds_vision_and_language_module_groups(self) -> None:
        inventory = MageVLComponentRegistry().inventory(make_model())

        self.assertEqual(len(inventory["vision_attention"]), 2)
        self.assertEqual(len(inventory["vision_mlp"]), 2)
        self.assertEqual(len(inventory["language_attention"]), 3)
        self.assertEqual(len(inventory["language_mlp"]), 3)
        self.assertEqual(inventory["vision_patch_merger"][0].path, "model.visual.merger")

    def test_empty_replacement_config_keeps_transformers_model_unchanged(self) -> None:
        model = make_model()
        returned, replaced = MageVLComponentRegistry().apply(model, {})

        self.assertIs(returned, model)
        self.assertEqual(replaced, ())

    def test_unknown_slot_and_unregistered_implementation_fail_closed(self) -> None:
        registry = MageVLComponentRegistry()
        with self.assertRaisesRegex(RuntimeError, "Unknown model.components.replacements target"):
            registry.validate_configuration({"text_attention": "optimized"})
        with self.assertRaisesRegex(RuntimeError, "No local Mage-VL replacement is registered"):
            registry.validate_configuration({"language_attention": "optimized"})
        with self.assertRaisesRegex(RuntimeError, "trimmed local implementation"):
            registry.validate_configuration({"language_attention": " optimized "})
        with self.assertRaisesRegex(RuntimeError, "Unknown model.components.replacements target"):
            registry.apply(make_model(), {"text_attention": "optimized"})
        with self.assertRaisesRegex(RuntimeError, "No local Mage-VL replacement is registered"):
            registry.apply(make_model(), {"language_attention": "optimized"})

    def test_registered_replacement_copies_weights_and_preserves_eval_state(self) -> None:
        registry = MageVLComponentRegistry()
        model = make_model()
        original = dict(model.named_modules())["model.visual.encoder.layers.0.self_attn"]
        registry.register("vision_attention", "test_attention", lambda _module, _path: FakeModule(weight=-1))

        fake_torch = types.ModuleType("torch")
        fake_nn = types.ModuleType("torch.nn")
        fake_nn.Module = FakeModule
        fake_torch.nn = fake_nn
        with patch.dict(sys.modules, {"torch": fake_torch, "torch.nn": fake_nn}):
            returned, replaced = registry.apply(model, {"vision_attention": "test_attention"})

        replacement = dict(returned.named_modules())["model.visual.encoder.layers.0.self_attn"]
        self.assertIs(returned, model)
        self.assertEqual(len(replaced), 2)
        self.assertIsNot(replacement, original)
        self.assertEqual(replacement.weight, 10)
        self.assertFalse(replacement.training)

    def test_failed_model_tree_update_rolls_back_prior_replacements(self) -> None:
        registry = MageVLComponentRegistry()
        model = make_model()
        original_attention = dict(model.named_modules())["model.visual.encoder.layers.0.self_attn"]
        original_head = model.lm_head
        registry.register("vision_attention", "test_attention", lambda _module, _path: FakeModule(weight=-1))
        registry.register("output_head", "test_head", lambda _module, _path: FakeModule(weight=-2))

        fake_torch = types.ModuleType("torch")
        fake_nn = types.ModuleType("torch.nn")
        fake_nn.Module = FakeModule
        fake_torch.nn = fake_nn
        original_setattr = FakeModule.__setattr__

        def fail_head_assignment(instance: FakeModule, name: str, value: object) -> None:
            if instance is model and name == "lm_head" and value is not original_head:
                raise RuntimeError("Injected assignment failure.")
            original_setattr(instance, name, value)

        with patch.dict(sys.modules, {"torch": fake_torch, "torch.nn": fake_nn}), patch.object(
            FakeModule, "__setattr__", fail_head_assignment
        ), self.assertRaisesRegex(RuntimeError, "Injected assignment failure"):
            registry.apply(model, {"vision_attention": "test_attention", "output_head": "test_head"})

        current_modules = dict(model.named_modules())
        self.assertIs(current_modules["model.visual.encoder.layers.0.self_attn"], original_attention)
        self.assertIs(model.lm_head, original_head)


if __name__ == "__main__":
    unittest.main()
