"""Mage-VL utils 探測工具的獨立匯入與離線契約。"""

from __future__ import annotations

import ast
import importlib.util
import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROBE_PATH = PROJECT_ROOT / "scripts" / "mage_vl_4b" / "test.py"
CONFIG_PATH = PROJECT_ROOT / "scripts" / "mage_vl_4b" / "test.yaml"


class UtilsProbeContractTests(unittest.TestCase):
    def test_probe_uses_model_generation_and_traces_cache(self) -> None:
        import torch

        spec = importlib.util.spec_from_file_location("mage_vl_probe", PROBE_PATH)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        probe = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(probe)

        class FakeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.visual_inputs_seen = []

            def forward(self, input_ids, past_key_values=None, attention_mask=None,
                        use_cache=False, pixel_values=None):
                if not use_cache:
                    raise AssertionError("probe must request cached model output")
                self.visual_inputs_seen.append(pixel_values is not None)
                previous_length = 0 if past_key_values is None else past_key_values[0][0].shape[-2]
                cache_length = previous_length + input_ids.shape[-1]
                cache_tensor = torch.zeros((1, 1, cache_length, 2))
                logits = torch.zeros((1, input_ids.shape[-1], 4))
                logits[..., 1] = 1
                return SimpleNamespace(past_key_values=((cache_tensor, cache_tensor),), logits=logits)

            def generate(self, input_ids, attention_mask, pixel_values, max_new_tokens,
                         do_sample, use_cache, return_dict_in_generate):
                if do_sample or not use_cache or not return_dict_in_generate:
                    raise AssertionError("probe must use greedy cached generation")
                sequence = input_ids
                cache = None
                for index in range(max_new_tokens):
                    output = self(
                        sequence if index == 0 else sequence[:, -1:],
                        past_key_values=cache,
                        attention_mask=torch.ones((1, input_ids.shape[-1] + index), dtype=torch.long),
                        pixel_values=pixel_values if index == 0 else None,
                        use_cache=use_cache,
                    )
                    cache = output.past_key_values
                    token = output.logits[:, -1].argmax(dim=-1, keepdim=True)
                    sequence = torch.cat((sequence, token), dim=-1)
                return SimpleNamespace(sequences=sequence)

        class FakeProcessor:
            @staticmethod
            def batch_decode(_token_ids, skip_special_tokens=True):
                return ["ok"]

        model = FakeModel()
        result = probe._generate_with_cache_trace(
            model,
            FakeProcessor(),
            {
                "input_ids": torch.tensor([[1, 2]]),
                "attention_mask": torch.ones((1, 2), dtype=torch.long),
                "pixel_values": torch.ones((1, 3, 2, 2)),
            },
            2,
            torch,
        )

        self.assertEqual(result["completion_tokens"], 2)
        self.assertEqual([step["phase"] for step in result["steps"]], ["prefill", "decode"])
        self.assertTrue(result["cached_generation_observed"])
        self.assertEqual(result["cache_after_prefill"]["sequence_length"], 2)
        self.assertEqual(result["steps"][1]["cache"]["sequence_length"], 3)
        self.assertEqual(model.visual_inputs_seen, [True, False])
        self.assertTrue(result["cache_grows_one_token_per_decode"])

    def test_probe_has_no_project_source_imports_or_path_injection(self) -> None:
        source = PROBE_PATH.read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported_roots = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_roots.update(alias.name.split(".", 1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_roots.add(node.module.split(".", 1)[0])
        project_modules = {"src", "launcher", "api", "engine", "media", "models"}

        self.assertFalse(imported_roots & project_modules)
        self.assertNotIn("sys.path", source)

    def test_probe_requires_explicit_paths_and_loads_model_yaml(self) -> None:
        source = PROBE_PATH.read_text(encoding="utf-8")
        config = CONFIG_PATH.read_text(encoding="utf-8")

        self.assertNotIn("Path(__file__)", source)
        self.assertNotIn("PROJECT_ROOT", source)
        self.assertNotIn("DEFAULT_CONFIG", source)
        self.assertIn('parser.add_argument("--config", required=True', source)
        self.assertIn('parser.add_argument("--model-path", required=True', source)
        self.assertIn('parser.add_argument("--question", required=True', source)
        self.assertIn("def _resolve_path", source)
        self.assertIn("目前工作目錄", source)
        self.assertNotIn("snapshot_download", source)
        for section in ("model:", "processor:", "runtime:", "api:", "generation:", "optimization:"):
            with self.subTest(section=section):
                self.assertIn(section, config)
        self.assertIn("id: microsoft/Mage-VL", config)
        self.assertIn("revision: d88b153285f1633a61b2f693c59c8576693af185", config)
        self.assertIn("local_files_only: true", config)
        self.assertIn("max_tokens_default: 8", config)

    def test_probe_forces_hugging_face_offline_loading(self) -> None:
        source = PROBE_PATH.read_text(encoding="utf-8")

        self.assertIn('os.environ["HF_HUB_OFFLINE"] = "1"', source)
        self.assertIn('os.environ["TRANSFORMERS_OFFLINE"] = "1"', source)
        self.assertIn('"local_files_only": True', source)
        self.assertIn("AutoProcessor.from_pretrained(snapshot, **processor_kwargs)", source)
        self.assertIn("AutoModelForCausalLM.from_pretrained(snapshot, **kwargs)", source)

    def test_help_does_not_require_model_dependencies(self) -> None:
        result = subprocess.run(
            [sys.executable, str(PROBE_PATH), "--help"],
            capture_output=True,
            check=False,
            text=True,
            timeout=15,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--image", result.stdout)
        self.assertIn("--video", result.stdout)
        self.assertIn("--output", result.stdout)


if __name__ == "__main__":
    unittest.main()
