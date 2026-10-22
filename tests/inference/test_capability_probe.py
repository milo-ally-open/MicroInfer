"""固定能力探測工具的獨立匯入與離線契約。"""

from __future__ import annotations

import ast
import subprocess
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROBE_PATH = PROJECT_ROOT / "scripts" / "mage_vl_4b" / "test.py"
CONFIG_PATH = PROJECT_ROOT / "scripts" / "mage_vl_4b" / "test.yaml"


class CapabilityProbeContractTests(unittest.TestCase):
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
        self.assertIn("local_files_only=True", source)

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
