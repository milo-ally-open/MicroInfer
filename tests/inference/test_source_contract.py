"""以標準函式庫檢查推論設定與 HTTP 路由來源，不載入 CUDA 或模型權重。"""

from __future__ import annotations

import ast
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


class InferenceSourceContractTests(unittest.TestCase):
    def test_project_alias_and_upstream_model_id_remain_distinct(self) -> None:
        registry = (ROOT / "src/inference/model_registry.yaml").read_text(encoding="utf-8")
        inference_config = (ROOT / "src/inference/mage_vl_4b/configs/inference.yml").read_text(encoding="utf-8")
        self.assertRegex(registry, r"(?m)^default_model:\s*mage-vl-4b\s*$")
        self.assertRegex(registry, r"(?m)^\s+mage-vl-4b:\s*$")
        self.assertRegex(inference_config, r"(?m)^\s+id:\s*microsoft/Mage-VL\s*$")
        self.assertIn("revision: d88b153285f1633a61b2f693c59c8576693af185", inference_config)

    def test_documented_inference_routes_exist_in_fastapi_source(self) -> None:
        main = (ROOT / "src/inference/mage_vl_4b/main.py").read_text(encoding="utf-8")
        service = (ROOT / "src/inference/mage_vl_4b/service.py").read_text(encoding="utf-8")
        for route in (
            '"/health"',
            '"/v1/models"',
            '"/v1/chat/completions"',
            '"/v1/videos/understand"',
            '"/v1/live/sessions"',
        ):
            with self.subTest(route=route):
                self.assertIn(route, main + service)
        self.assertIn('"/v1/live/sessions/{session_id}/events"', service)
        self.assertIn('"/v1/live/sessions/{session_id}"', service)

    def test_configured_request_limits_match_middleware_names(self) -> None:
        main = (ROOT / "src/inference/mage_vl_4b/main.py").read_text(encoding="utf-8")
        inference_config = (ROOT / "src/inference/mage_vl_4b/configs/inference.yml").read_text(encoding="utf-8")
        limits = {
            "max_chat_request_bytes": 15728640,
            "max_live_request_bytes": 65536,
            "max_video_request_bytes": 210763776,
            "max_video_bytes": 209715200,
        }
        for name, value in limits.items():
            with self.subTest(limit=name):
                self.assertRegex(inference_config, rf"(?m)^\s*{re.escape(name)}:\s*{value}\s*$")
        for status in ("411", "413"):
            with self.subTest(status=status):
                self.assertIn(status, main)
        self.assertIn("Content-Length", main)
        self.assertEqual(main.count("inference.MAX_LIVE_REQUEST_BYTES"), 2)

    def test_inference_tests_do_not_import_or_start_model_runtime(self) -> None:
        source = Path(__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        self.assertTrue(imported.isdisjoint({"torch", "transformers", "huggingface_hub"}))


if __name__ == "__main__":
    unittest.main()
