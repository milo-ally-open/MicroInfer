"""確保已登錄模型提供 Engine 所需的最小 Runner／Adapter 合約。"""

from __future__ import annotations

import unittest
from pathlib import Path

import yaml

from model_contract import assert_model_contract


ROOT = Path(__file__).resolve().parents[2]
INFERENCE = ROOT / "src/inference"


class ModelIntegrationContractTests(unittest.TestCase):
    def test_registered_model_factories_implement_declared_capabilities(self) -> None:
        registry = yaml.safe_load((INFERENCE / "model_registry.yaml").read_text(encoding="utf-8"))
        for alias, model in registry["models"].items():
            with self.subTest(model=alias):
                api = model["api"]
                capabilities = api["capabilities"]
                components = model.get("components", {})
                assert_model_contract(
                    self,
                    INFERENCE,
                    profile=model["integration_profile"],
                    runner_factory=components.get("runner"),
                    adapter_factory=components.get("adapter"),
                    input_processor_factory=api.get("input_processor"),
                    output_mode=capabilities["output_mode"],
                    video_upload=capabilities["video_upload"],
                    live_video=capabilities["live_video"],
                )
                if capabilities["output_mode"] == "token_ids":
                    self.assertTrue(api.get("input_processor"))


if __name__ == "__main__":
    unittest.main()
