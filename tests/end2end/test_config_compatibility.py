"""確保 Gateway 本文上限不高於目前推論服務的相應設定。"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def yaml_integer(path: Path, key: str) -> int:
    source = path.read_text(encoding="utf-8")
    match = re.search(rf"(?m)^\s*{re.escape(key)}:\s*(\d+)\s*$", source)
    if not match:
        raise AssertionError(f"missing integer YAML value for {key} in {path}")
    return int(match.group(1))


class GatewayInferenceConfigCompatibilityTests(unittest.TestCase):
    def test_gateway_body_limits_do_not_exceed_inference_limits(self) -> None:
        gateway = ROOT / "src/gateway/configs/gateway.yml"
        inference = ROOT / "src/inference/models/mage_vl_4b/configs/inference.yml"
        pairs = (
            ("chat_request_bytes", "max_chat_request_bytes"),
            ("video_request_bytes", "max_video_request_bytes"),
            ("live_request_bytes", "max_live_request_bytes"),
        )
        for gateway_key, inference_key in pairs:
            with self.subTest(request_type=gateway_key):
                self.assertLessEqual(yaml_integer(gateway, gateway_key), yaml_integer(inference, inference_key))

    def test_gateway_routes_match_inference_aliases_upstreams_and_capabilities(self) -> None:
        gateway = yaml.safe_load((ROOT / "src/gateway/configs/gateway.yml").read_text(encoding="utf-8"))
        registry = yaml.safe_load((ROOT / "src/inference/model_registry.yaml").read_text(encoding="utf-8"))
        self.assertEqual(set(gateway["models"]), set(registry["models"]))
        self.assertIn(gateway["defaults"]["video_model"], gateway["models"])
        self.assertIn(gateway["defaults"]["live_model"], gateway["models"])

        for alias, inference_model in registry["models"].items():
            with self.subTest(model=alias):
                route = gateway["models"][alias]
                inference_api = inference_model["api"]
                inference_capabilities = inference_api["capabilities"]
                gateway_capabilities = route["capabilities"]
                self.assertEqual(route["upstream_model"], inference_api["model_id"])
                self.assertEqual(gateway_capabilities["text_chat"], "text" in inference_capabilities["chat_inputs"])
                self.assertEqual(gateway_capabilities["image_chat"], "image" in inference_capabilities["chat_inputs"])
                self.assertEqual(gateway_capabilities["streaming"], inference_capabilities["streaming"])
                self.assertEqual(gateway_capabilities["video_upload"], inference_capabilities["video_upload"])
                self.assertEqual(gateway_capabilities["live_video"], inference_capabilities["live_video"])

                network_path = ROOT / inference_model["network_config"]
                network = yaml.safe_load(network_path.read_text(encoding="utf-8"))
                api = network["api"]
                self.assertEqual(route["base_url"], f"http://{api['host']}:{api['port']}")


if __name__ == "__main__":
    unittest.main()
