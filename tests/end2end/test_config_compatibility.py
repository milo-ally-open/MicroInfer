"""確保 Gateway 本文上限不高於目前推論服務的相應設定。"""

from __future__ import annotations

import re
import unittest
from pathlib import Path


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


if __name__ == "__main__":
    unittest.main()
