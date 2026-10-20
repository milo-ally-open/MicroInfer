"""驗證 OpenAI SDK 範例在模組頂層送出的圖片聊天請求。"""

from __future__ import annotations

import base64
import contextlib
import io
import os
import runpy
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


class ExampleTests(unittest.TestCase):
    def test_sends_image_chat_through_openai_sdk(self) -> None:
        sdk_client = Mock()
        sdk_client.chat.completions.create.return_value.choices = [
            types.SimpleNamespace(message=types.SimpleNamespace(content="A mock answer."))
        ]
        openai_module = types.ModuleType("openai")
        openai_module.OpenAI = Mock(return_value=sdk_client)
        source = Path(__file__).resolve().parents[2] / "example.py"
        image_bytes = b"synthetic image bytes"

        with tempfile.TemporaryDirectory() as directory:
            example_path = Path(directory) / "example.py"
            example_path.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
            example_path.with_name("test.jpg").write_bytes(image_bytes)
            output = io.StringIO()
            with (
                patch.dict("sys.modules", {"openai": openai_module}),
                patch.dict(os.environ, {"VIDEO_API_KEY": "test-api-key", "GATEWAY_URL": "http://gateway.test"}),
                contextlib.redirect_stdout(output),
            ):
                runpy.run_path(str(example_path))

        openai_module.OpenAI.assert_called_once_with(
            api_key="test-api-key",
            base_url="http://gateway.test/v1",
        )
        request = sdk_client.chat.completions.create.call_args.kwargs
        self.assertEqual(request["model"], "mage-vl-4b")
        content = request["messages"][0]["content"]
        self.assertEqual(content[0], {"type": "text", "text": "請簡要描述圖片內容。"})
        self.assertEqual(
            content[1]["image_url"]["url"],
            "data:image/jpeg;base64," + base64.b64encode(image_bytes).decode("ascii"),
        )
        self.assertEqual(output.getvalue().strip(), "A mock answer.")


if __name__ == "__main__":
    unittest.main()
