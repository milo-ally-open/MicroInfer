"""檢查 HTTP 序列化維持現有公開 response 欄位。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

INFERENCE_ROOT = Path(__file__).resolve().parents[2] / "src" / "inference"
sys.path.insert(0, str(INFERENCE_ROOT))

from api.serializers import chat_completion, chat_completion_chunk
from engine.engine import InferenceResult


class APISerializerTests(unittest.TestCase):
    def test_chat_response_keeps_openai_compatible_shape(self) -> None:
        response = chat_completion(InferenceResult("hello", 5, 2, "stop"), "mage-vl-4b")
        self.assertRegex(response["id"], r"^chatcmpl-[0-9a-f]+$")
        self.assertEqual(response["object"], "chat.completion")
        self.assertEqual(response["model"], "mage-vl-4b")
        self.assertEqual(response["choices"][0]["message"], {"role": "assistant", "content": "hello"})
        self.assertEqual(response["choices"][0]["finish_reason"], "stop")
        self.assertEqual(response["usage"], {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7})

    def test_video_response_keeps_video_metadata_and_measurements(self) -> None:
        result = InferenceResult("a person", 8, 3, "length", {
            "video": {"backend": "frames", "upload_bytes": 25},
            "timing_seconds": {"server_total": 1.25},
            "gpu_memory_bytes": {"peak_allocated": 100, "peak_reserved": 200},
        })
        response = chat_completion(result, "mage-vl-4b", video=True)
        self.assertRegex(response["id"], r"^videocmpl-[0-9a-f]+$")
        self.assertEqual(response["object"], "magevl4b.video.understanding")
        self.assertEqual(response["video"]["upload_bytes"], 25)
        self.assertEqual(response["timing_seconds"]["server_total"], 1.25)
        self.assertEqual(response["gpu_memory_bytes"]["peak_reserved"], 200)

    def test_stream_chunks_keep_role_content_finish_and_usage_fields(self) -> None:
        role = chat_completion_chunk("chatcmpl-id", 10, "mage-vl-4b", delta={"role": "assistant"})
        content = chat_completion_chunk("chatcmpl-id", 10, "mage-vl-4b", delta={"content": "hello"})
        finish = chat_completion_chunk(
            "chatcmpl-id", 10, "mage-vl-4b", delta={}, finish_reason="stop",
            usage={"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
        )
        self.assertEqual(role["choices"][0], {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None})
        self.assertEqual(content["choices"][0]["delta"], {"content": "hello"})
        self.assertEqual(finish["choices"][0]["finish_reason"], "stop")
        self.assertEqual(finish["usage"]["total_tokens"], 4)


if __name__ == "__main__":
    unittest.main()
