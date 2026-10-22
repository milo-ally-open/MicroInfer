"""驗證五層推理引擎中的輸入、輸出、排程與 Runner 批次契約。"""

from __future__ import annotations

import pickle
import sys
import unittest
from pathlib import Path

INFERENCE_ROOT = Path(__file__).resolve().parents[2] / "src" / "inference"
sys.path.insert(0, str(INFERENCE_ROOT))

from api.output_processor import IncrementalDetokenizer
from engine.contracts import (
    InferenceResult,
    InferenceStreamEvent,
    PreparedChatInput,
    WorkerResult,
)
from engine.engine import InferenceEngine
from engine.output_processor import EngineOutputProcessor
from engine.platform import create_platform


class FakeTokenizer:
    def __init__(self, tokens: dict[int, str]) -> None:
        self.tokens = tokens

    def decode(self, token_ids: list[int], **_: object) -> str:
        return "".join(self.tokens[token_id] for token_id in token_ids if token_id in self.tokens)


class EngineLayerTests(unittest.TestCase):
    def test_prepared_chat_input_is_serializable_across_process_boundary(self) -> None:
        request = PreparedChatInput("qwen3-0.6b", (1, 2, 3), {"max_new_tokens": 16})
        self.assertEqual(pickle.loads(pickle.dumps(request)), request)

    def test_incremental_detokenizer_preserves_words_and_cjk_text(self) -> None:
        detokenizer = IncrementalDetokenizer(
            FakeTokenizer({1: "Hel", 2: "lo", 3: " ", 4: "world", 5: "中", 6: "文"})
        )
        self.assertEqual(detokenizer.push(1), "Hel")
        self.assertEqual(detokenizer.push(2), "lo")
        self.assertEqual(detokenizer.push(3), "")
        self.assertEqual(detokenizer.push(4), " world")
        self.assertEqual(detokenizer.finish(), "")

        cjk = IncrementalDetokenizer(FakeTokenizer({5: "中", 6: "文"}))
        self.assertEqual(cjk.push(5), "中")
        self.assertEqual(cjk.push(6), "文")
        self.assertEqual(cjk.finish(), "")

    def test_incremental_detokenizer_buffers_incomplete_unicode(self) -> None:
        class UnicodeTokenizer:
            def decode(self, token_ids: list[int], **_: object) -> str:
                return "\ufffd" if len(token_ids) == 1 else "漢"

        detokenizer = IncrementalDetokenizer(UnicodeTokenizer())
        self.assertEqual(detokenizer.push(1), "")
        self.assertEqual(detokenizer.push(2), "漢")
        self.assertEqual(detokenizer.finish(), "")

    def test_engine_output_processor_keeps_token_and_text_paths_separate(self) -> None:
        processor = EngineOutputProcessor()
        self.assertEqual(processor.process(WorkerResult("token-request", token_id=42))[0].token_id, 42)
        self.assertEqual(processor.process(WorkerResult("text-request", text_delta="word"))[0].text, "word")
        self.assertEqual(processor.process(WorkerResult("prefill-only")), ())
        with self.assertRaisesRegex(ValueError, "both token_id and text_delta"):
            processor.process(WorkerResult("invalid", token_id=42, text_delta="word"))

    def test_non_stream_result_preserves_worker_token_ids_for_api_detokenization(self) -> None:
        result = InferenceEngine._collect_result(
            iter(
                (
                    InferenceStreamEvent("token", token_id=42),
                    InferenceStreamEvent("finish", token_ids=(42,), prompt_tokens=3, completion_tokens=1),
                )
            )
        )
        self.assertEqual(result.content, "")
        self.assertEqual(result.token_ids, (42,))
        self.assertEqual(result.prompt_tokens, 3)

    def test_unknown_platform_fails_without_silent_fallback(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported inference platform"):
            create_platform("unregistered")

if __name__ == "__main__":
    unittest.main()
