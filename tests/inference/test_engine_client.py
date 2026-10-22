"""驗證 EngineClient 程序邊界與既有 HTTP SSE 行為。"""

from __future__ import annotations

import sys
import time
import unittest
from pathlib import Path

INFERENCE_ROOT = Path(__file__).resolve().parents[2] / "src" / "inference"
if str(INFERENCE_ROOT) not in sys.path:
    sys.path.insert(0, str(INFERENCE_ROOT))

from engine.client import EngineClient, EngineClientError


class EngineClientProcessTests(unittest.TestCase):
    def make_client(self) -> EngineClient:
        return EngineClient(
            "engine_client_fakes:create_engine_core",
            startup_timeout_seconds=15,
            request_timeout_seconds=15,
            max_video_bytes=1024,
        )

    def test_api_engine_worker_processes_are_separate_and_stream_order_is_preserved(self) -> None:
        engine = self.make_client()
        engine.start()
        try:
            health = engine.health()
            process_ids = {health["api_pid"], health["engine_core_pid"], health["worker_pid"]}
            self.assertEqual(len(process_ids), 3)

            events = list(engine.stream_chat({"messages": [{"role": "user", "content": "hello"}]}))
            self.assertEqual([event.kind for event in events], ["start", "text", "text", "finish"])
            self.assertEqual([event.text for event in events if event.kind == "text"], ["h", "i"])
            self.assertEqual(events[-1].finish_reason, "stop")
        finally:
            engine.close()

    def test_remote_errors_are_returned_and_engine_stays_usable(self) -> None:
        engine = self.make_client()
        engine.start()
        try:
            with self.assertRaisesRegex(EngineClientError, "fake preparation failure"):
                engine.generate_chat({"fail": True})
            self.assertEqual(engine.generate_chat({}).content, "hi")
        finally:
            engine.close()

    def test_closing_stream_cancels_request_and_close_is_graceful(self) -> None:
        engine = self.make_client()
        engine.start()
        process = engine._process
        try:
            stream = engine.stream_chat({})
            self.assertEqual(next(stream).kind, "start")
            stream.close()
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                health = engine.health()
                if health["scheduler"]["running"] == 0 and health["kv_cache"]["used_blocks"] == 0:
                    break
                time.sleep(0.05)
            self.assertEqual(health["scheduler"]["running"], 0)
            self.assertEqual(health["kv_cache"]["used_blocks"], 0)
        finally:
            engine.close()
        self.assertIsNotNone(process)
        self.assertFalse(process.is_alive())


if __name__ == "__main__":
    unittest.main()
