"""以本機 HTTP stub 驗證性能基線客戶端的請求格式。"""

from __future__ import annotations

import json
import sys
import threading
import tempfile
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from gateway_client import GatewayClient  # noqa: E402


class GatewayStub(BaseHTTPRequestHandler):
    requests: list[tuple[str, dict[str, str], bytes]] = []
    lock = threading.Lock()

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        with self.lock:
            self.requests.append((self.path, dict(self.headers.items()), body))
        if self.path.endswith("/chat/completions"):
            self._reply_sse()
        else:
            self._reply_json({"text": "A mock video summary."})

    def _reply_json(self, value: dict) -> None:
        payload = json.dumps(value).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _reply_sse(self) -> None:
        payload = (
            b'data: {"choices":[{"delta":{"content":"A mock answer."}}]}\n\n'
            b"data: [DONE]\n\n"
        )
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, _format: str, *_args: object) -> None:
        return


class GatewayClientTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        GatewayStub.requests = []
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), GatewayStub)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.client = GatewayClient(f"http://127.0.0.1:{cls.server.server_port}", "test-api-key")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)

    def test_streaming_image_chat(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "sample.jpg"
            image.write_bytes(b"synthetic image bytes")
            result = self.client.stream_chat_with_image(image, "Describe this image.")

        self.assertEqual(result["status"], 200)
        self.assertIsNotNone(result["first_token_seconds"])
        path, headers, body = GatewayStub.requests[-1]
        self.assertEqual(path, "/v1/chat/completions")
        self.assertEqual(headers["Authorization"], "Bearer test-api-key")
        request = json.loads(body)
        self.assertTrue(request["stream"])
        self.assertEqual(request["model"], "mage-vl-4b")
        self.assertTrue(request["messages"][0]["content"][1]["image_url"]["url"].startswith("data:image/jpeg;base64,"))

    def test_video_uses_content_length_multipart_upload(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            video = Path(directory) / "sample.mp4"
            video.write_bytes(b"synthetic video bytes" * 32)
            response = self.client.understand_video(video, "Summarize this clip.")

        self.assertEqual(response["text"], "A mock video summary.")
        path, headers, body = GatewayStub.requests[-1]
        self.assertEqual(path, "/v1/videos/understand")
        self.assertEqual(int(headers["Content-Length"]), len(body))
        self.assertIn("multipart/form-data; boundary=", headers["Content-Type"])
        self.assertIn(b'name="video"; filename="sample.mp4"', body)
        self.assertIn(b"synthetic video bytes", body)
        self.assertIn(b"Summarize this clip.", body)


if __name__ == "__main__":
    unittest.main()
