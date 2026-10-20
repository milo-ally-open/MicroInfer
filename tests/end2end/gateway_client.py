"""端到端性能測試使用的 Gateway HTTP 客戶端。"""

from __future__ import annotations

import base64
import http.client
import json
import time
import uuid
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


MODEL_ALIAS = "mage-vl-4b"
MAX_RESPONSE_BYTES = 10 * 1024 * 1024


class GatewayAPIError(RuntimeError):
    """表示 Gateway 回傳非成功狀態。"""


class GatewayClient:
    """提供性能基線所需的 SSE 聊天和影片上傳呼叫。"""

    def __init__(self, base_url: str, api_key: str, timeout: float = 300.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.last_video_upload_seconds: float | None = None

    def stream_chat_with_image(self, image_path: Path, question: str) -> dict[str, float | int | None]:
        payload = self._image_chat_payload(image_path, question)
        request = Request(
            self.base_url + "/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Accept": "text/event-stream",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        started = time.perf_counter()
        first_token: float | None = None
        try:
            with urlopen(request, timeout=self.timeout) as response:
                for raw_line in response:
                    if len(raw_line) > MAX_RESPONSE_BYTES:
                        raise GatewayAPIError("Gateway SSE event 超過客戶端回應大小限制。")
                    if not raw_line.startswith(b"data:"):
                        continue
                    data = raw_line[5:].strip()
                    if data == b"[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    for choice in chunk.get("choices", []):
                        content = choice.get("delta", {}).get("content")
                        if isinstance(content, str) and content:
                            if first_token is None:
                                first_token = time.perf_counter() - started
                            break
                return {
                    "status": response.status,
                    "first_token_seconds": first_token,
                    "duration_seconds": time.perf_counter() - started,
                }
        except HTTPError as error:
            detail = error.read(MAX_RESPONSE_BYTES).decode("utf-8", errors="replace")
            raise GatewayAPIError(f"Gateway 回傳 HTTP {error.code}: {detail}") from error
        except URLError as error:
            raise GatewayAPIError(f"無法連線至 Gateway：{error.reason}") from error

    @staticmethod
    def _image_chat_payload(image_path: Path, question: str) -> dict:
        encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
        mime_type = "image/png" if image_path.suffix.lower() == ".png" else "image/jpeg"
        return {
            "model": MODEL_ALIAS,
            "stream": True,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                        {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{encoded}"}},
                    ],
                }
            ],
        }

    def understand_video(self, video_path: Path, question: str) -> dict:
        """以 multipart 串流上傳影片，避免將完整檔案載入記憶體。"""
        started = time.perf_counter()
        parsed = urlsplit(self.base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("Gateway URL 必須是完整的 http 或 https URL。")

        boundary = "----VideoStreamAPI-" + uuid.uuid4().hex
        filename = video_path.name.replace('"', "_").replace("\r", "_").replace("\n", "_")
        mime_type = "video/mp4" if video_path.suffix.lower() == ".mp4" else "application/octet-stream"
        file_header = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="video"; filename="{filename}"\r\n'
            f"Content-Type: {mime_type}\r\n\r\n"
        ).encode("utf-8")
        question_part = (
            f"\r\n--{boundary}\r\n"
            'Content-Disposition: form-data; name="question"\r\n\r\n'
            f"{question}"
        ).encode("utf-8")
        closing = f"\r\n--{boundary}--\r\n".encode("ascii")
        content_length = len(file_header) + video_path.stat().st_size + len(question_part) + len(closing)
        path = (parsed.path.rstrip("/") or "") + "/v1/videos/understand"
        if parsed.query:
            path += "?" + parsed.query
        connection_class = http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
        connection = connection_class(parsed.hostname, parsed.port, timeout=self.timeout)
        try:
            connection.putrequest("POST", path)
            connection.putheader("Authorization", f"Bearer {self.api_key}")
            connection.putheader("Accept", "application/json")
            connection.putheader("Content-Type", f"multipart/form-data; boundary={boundary}")
            connection.putheader("Content-Length", str(content_length))
            connection.endheaders()
            connection.send(file_header)
            with video_path.open("rb") as video:
                while chunk := video.read(1024 * 1024):
                    connection.send(chunk)
            connection.send(question_part)
            connection.send(closing)
            self.last_video_upload_seconds = time.perf_counter() - started
            response = connection.getresponse()
            raw = response.read(MAX_RESPONSE_BYTES + 1)
            if len(raw) > MAX_RESPONSE_BYTES:
                raise GatewayAPIError("Gateway 回應超過客戶端大小限制。")
            if not 200 <= response.status < 300:
                detail = raw.decode("utf-8", errors="replace")
                raise GatewayAPIError(f"Gateway 回傳 HTTP {response.status}: {detail}")
            return json.loads(raw)
        finally:
            connection.close()
