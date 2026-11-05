"""FastAPI 推理服务共用的 API Key、租户限流与 Live 句柄认证。"""

from __future__ import annotations

import asyncio
import base64
import binascii
import hashlib
import hmac
import json
import math
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from starlette.responses import JSONResponse


@dataclass(frozen=True)
class APIKeyIdentity:
    """API Key 验证后得到的服务端身份，不保存明文密钥。"""

    key_id: str
    tenant_id: str
    digest: bytes


class APIKeySet:
    """从受保护的 JSON 文件加载 Bearer 密钥并映射至 tenant。"""

    def __init__(self, entries: list[dict[str, Any]]) -> None:
        if not isinstance(entries, list) or not entries:
            raise ValueError("API key file must contain a non-empty JSON array.")
        self._by_digest: dict[bytes, APIKeyIdentity] = {}
        ids: set[str] = set()
        for entry in entries:
            if not isinstance(entry, dict):
                raise ValueError("Each API key entry must be a JSON object.")
            key_id = entry.get("id")
            tenant_id = entry.get("tenant_id")
            secret = entry.get("key")
            if (
                not isinstance(key_id, str)
                or not key_id.strip()
                or key_id != key_id.strip()
                or not isinstance(tenant_id, str)
                or not tenant_id.strip()
                or tenant_id != tenant_id.strip()
                or not isinstance(secret, str)
                or len(secret) < 16
                or any(char.isspace() for char in secret)
            ):
                raise ValueError("Each API key entry requires id, tenant_id, and a key of at least 16 characters.")
            digest = hashlib.sha256(secret.encode("utf-8")).digest()
            if key_id in ids or digest in self._by_digest:
                raise ValueError("API key IDs and secrets must be unique.")
            ids.add(key_id)
            self._by_digest[digest] = APIKeyIdentity(key_id, tenant_id, digest)

    @classmethod
    def from_file(cls, path: str | Path) -> APIKeySet:
        """读取 JSON 密钥文件；错误不包含文件内容或密钥值。"""
        try:
            entries = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Could not read the configured API key file: {path}") from exc
        return cls(entries)

    def authenticate(self, authorization: str) -> APIKeyIdentity | None:
        scheme, separator, token = authorization.partition(" ")
        token = token.strip()
        if not separator or scheme.lower() != "bearer" or not token or any(char.isspace() for char in token):
            return None
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        identity = self._by_digest.get(digest)
        if identity is None or not hmac.compare_digest(digest, identity.digest):
            return None
        return identity


class TenantRateLimiter:
    """单进程 token bucket；每个已认证 tenant 独立计数。"""

    def __init__(self, requests_per_second: float, burst: int) -> None:
        if not math.isfinite(requests_per_second) or requests_per_second <= 0 or burst < 1:
            raise ValueError("API rate limit must have a positive rate and burst.")
        self.rate = requests_per_second
        self.burst = float(burst)
        self._buckets: dict[str, tuple[float, float]] = {}
        self._lock = threading.Lock()

    def allow(self, tenant_id: str, now: float | None = None) -> tuple[bool, int]:
        now = time.monotonic() if now is None else now
        with self._lock:
            tokens, last = self._buckets.get(tenant_id, (self.burst, now))
            tokens = min(self.burst, tokens + max(0.0, now - last) * self.rate)
            if tokens >= 1:
                self._buckets[tenant_id] = (tokens - 1, now)
                return True, 0
            self._buckets[tenant_id] = (tokens, now)
            return False, max(1, math.ceil((1 - tokens) / self.rate))


class APIKeyAuthMiddleware:
    """保护推理 API，并向 FastAPI request.state 注入 key 与 tenant 身份。"""

    def __init__(
        self,
        app: Any,
        *,
        api_keys: APIKeySet,
        limiter: TenantRateLimiter,
        exempt_paths: frozenset[str] = frozenset({"/health", "/v1/live/mediamtx/auth"}),
    ) -> None:
        self.app = app
        self.api_keys = api_keys
        self.limiter = limiter
        self.exempt_paths = exempt_paths

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        path = str(scope.get("path", ""))
        if path in self.exempt_paths or not path.startswith("/v1/"):
            await self.app(scope, receive, send)
            return

        authorization_headers = [
            value.decode("latin-1")
            for name, value in scope.get("headers", ())
            if name.lower() == b"authorization"
        ]
        identity = self.api_keys.authenticate(authorization_headers[0]) if len(authorization_headers) == 1 else None
        if identity is None:
            response = JSONResponse(
                status_code=401,
                content={"detail": "A valid Bearer API key is required."},
                headers={"WWW-Authenticate": "Bearer"},
            )
            await response(scope, receive, send)
            return

        allowed, retry_after = self.limiter.allow(identity.tenant_id)
        if not allowed:
            response = JSONResponse(
                status_code=429,
                content={"detail": "The tenant request rate limit has been exceeded."},
                headers={"Retry-After": str(retry_after)},
            )
            await response(scope, receive, send)
            return

        state = scope.setdefault("state", {})
        state["api_key_id"] = identity.key_id
        state["tenant_id"] = identity.tenant_id
        await self.app(scope, receive, send)


def parse_hmac_keys(value: str) -> tuple[bytes, ...]:
    """解析逗号分隔的 Base64URL HMAC 密钥轮替列表。"""
    keys: list[bytes] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            raise ValueError("Live session HMAC key list contains an empty value.")
        try:
            key = base64.urlsafe_b64decode(part + "=" * (-len(part) % 4))
        except (ValueError, binascii.Error) as exc:
            raise ValueError("Live session HMAC keys must be valid Base64URL values.") from exc
        if len(key) < 32:
            raise ValueError("Live session HMAC keys must contain at least 32 decoded bytes.")
        keys.append(key)
    return tuple(keys)


def sign_live_session_handle(tenant_id: str, session_id: str, expires_at: int, keys: tuple[bytes, ...]) -> str:
    """生成绑定 tenant 与有效期的 URL 安全 Live session 句柄。"""
    if not tenant_id or not session_id or expires_at <= 0 or not keys or len(keys[0]) < 32:
        raise ValueError("Live session handle signing inputs are invalid.")
    payload = json.dumps(
        {"v": "v1", "tenant": tenant_id, "session": session_id, "exp": expires_at},
        separators=(",", ":"),
    ).encode("utf-8")
    encoded = base64.urlsafe_b64encode(payload).rstrip(b"=").decode("ascii")
    signature = hmac.new(keys[0], encoded.encode("ascii"), hashlib.sha256).digest()
    return f"{encoded}.{base64.urlsafe_b64encode(signature).rstrip(b'=').decode('ascii')}"


def verify_live_session_handle(
    handle: str, tenant_id: str, keys: tuple[bytes, ...], now: int | None = None
) -> str | None:
    """验证句柄签名、tenant 和过期时间；无效时统一返回 None。"""
    parts = handle.split(".")
    if len(parts) != 2 or not tenant_id or not keys:
        return None
    try:
        encoded_payload = parts[0].encode("ascii")
        supplied_signature = base64.urlsafe_b64decode(parts[1] + "=" * (-len(parts[1]) % 4))
        payload_bytes = base64.urlsafe_b64decode(parts[0] + "=" * (-len(parts[0]) % 4))
    except (UnicodeEncodeError, ValueError, binascii.Error):
        return None
    valid_signature = False
    for key in keys:
        expected = hmac.new(key, encoded_payload, hashlib.sha256).digest()
        valid_signature = hmac.compare_digest(supplied_signature, expected) or valid_signature
    if not valid_signature:
        return None
    try:
        payload = json.loads(payload_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    current_time = int(time.time()) if now is None else now
    if (
        not isinstance(payload, dict)
        or payload.get("v") != "v1"
        or payload.get("tenant") != tenant_id
        or not isinstance(payload.get("session"), str)
        or not payload["session"]
        or type(payload.get("exp")) is not int
        or payload["exp"] <= current_time
    ):
        return None
    return payload["session"]


if __name__ == "__main__":
    """对认证中间件、限流与 Live 句柄做不依赖 pytest 的本地验收。"""

    async def check_request(
        middleware: APIKeyAuthMiddleware,
        authorization: str | tuple[str, ...] = "",
        path: str = "/v1/models",
    ) -> tuple[int, dict[str, Any]]:
        authorization_values = (authorization,) if isinstance(authorization, str) else authorization
        scope = {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.3"},
            "http_version": "1.1",
            "method": "GET",
            "scheme": "http",
            "path": path,
            "raw_path": path.encode("ascii"),
            "query_string": b"",
            "headers": [(b"authorization", value.encode("ascii")) for value in authorization_values if value],
            "state": {},
            "client": ("127.0.0.1", 12345),
            "server": ("127.0.0.1", 8000),
        }
        sent: list[dict[str, Any]] = []

        async def receive() -> dict[str, Any]:
            return {"type": "http.request", "body": b"", "more_body": False}

        async def send(message: dict[str, Any]) -> None:
            sent.append(message)

        await middleware(scope, receive, send)
        status = next(message["status"] for message in sent if message["type"] == "http.response.start")
        body = b"".join(message.get("body", b"") for message in sent if message["type"] == "http.response.body")
        return status, json.loads(body)

    async def run_checks() -> None:
        key_set = APIKeySet([
            {"id": "test-key-a", "tenant_id": "tenant-a", "key": "test-secret-at-least-16"},
            {"id": "test-key-b", "tenant_id": "tenant-b", "key": "another-test-secret-16"},
        ])

        async def downstream(scope: dict[str, Any], receive: Any, send: Any) -> None:
            state = scope["state"]
            await JSONResponse({"api_key_id": state.get("api_key_id"), "tenant_id": state.get("tenant_id")})(
                scope, receive, send
            )

        middleware = APIKeyAuthMiddleware(
            downstream, api_keys=key_set, limiter=TenantRateLimiter(1, 1)
        )
        assert (await check_request(middleware))[0] == 401
        assert (await check_request(middleware, "Bearer wrong-secret-value"))[0] == 401
        assert (await check_request(
            middleware, ("Bearer test-secret-at-least-16", "Bearer another-test-secret-16")
        ))[0] == 401
        assert await check_request(middleware, "Bearer test-secret-at-least-16") == (
            200, {"api_key_id": "test-key-a", "tenant_id": "tenant-a"}
        )
        assert await check_request(middleware, "Bearer another-test-secret-16") == (
            200, {"api_key_id": "test-key-b", "tenant_id": "tenant-b"}
        )
        limited_status, limited_body = await check_request(middleware, "Bearer test-secret-at-least-16")
        assert limited_status == 429 and "detail" in limited_body
        assert await check_request(middleware, path="/health") == (
            200, {"api_key_id": None, "tenant_id": None}
        )

        hmac_key = os.urandom(32)
        handle = sign_live_session_handle("tenant-a", "session-1", int(time.time()) + 30, (hmac_key,))
        assert verify_live_session_handle(handle, "tenant-a", (hmac_key,)) == "session-1"
        assert verify_live_session_handle(handle, "tenant-b", (hmac_key,)) is None
        assert verify_live_session_handle(handle, "tenant-a", (hmac_key,), now=int(time.time()) + 31) is None

    asyncio.run(run_checks())
    print("API key auth, tenant rate limiting, and Live session handle checks passed.")
