"""Qwen3 Hermes 风格工具输出与 OpenAI 工具协议之间的适配。"""

from __future__ import annotations

import json
import uuid
from typing import Any

from inference.api.contracts.tool_calling import (
    ChatOutputDelta,
    ParsedAssistantOutput,
    ParsedToolCall,
    ToolCallDelta,
    ToolCallStreamParser,
    tool_schema_by_name,
    validate_tool_arguments,
)


_OPEN = "<tool_call>"
_CLOSE = "</tool_call>"
_STREAM_ARGUMENT_CHARS = 64


class Qwen3ToolCallAdapter:
    """将 Qwen3 的 <tool_call> 块转换为 OpenAI-compatible function calls。"""

    def parse_output(
        self,
        text: str,
        tools: list[dict[str, Any]],
        *,
        allow_parallel: bool = False,
    ) -> ParsedAssistantOutput:
        schemas = tool_schema_by_name(tools)
        content: list[str] = []
        calls: list[ParsedToolCall] = []
        cursor = 0
        while True:
            start = text.find(_OPEN, cursor)
            stray_close = text.find(_CLOSE, cursor)
            if start < 0:
                if stray_close >= 0:
                    raise ValueError("Qwen3 output contains an unmatched </tool_call> marker.")
                content.append(text[cursor:])
                break
            if 0 <= stray_close < start:
                raise ValueError("Qwen3 output contains an unmatched </tool_call> marker.")
            content.append(text[cursor:start])
            end = text.find(_CLOSE, start + len(_OPEN))
            if end < 0:
                raise ValueError("Qwen3 output ended inside an incomplete <tool_call> block.")
            if len(calls) and not allow_parallel:
                raise ValueError("Qwen3 returned multiple tool calls while parallel_tool_calls is disabled.")
            calls.append(self._parse_call(text[start + len(_OPEN):end], schemas))
            cursor = end + len(_CLOSE)
        final_content = "".join(content)
        return ParsedAssistantOutput(final_content if final_content else None, tuple(calls))

    def create_stream_parser(
        self,
        tools: list[dict[str, Any]],
        *,
        allow_parallel: bool = False,
    ) -> ToolCallStreamParser:
        return Qwen3ToolCallStreamParser(self, tool_schema_by_name(tools), allow_parallel=allow_parallel)

    @staticmethod
    def _parse_call(
        raw: str, schemas: dict[str, dict[str, Any]], *, call_id: str | None = None
    ) -> ParsedToolCall:
        try:
            payload = json.loads(raw.strip(), object_pairs_hook=_unique_json_object)
        except json.JSONDecodeError as exc:
            raise ValueError("Qwen3 returned invalid JSON inside <tool_call>.") from exc
        if not isinstance(payload, dict) or set(payload) - {"name", "arguments"}:
            raise ValueError("Qwen3 tool call must contain only name and arguments fields.")
        name = payload.get("name")
        if not isinstance(name, str) or name not in schemas:
            raise ValueError(f"Qwen3 requested a tool that is not allowed: {name!r}.")
        arguments = validate_tool_arguments(name, payload.get("arguments"), schemas[name])
        encoded = json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))
        return ParsedToolCall(id=call_id or f"call_{uuid.uuid4().hex}", name=name, arguments=encoded)


class Qwen3ToolCallStreamParser:
    """增量识别 Qwen3 工具块，先解析完整 JSON，再分片发出标准 delta。"""

    def __init__(
        self,
        adapter: Qwen3ToolCallAdapter,
        schemas: dict[str, dict[str, Any]],
        *,
        allow_parallel: bool,
    ) -> None:
        self.adapter = adapter
        self.schemas = schemas
        self.allow_parallel = allow_parallel
        self._buffer = ""
        self._tool_payload = ""
        self._inside_tool = False
        self._calls: list[ParsedToolCall] = []
        self._streamed_call_id: str | None = None
        self._streamed_call_name: str | None = None
        self._arguments_start: int | None = None
        self._arguments_emitted = 0

    @property
    def tool_calls(self) -> tuple[ParsedToolCall, ...]:
        return tuple(self._calls)

    def feed(self, text: str) -> list[ChatOutputDelta]:
        if text:
            self._buffer += text
        output: list[ChatOutputDelta] = []
        while True:
            if not self._inside_tool:
                start = self._buffer.find(_OPEN)
                stray_close = self._buffer.find(_CLOSE)
                if stray_close >= 0 and (start < 0 or stray_close < start):
                    raise ValueError("Qwen3 output contains an unmatched </tool_call> marker.")
                if start >= 0:
                    self._emit_content(output, self._buffer[:start])
                    self._buffer = self._buffer[start + len(_OPEN):]
                    self._tool_payload = ""
                    self._inside_tool = True
                    continue
                keep = _suffix_prefix_length(self._buffer, _OPEN)
                safe_length = len(self._buffer) - keep
                self._emit_content(output, self._buffer[:safe_length])
                self._buffer = self._buffer[safe_length:]
                return output

            end = self._buffer.find(_CLOSE)
            if end < 0:
                self._tool_payload += self._buffer
                self._buffer = ""
                self._emit_tool_progress(output)
                return output
            self._tool_payload += self._buffer[:end]
            self._buffer = self._buffer[end + len(_CLOSE):]
            self._emit_tool_progress(output)
            if self._calls and not self.allow_parallel:
                raise ValueError("Qwen3 returned multiple tool calls while parallel_tool_calls is disabled.")
            call = self.adapter._parse_call(
                self._tool_payload, self.schemas, call_id=self._streamed_call_id
            )
            call_index = len(self._calls)
            self._calls.append(call)
            if self._streamed_call_id is None:
                output.append(ChatOutputDelta(tool_call=ToolCallDelta(
                    index=call_index,
                    id=call.id,
                    call_type="function",
                    name=call.name,
                    arguments="",
                )))
            self._inside_tool = False
            self._tool_payload = ""
            self._streamed_call_id = None
            self._streamed_call_name = None
            self._arguments_start = None
            self._arguments_emitted = 0

    def finish(self) -> list[ChatOutputDelta]:
        if self._inside_tool:
            raise ValueError("Qwen3 output ended inside an incomplete <tool_call> block.")
        if _suffix_prefix_length(self._buffer, _OPEN):
            raise ValueError("Qwen3 output ended inside an incomplete <tool_call> marker.")
        if _CLOSE in self._buffer:
            raise ValueError("Qwen3 output contains an unmatched </tool_call> marker.")
        output: list[ChatOutputDelta] = []
        self._emit_content(output, self._buffer)
        self._buffer = ""
        return output

    @staticmethod
    def _emit_content(output: list[ChatOutputDelta], text: str) -> None:
        if text:
            output.append(ChatOutputDelta(content=text))

    def _emit_tool_progress(self, output: list[ChatOutputDelta]) -> None:
        name, arguments_start = _top_level_tool_fields(self._tool_payload)
        if self._streamed_call_id is None and name is not None:
            if name not in self.schemas:
                raise ValueError(f"Qwen3 requested a tool that is not allowed: {name!r}.")
            if self._calls and not self.allow_parallel:
                raise ValueError("Qwen3 returned multiple tool calls while parallel_tool_calls is disabled.")
            self._streamed_call_id = f"call_{uuid.uuid4().hex}"
            self._streamed_call_name = name
            output.append(ChatOutputDelta(tool_call=ToolCallDelta(
                index=len(self._calls),
                id=self._streamed_call_id,
                call_type="function",
                name=name,
                arguments="",
            )))
        if self._arguments_start is None and arguments_start is not None:
            self._arguments_start = arguments_start
        if self._streamed_call_id is None or self._arguments_start is None:
            return
        complete_end = _json_object_end(self._tool_payload, self._arguments_start)
        available_end = complete_end if complete_end is not None else len(self._tool_payload)
        new_arguments = self._tool_payload[self._arguments_start:available_end]
        pending = new_arguments[self._arguments_emitted:]
        if not pending:
            return
        self._arguments_emitted = len(new_arguments)
        for offset in range(0, len(pending), _STREAM_ARGUMENT_CHARS):
            output.append(ChatOutputDelta(tool_call=ToolCallDelta(
                index=len(self._calls),
                arguments=pending[offset:offset + _STREAM_ARGUMENT_CHARS],
            )))


def _suffix_prefix_length(value: str, marker: str) -> int:
    """保留可能跨 token/chunk 切开的 marker 后缀。"""
    for length in range(min(len(value), len(marker) - 1), 0, -1):
        if value.endswith(marker[:length]):
            return length
    return 0


def _top_level_tool_fields(payload: str) -> tuple[str | None, int | None]:
    """读取顶层 name 与 arguments 位置，不误把参数内部同名字段当成工具名。"""
    cursor = 0
    while cursor < len(payload) and payload[cursor].isspace():
        cursor += 1
    if cursor == len(payload):
        return None, None
    if payload[cursor] != "{":
        raise ValueError("Qwen3 tool call must be a JSON object.")
    cursor += 1
    decoder = json.JSONDecoder()
    name: str | None = None
    arguments_start: int | None = None
    seen: set[str] = set()

    while True:
        while cursor < len(payload) and payload[cursor].isspace():
            cursor += 1
        if cursor >= len(payload) or payload[cursor] == "}":
            return name, arguments_start
        try:
            key, key_end = decoder.raw_decode(payload, cursor)
        except json.JSONDecodeError:
            # 字串可能刚好在 tokenizer chunk 边界被切开，等下一段再解析。
            return name, arguments_start
        if not isinstance(key, str) or key not in {"name", "arguments"}:
            raise ValueError("Qwen3 tool call must contain only name and arguments fields.")
        if key in seen:
            raise ValueError(f"Qwen3 tool call contains a duplicate {key!r} field.")
        seen.add(key)
        cursor = key_end
        while cursor < len(payload) and payload[cursor].isspace():
            cursor += 1
        if cursor >= len(payload):
            return name, arguments_start
        if payload[cursor] != ":":
            raise ValueError("Qwen3 tool call contains malformed JSON.")
        cursor += 1
        while cursor < len(payload) and payload[cursor].isspace():
            cursor += 1
        if cursor >= len(payload):
            return name, arguments_start
        if key == "name":
            try:
                value, value_end = decoder.raw_decode(payload, cursor)
            except json.JSONDecodeError:
                return name, arguments_start
            if not isinstance(value, str):
                raise ValueError("Qwen3 tool name must be a string.")
            name = value
            cursor = value_end
        else:
            if payload[cursor] != "{":
                raise ValueError("Qwen3 tool arguments must be a JSON object.")
            arguments_start = cursor
            value_end = _json_object_end(payload, cursor)
            if value_end is None:
                return name, arguments_start
            cursor = value_end

        while cursor < len(payload) and payload[cursor].isspace():
            cursor += 1
        if cursor >= len(payload):
            return name, arguments_start
        if payload[cursor] == ",":
            cursor += 1
            continue
        if payload[cursor] == "}":
            return name, arguments_start
        raise ValueError("Qwen3 tool call contains malformed JSON.")


def _json_object_end(value: str, start: int | None) -> int | None:
    """找到 JSON object 的闭合位置；返回值为 end-exclusive。"""
    if start is None or start >= len(value) or value[start] != "{":
        return None
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(value)):
        char = value[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return index + 1
    return None


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"Qwen3 tool call contains a duplicate {key!r} field.")
        value[key] = item
    return value
