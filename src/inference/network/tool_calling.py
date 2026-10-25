"""跨模型共用的工具调用协议类型与验证。"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class ParsedToolCall:
    """已从模型原生输出解析出的 OpenAI 风格工具调用。"""

    id: str
    name: str
    arguments: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": "function",
            "function": {"name": self.name, "arguments": self.arguments},
        }


@dataclass(frozen=True)
class ParsedAssistantOutput:
    content: str | None
    tool_calls: tuple[ParsedToolCall, ...] = ()


@dataclass(frozen=True)
class ToolCallDelta:
    index: int
    id: str | None = None
    call_type: str | None = None
    name: str | None = None
    arguments: str | None = None

    def as_dict(self) -> dict[str, Any]:
        call: dict[str, Any] = {"index": self.index}
        if self.id is not None:
            call["id"] = self.id
        if self.call_type is not None:
            call["type"] = self.call_type
        function: dict[str, Any] = {}
        if self.name is not None:
            function["name"] = self.name
        if self.arguments is not None:
            function["arguments"] = self.arguments
        if function:
            call["function"] = function
        return call


@dataclass(frozen=True)
class ChatOutputDelta:
    content: str | None = None
    tool_call: ToolCallDelta | None = None


class ToolCallStreamParser(Protocol):
    """把模型文字增量转换成内容或结构化工具调用增量。"""

    @property
    def tool_calls(self) -> tuple[ParsedToolCall, ...]: ...

    def feed(self, text: str) -> list[ChatOutputDelta]: ...

    def finish(self) -> list[ChatOutputDelta]: ...


class ToolCallAdapter(Protocol):
    """模型专属的原生工具调用格式适配器。"""

    def parse_output(self, text: str, tools: list[dict[str, Any]], *, allow_parallel: bool) -> ParsedAssistantOutput: ...

    def create_stream_parser(
        self, tools: list[dict[str, Any]], *, allow_parallel: bool
    ) -> ToolCallStreamParser: ...


_TOOL_NAME = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def normalize_chat_messages(raw_messages: Any) -> list[dict[str, Any]]:
    """保留标准 assistant.tool_calls 与 tool 结果，并检查工具历史闭合。"""
    if not isinstance(raw_messages, list) or not raw_messages:
        raise ValueError("messages must be a non-empty list.")
    messages: list[dict[str, Any]] = []
    pending: dict[str, str] = {}
    seen_call_ids: set[str] = set()
    for message in raw_messages:
        if not isinstance(message, dict):
            raise ValueError("Each message must be an object.")
        role = message.get("role")
        if role not in {"system", "user", "assistant", "tool"}:
            raise ValueError("Message role must be system, user, assistant, or tool.")
        raw_content = message.get("content", "")
        if isinstance(raw_content, list):
            parts: list[str] = []
            for part in raw_content:
                if not isinstance(part, dict) or part.get("type") != "text":
                    raise ValueError("Tool-enabled text models accept text-only message content.")
                parts.append(str(part.get("text", "")))
            content: str | None = "".join(parts)
        elif raw_content is None and role == "assistant":
            content = None
        elif isinstance(raw_content, str):
            content = raw_content
        else:
            raise ValueError("Message content must be text; only assistant tool-call messages may use null.")

        normalized: dict[str, Any] = {"role": role, "content": content}
        if role != "assistant" and message.get("tool_calls") is not None:
            raise ValueError("Only assistant messages may contain tool_calls.")
        if role != "tool" and message.get("tool_call_id") is not None:
            raise ValueError("Only tool messages may contain tool_call_id.")
        name = message.get("name")
        if name is not None:
            if not isinstance(name, str) or not name:
                raise ValueError("Message name must be a non-empty string when provided.")
            normalized["name"] = name

        if role == "assistant":
            raw_calls = message.get("tool_calls")
            if raw_calls is None:
                raw_calls = []
            if not isinstance(raw_calls, list):
                raise ValueError("assistant.tool_calls must be a list.")
            calls: list[dict[str, Any]] = []
            for call in raw_calls:
                if not isinstance(call, dict) or call.get("type", "function") != "function":
                    raise ValueError("Only function assistant.tool_calls are supported.")
                call_id = call.get("id")
                function = call.get("function")
                if not isinstance(call_id, str) or not call_id or call_id in seen_call_ids:
                    raise ValueError("Each assistant tool call requires a unique non-empty id.")
                if not isinstance(function, dict):
                    raise ValueError("Each assistant tool call requires a function object.")
                function_name = function.get("name")
                arguments = function.get("arguments")
                if not isinstance(function_name, str) or not _TOOL_NAME.fullmatch(function_name):
                    raise ValueError("Each assistant tool call requires a valid function name.")
                if not isinstance(arguments, str):
                    raise ValueError("Tool call function.arguments must be a JSON string.")
                try:
                    parsed_arguments = json.loads(arguments, object_pairs_hook=_unique_json_object)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Tool call {function_name!r} has invalid JSON arguments.") from exc
                if not isinstance(parsed_arguments, dict):
                    raise ValueError("Tool call arguments must decode to a JSON object.")
                calls.append({
                    "id": call_id,
                    "type": "function",
                    "function": {"name": function_name, "arguments": arguments},
                })
                pending[call_id] = function_name
                seen_call_ids.add(call_id)
            if calls:
                normalized["tool_calls"] = calls
            if content is None and not calls:
                raise ValueError("An assistant message with null content must include tool_calls.")

        elif role == "tool":
            call_id = message.get("tool_call_id")
            if not isinstance(call_id, str) or call_id not in pending:
                raise ValueError("Each tool message must reference a preceding assistant tool_call_id.")
            expected_name = pending.pop(call_id)
            if name is not None and name != expected_name:
                raise ValueError("Tool message name does not match its assistant tool call.")
            normalized["tool_call_id"] = call_id
            normalized["name"] = expected_name
            if content is None:
                raise ValueError("Tool message content must be text.")
        elif content is None:
            raise ValueError(f"{role} message content must be text.")

        messages.append(normalized)
    if pending:
        raise ValueError("Every assistant tool call in the conversation must have a matching tool result.")
    return messages


def validate_tool_definitions(raw_tools: Any) -> list[dict[str, Any]]:
    """验证 OpenAI function tool 定义及其中的 JSON Schema。"""
    if not isinstance(raw_tools, list):
        raise ValueError("tools must be a list.")
    if len(raw_tools) > 32:
        raise ValueError("At most 32 tools may be provided in one request.")
    try:
        from jsonschema import Draft202012Validator
    except ImportError as exc:
        raise RuntimeError("Tool calling requires the jsonschema package in the inference environment.") from exc

    normalized: list[dict[str, Any]] = []
    names: set[str] = set()
    for item in raw_tools:
        if not isinstance(item, dict) or item.get("type") != "function":
            raise ValueError("Only tools with type='function' are supported.")
        function = item.get("function")
        if not isinstance(function, dict):
            raise ValueError("Each function tool requires a function object.")
        name = function.get("name")
        if not isinstance(name, str) or not _TOOL_NAME.fullmatch(name):
            raise ValueError("Function names must match [A-Za-z0-9_-]{1,64}.")
        if name in names:
            raise ValueError(f"Tool name {name!r} is duplicated.")
        names.add(name)
        description = function.get("description", "")
        if not isinstance(description, str) or len(description) > 2048:
            raise ValueError(f"Tool {name!r} description must be text no longer than 2048 characters.")
        parameters = function.get("parameters", {"type": "object", "properties": {}})
        if not isinstance(parameters, dict) or parameters.get("type") != "object":
            raise ValueError(f"Tool {name!r} parameters must be a JSON Schema object.")
        try:
            Draft202012Validator.check_schema(parameters)
        except Exception as exc:
            raise ValueError(f"Tool {name!r} has an invalid JSON Schema: {exc.message if hasattr(exc, 'message') else exc}") from exc
        strict = function.get("strict", False)
        if strict is True:
            raise ValueError("strict=true is not supported because this runtime does not constrain decoding to JSON Schema.")
        if strict is not None and not isinstance(strict, bool):
            raise ValueError(f"Tool {name!r} strict must be a boolean.")
        normalized.append({
            "type": "function",
            "function": {"name": name, "description": description, "parameters": parameters},
        })
    return normalized


def validate_tool_arguments(name: str, arguments: Any, schema: dict[str, Any]) -> dict[str, Any]:
    """验证模型生成的实参；执行工具前必须通过此函数。"""
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments, object_pairs_hook=_unique_json_object)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Tool {name!r} returned invalid JSON arguments.") from exc
    if not isinstance(arguments, dict):
        raise ValueError(f"Tool {name!r} arguments must be a JSON object.")
    try:
        from jsonschema import Draft202012Validator
    except ImportError as exc:
        raise RuntimeError("Tool execution requires the jsonschema package.") from exc
    errors = sorted(
        Draft202012Validator(schema).iter_errors(arguments),
        key=lambda error: tuple(str(part) for part in error.path),
    )
    if errors:
        first = errors[0]
        location = ".".join(str(part) for part in first.path) or "arguments"
        raise ValueError(f"Tool {name!r} arguments failed schema validation at {location}: {first.message}")
    return arguments


def tool_schema_by_name(tools: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {tool["function"]["name"]: tool["function"]["parameters"] for tool in tools}


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"JSON object contains a duplicate {key!r} key.")
        value[key] = item
    return value
