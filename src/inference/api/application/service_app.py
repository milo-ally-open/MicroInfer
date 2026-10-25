"""依模型登录表建立共用 API Server。"""

from __future__ import annotations

from typing import Any

from inference.api.application.app import create_app
from inference.engine.transport.client import EngineClient
from inference.engine.config.model_config import (
    load_factory,
    model_api_settings,
    selected_config,
    selected_model,
)


model_alias, model_spec = selected_model()
inference_config = selected_config("INFERENCE_CONFIG")
api_config: dict[str, Any] = inference_config.get("api", {})
runtime_config: dict[str, Any] = inference_config.get("runtime", {})
api_settings = model_api_settings(model_spec)
if not isinstance(api_config, dict) or not isinstance(runtime_config, dict):
    raise ValueError("inference.yml api and runtime settings must be mappings.")

engine = EngineClient(
    "inference.engine.bootstrap.model_bootstrap:create_engine_core",
    startup_timeout_seconds=float(runtime_config.get("engine_core_startup_timeout_seconds", 2430)),
    request_timeout_seconds=float(runtime_config.get("engine_core_request_timeout_seconds", 1830)),
    max_video_bytes=int(api_config.get("max_video_bytes", 1)),
)

capabilities = api_settings.get("capabilities")
if not isinstance(capabilities, dict):
    raise ValueError("model.yml api.capabilities must be a mapping.")
chat_inputs = capabilities.get("chat_inputs")
if (
    not isinstance(chat_inputs, list)
    or not chat_inputs
    or any(not isinstance(value, str) or value not in {"text", "image"} for value in chat_inputs)
    or len(set(chat_inputs)) != len(chat_inputs)
):
    raise ValueError("model.yml api.capabilities.chat_inputs must contain text and/or image.")
streaming_enabled = capabilities.get("streaming")
tool_calling_enabled = capabilities.get("tool_calling")
video_enabled = capabilities.get("video_upload")
live_enabled = capabilities.get("live_video")
output_mode = capabilities.get("output_mode")
if any(not isinstance(value, bool) for value in (streaming_enabled, tool_calling_enabled, video_enabled, live_enabled)):
    raise ValueError("model.yml streaming, tool_calling, video_upload, and live_video capabilities must be booleans.")
if output_mode not in {"text_delta", "token_ids"}:
    raise ValueError("model.yml api.capabilities.output_mode must be text_delta or token_ids.")

input_processor_factory = api_settings.get("input_processor")
if input_processor_factory is not None and not isinstance(input_processor_factory, str):
    raise ValueError("model.yml api.input_processor must be a factory path string.")
profile = str(model_spec.get("integration_profile", "custom"))
if input_processor_factory:
    input_processor = load_factory(input_processor_factory)()
elif profile == "standard":
    from inference.api.processors.default_chat_processor import DefaultChatInputProcessor

    input_processor = DefaultChatInputProcessor()
else:
    input_processor = None
if input_processor is not None:
    required_processor_methods = {"start", "prepare_chat", "close"}
    if output_mode == "token_ids":
        required_processor_methods |= {"decode_tokens", "create_stream_output_processor"}
    missing_processor_methods = {
        name for name in required_processor_methods if not callable(getattr(input_processor, name, None))
    }
    if missing_processor_methods:
        raise ValueError(f"model input processor is missing required methods: {sorted(missing_processor_methods)}")
elif output_mode == "token_ids":
    raise ValueError("token_ids output requires an input processor with decoding and streaming output methods.")

tool_call_adapter_factory = api_settings.get("tool_call_adapter")
if tool_call_adapter_factory is not None and not isinstance(tool_call_adapter_factory, str):
    raise ValueError("model.yml api.tool_call_adapter must be a factory path string.")
if tool_calling_enabled and not tool_call_adapter_factory:
    raise ValueError("tool_calling capability requires api.tool_call_adapter in model.yml.")
if tool_call_adapter_factory:
    tool_call_adapter = load_factory(tool_call_adapter_factory)()
    required_adapter_methods = {"parse_output", "create_stream_parser"}
    missing_adapter_methods = {
        name for name in required_adapter_methods if not callable(getattr(tool_call_adapter, name, None))
    }
    if missing_adapter_methods:
        raise ValueError(f"model tool-call adapter is missing methods: {sorted(missing_adapter_methods)}")
else:
    tool_call_adapter = None

live_manager_factory = api_settings.get("live_session_manager")
if live_manager_factory is not None and not isinstance(live_manager_factory, str):
    raise ValueError("model.yml api.live_session_manager must be a factory path string.")
if live_enabled and not live_manager_factory:
    raise ValueError("Live sessions require api.live_session_manager in model.yml.")
if live_manager_factory:
    manager = load_factory(live_manager_factory)(engine)
    engine.attach_live_sessions(manager)

request_limits: dict[str, tuple[str, int]] = {
    "/v1/chat/completions": ("Chat request", int(api_config["max_chat_request_bytes"])),
}
if video_enabled:
    request_limits["/v1/videos/understand"] = (
        "Video upload request", int(api_config["max_video_request_bytes"])
    )
if live_enabled:
    max_live_bytes = int(api_config["max_live_request_bytes"])
    request_limits["/v1/live/sessions"] = ("Live session request", max_live_bytes)
    request_limits["/v1/live/mediamtx/auth"] = ("MediaMTX authorization request", max_live_bytes)

app = create_app(
    engine=engine,
    model_id=str(api_settings.get("model_id") or model_alias),
    title=f"{model_spec.get('display_name', model_alias)} API",
    request_limits=request_limits,
    input_processor=input_processor,
    tool_call_adapter=tool_call_adapter,
    tool_calling_enabled=tool_calling_enabled,
    chat_input_modalities=frozenset(chat_inputs),
    enable_streaming=streaming_enabled,
    enable_video=video_enabled,
    enable_live_sessions=live_enabled,
)
