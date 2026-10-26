from pathlib import Path

from inference.models.qwen3_0_6b import create_engine_client
from inference.models.qwen3_0_6b.components.decoder import Qwen3Decoder
from inference.models.qwen3_0_6b.components.encoder import Qwen3Encoder
from inference.models.qwen3_0_6b.components.tokenizer import Qwen3Tokenizer

INFERENCE_CONFIG = Path(__file__).resolve().parent / "configs" / "inference.yml"


# 工厂在启动前登记模型配置；各组件保持独立，按推理顺序显式组合。
engine = create_engine_client(inference_config_path=INFERENCE_CONFIG)
tokenizer = Qwen3Tokenizer()
encoder = Qwen3Encoder()
decoder = Qwen3Decoder(tokenizer)
request = {
    "model": "qwen3-0.6b",
    "messages": [{"role": "user", "content": "详细介绍一下 Paged Attention。"}],
    "max_tokens": 256,
}

# 启动推理引擎和模型 tokenizer。
engine.start()
tokenizer.start()

# Encoder 规范化消息与采样参数；Tokenizer 套用 Qwen3 Chat Template 并生成 token IDs。
plan = encoder.prepare(request)
prompt_token_ids = tokenizer.tokenize_chat(
    plan.messages,
    tools=plan.template_tools,
    enable_thinking=plan.enable_thinking,
)
prepared_input = encoder.build(plan, prompt_token_ids)

# Engine 逐 token 返回事件；Decoder 创建的 Detokenizer 负责增量还原文本。
detokenizer = decoder.create_detokenizer()
stream = engine.stream_chat(request, prepared_input)
for event in stream:
    if event.kind == "token" and event.token_id is not None:
        print(detokenizer.push(event.token_id), end="", flush=True)
    elif event.kind == "finish":
        print(detokenizer.finish(), end="", flush=True)
print()

# 正常结束后关闭串流、Tokenizer 和 Engine。
stream.close()
tokenizer.close()
engine.close()
