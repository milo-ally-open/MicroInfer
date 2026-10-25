# 推理服务

推理服务作为 `inference` Python 包注册，由 `inference` CLI 启动。Launcher 自动发现各模型包中的 configs/model.yml；启动时必须用 --model 选择一个模型。推理默认优先使用 CUDA，仅在 CUDA 不可用时回退到 CPU。API Server、Engine Core 和 Model Worker 是独立进程；Go Gateway 与 Inference 分别配置和运行。

在所选模型的 Python 环境中，从仓库根目录执行 `python -m pip install -e "src/inference[mage-vl]"` 或 `python -m pip install -e "src/inference[qwen3]"`。只安装所选模型的依赖，不安装其他模型依赖。PyTorch/CUDA wheel 需按目标 GPU 环境预先安装；模型 extras 分别读取各模型包内的 `requirements.txt`。

## 快速检查与启动

从仓库根目录运行：

- inference list：列出已发现模型。
- inference validate --model {alias}：检查 manifest、YAML、Runtime 工厂路径和 API 能力，不导入模型工厂、不载入权重、不下载 snapshot、不初始化 GPU。
- inference start --model {alias}：启动指定模型。

目前登记的 alias 是 mage-vl-4b 和 qwen3-0.6b。模型 API 位址分别由各自的 configs/network.yml 决定。

## 最简模型引擎使用示例

以下以 Qwen3-0.6B 为例，在 Python 服务端导入模型插件包并建立 Engine。插件会读取自己的 manifest 与 YAML，再启动独立 Engine Core 和 Model Worker；Scheduler、KV 管理和模型 Runtime 仍由 Engine 负责。

Qwen3 模型依赖、CUDA PyTorch 和 `configs/inference.yml` 指定的本机 checkpoint 必须已准备好：

```python
from inference.models.qwen3_0_6b import create_engine_client

def main():
    engine = create_engine_client()
    engine.start()
    try:
        result = engine.generate_chat({
            "model": "qwen3-0.6b",
            "messages": [{"role": "user", "content": "用一句话介绍分页 KV Cache。"}],
            "max_tokens": 128,
        })
        print(result.content)
    finally:
        engine.close()


if __name__ == "__main__":
    main()
```

`generate_chat()` 返回 `InferenceResult`，生成文本位于 `result.content`。模型插件根据包内 manifest/YAML 选择模型；部署 wheel 时可在进程环境中设置 `INFERENCE_PROJECT_ROOT` 指向 checkpoints 和运行配置所在目录。将 Engine 嵌入长驻服务时，应在服务启动／关闭钩子中分别调用 `engine.start()`／`engine.close()`，不要为每个请求重新创建 Engine。

## 标准工具调用协议

共用 `POST /v1/chat/completions` 支援标准 function tool 请求与历史消息字段：`tools`、`tool_choice: "auto"/"none"`、`assistant.tool_calls`、`role: "tool"`／`tool_call_id`、非串流 `finish_reason: "tool_calls"` 和 SSE `delta.tool_calls`。首版每轮只接受一个调用；`parallel_tool_calls: true`、指定某个函数的 `tool_choice` 对象及 `strict: true` 会明确回错。工具 definitions 与模型生成参数都通过 JSON Schema 验证。

Inference 只负责把模型的原生格式解析为标准 tool call，不执行工具。调用方通过现有 `POST /v1/chat/completions` 提交工具定义，收到 `assistant.tool_calls` 后由客户端／Agent 应用自行校验并执行，再以 `role: "tool"` 发送结果并继续调用同一接口。Gateway 只转发标准请求和回应。

工具协议采用共用标准消息／Schema 层与模型专属 parser adapter 的组合。manifest `api.capabilities.tool_calling` 必须明确宣告；开启时需登记 `api.tool_call_adapter` factory，Launcher 会预检 adapter 方法，API 启动时载入该 adapter。当前只有 Qwen3-0.6B 宣告并实现该能力，使用 Qwen3 `<tool_call>...</tool_call>` 解析器；Mage-VL 明确关闭。未来模型沿用同一标准 API schema，只需为自己的原生输出格式实现 adapter，并在自身 manifest 登记。

## 工具调用流程

整个工具循环使用同一个 OpenAI 兼容 endpoint：

1. 调用方将 `tools` 和 `tool_choice="auto"` 送至 `/v1/chat/completions`。
2. Inference 回传标准 `tool_calls` 或 SSE `delta.tool_calls`；Gateway 原样转送协议字段并改写公开模型 alias。
3. 客户端／Agent 应用按自己的工具注册表校验名称和参数，执行获准的 handler，再将 assistant tool call 与 `role: "tool"` 结果追加到消息历史，继续调用同一 endpoint。

## 新模型接入概要

详细步骤和文件内容要求见 [新模型接入指南](../../docs/Specifications/inference/model-integration.md)。模型包目录名是 Python 包名，alias 是客户端选模名称，两者可以不同；例如 qwen3_0_6b 对应 qwen3-0.6b。上游权重 ID 单独写在 inference.yml 的 model.id。

| 文件或目录 | 必需性 | 内容职责 |
| --- | --- | --- |
| models/{package_name}/configs/model.yml | 必需 | alias、display name、standard/custom profile、backend、Runtime 工厂、配置路径、api.model_id 和已实现的能力；工具调用模型另登记 `api.tool_call_adapter`。 |
| models/{package_name}/configs/inference.yml | 必需 | 固定模型 revision、dtype/device/量化、权重路径、服务限制、generation、Scheduler/KV、Worker 超时和优化参数。 |
| models/{package_name}/configs/network.yml | 必需 | Inference API 监听位址；使用 Live 时另列推理服务所需媒体端点。 |
| models/{package_name}/runtime.py | 必需 | create_runtime() 工厂及模型 Runner/Adapter 的组合。 |
| models/{package_name}/processor.py | 按模型需要 | Tokenizer、Chat Template、文本规范化、图片/视频预处理和模型专属采样参数适配。 |
| models/{package_name}/utils/loader.py | 按模型需要 | 集中选择 Transformers Config、Tokenizer/Processor、snapshot 和权重载入方式。 |
| model/、layers/、components/ | 按模型需要 | 分别承载拓扑、数值计算层、Runner/Adapter/Batch/Cache/Sampler 等执行元件。 |
| models/{package_name}/requirements.txt | 按模型需要 | 模型专属依赖及兼容版本；沿用项目指定环境，不为每个模型重复创建 Conda 环境。 |
| tests/{package_name}/ | 必需 | test.yml 和 test_components.py、test_performance.py、test_end2end.py 三项测试。 |

runtime.py 是每个模型包都需要的工厂入口。符合当前 standard 条件的模型可以使用共用文字 Runner，减少模型专属 Runner/Adapter/Processor；standard 当前只适用于单装置、纯文字 decoder-only、token_ids 输出和可延续 past_key_values 的模型。多模态、特殊 Cache、自有计算图或权重映射使用 custom。profile 只表达接入路径，不会自动提供模型实现。

api.model_id 是 Inference API 接受并返回的模型 ID；model.id 是上游模型 repo ID；Gateway 公开 alias 由模型 alias 决定。通过 Gateway 开放新模型时，修改 src/gateway/configs/gateway.yml 的模型映射，使 upstream_model 与 api.model_id 一致、base_url 指向 network.yml 的 Inference API，并只宣告已实现的 Gateway 能力。

## 配置边界

所有模型服务和推理参数放在对应模型的 configs/inference.yml；API 监听与模型侧媒体连接端点放在 configs/network.yml。API Key、HMAC 密钥和密码只通过环境变量或 Git 外部的密钥文件提供，不写入 YAML 或 Git。

MediaMTX 是独立 sidecar，不属于模型包、manifest 或 Launcher。配置统一位于 deploy/mediamtx/mediamtx.yml，由部署者单独启动。模型 network.yml 只描述推理服务连接媒体端点所需的信息。独立部署方式见 [MediaMTX README](../../deploy/mediamtx/README.md)；图片和上传视频不依赖 MediaMTX。

## 共用架构边界

| 组件 | 位置 | 职责 |
| --- | --- | --- |
| API Server | `network/` | `app.py` 装配应用；`service.py` 建立模型服务；`schemas.py`、`serializers.py`、`tool_calling.py` 管理 HTTP 协议；各路由和输入／输出处理器直接放在此目录。 |
| 共用 Processors | `processors/` | 视频文件读写与元数据／影格辅助、Live session 管理等跨模型处理能力；可按后续需求增加共用处理模块。 |
| Engine Core | `engine/inference_engine.py` | 请求生命周期、调度循环和输出事件分发。 |
| Scheduler 与 KV 管理 | `engine/scheduler.py`、`engine/block_manager.py` | 建立 prefill/decode 批次并管理 Engine 侧逻辑资源租约。 |
| Worker 与 IPC | `engine/worker.py`、`engine/engine_client.py` | 子进程、IPC、模型操作调用及资源清理。 |
| 服务启动与配置 | `engine/bootstrap.py`、`engine/config.py` | 组合 Engine／Worker，读取启动器选定的模型及运行配置。 |
| 模型 Runtime | models/{package_name}/ | 模型载入、拓扑、模型专属输入与 Cache 行为。 |

API 层以 `network/` 作为单一平面目录，包含应用装配、模型服务入口、schemas、serializers、工具调用协议、HTTP 路由和处理器。Engine 保留少量根目录模块：`sequence.py` 定义跨进程请求资料，`scheduler.py` 管理逐轮批次，`block_manager.py` 管理 Engine 侧 KV 租约，`inference_engine.py` 编排请求，`worker.py` 管理 Worker 与本机执行，`engine_client.py` 负责 API 到 Engine Core 的 IPC，`config.py`／`bootstrap.py` 负责读取模型设定和装配。模型包继续拥有模型专属 Processor、Runner、Adapter、拓扑和算子；Engine 不额外维护插件注册框架。

`processors/` 放跨模型可复用的请求处理组件，例如 `video.py` 的上传视频与媒体读取辅助，以及 `live_sessions.py` 的 Live session 管理。模型 tokenizer、chat template、模型输入格式和视觉预处理仍由模型包负责；新增共用处理能力时可在此扩展，不需把通用模块塞入 Network 或某个模型包。

模型 Runner 批次接口不代表一定执行合并 forward。Qwen3 使用实体 KV page pool；PyTorch backend 的 Attention 会 gather KV pages。Qwen3 模型包内的选用式 native CUDA 单 token decode Attention 已在本机 RTX 4060 Laptop 完成构建、float16/bfloat16 数值对照、checkpoint 单层 profile 和 CUDA backend API/Gateway SSE 验收。当前测试显示它慢于 PyTorch 路径，所以 YAML 默认仍选 `pytorch`；不宣称性能提升或生产环境普遍适用。Mage-VL 使用 Transformers Cache，Engine KV 管理仍是逻辑 reservation。项目借鉴 vLLM/nano-vLLM 的组件职责，不表示兼容 vLLM。

### Mage-VL 模型组件地图

Mage-VL 的模型拓扑入口是 `models/mage_vl_4b/model/mage_vl.py`，对应 Qwen3 的 `model/qwen3.py`：只组装视觉塔、Qwen3 文本骨干和模型前向。条件生成与 Transformers `GenerationMixin` 契约留在 `conditional_generation.py`。

| 组件 | 文件 | 职责 |
|---|---|---|
| 模型拓扑与前向 | `model/mage_vl.py` | 装配视觉塔／文本骨干、确定位置与调用顺序 |
| 视觉特征 | `model/vision_features.py` | 图像／视频编码输出切分、placeholder token 数量校验 |
| 多模态输入注入 | `model/multimodal_inputs.py` | 将视觉特征按 image／video mask 写入文字 embedding |
| StreamMind 门控 | `model/streammind.py` | 可选 Gate 权重载入与单段／连续片段推理 |
| 视觉塔 | `model/vision_tower.py` | 汇整预处理基底、图像 patch 编码与视觉输出 |
| 视觉 RoPE | `layers/vision_rope.py` | T/H/W 位置频率生成与旋转计算 |
| Patch 表示 | `layers/vision_embeddings.py` | Patch embedding 与空间 patch merger |
| 视觉注意力 | `layers/vision_attention.py` | normalization、QKᵀ／softmax／V eager 路径 |
| Encoder 堆叠 | `layers/vision_stack.py` | 视觉 block 与多层 encoder |

`model/multimodal_model.py` 与 `layers/vision_encoder.py` 保留旧导入路径；后者汇出新的视觉组件，并保留 CUDA 优先、CUDA 不可用时回退 CPU 的前向对照入口。模块边界不增加模型参数层级，checkpoint 的 state-dict key 保持原路径。

## 三项模型测试

每个模型目录只维护三项测试，设定和本机路径由 tests/{package_name}/test.yml 提供，执行时明确传入配置：

1. test_components.py：检查模型结构和组件基线；meta 结构检查不载入权重，不做大型 GPU 测试，但需要 YAML 指向本机 snapshot 的配置资料。
2. test_performance.py：比较已启动的 Inference API 与同 revision Transformers 基线的生成表现；需要本机权重和推理 API，优先 CUDA，CUDA 不可用时使用 CPU。
3. test_end2end.py：通过 Go Gateway 验证公开 alias、认证、SSE 和模型已声明的多模态路由；需要 Gateway、Inference API、API Key 及适用素材。

测试脚本将 JSON 摘要输出到标准输出，不会自动写入文件。需要归档时，可将结果保存到对应模型目录的 results/，并记录模型 revision、硬件和测试环境。三项测试不做资料集精度评估。性能测试是实测比较，不自动构成 SLA；未运行的测试应保持未验收。

## 模型文件内的小型基线入口

独立数值计算的模型层、视觉组件、KV/采样执行组件和模型前向文件，在自身 if __name__ == "__main__": 入口中直接建立小型输入、运行引擎实现与基线、打印 ==========engine=========== 和 ==========baseline========== 并断言输出。入口不通过另行封装的测试方法执行。纯配置、输出结构和无独立前向行为的文件不需要空测试。

在已安装所选模型 extra 的 Python 环境中，从仓库根目录用 `python -m inference.models.{package_name}.{module_path}` 执行对应模块。Mage-VL 和 Qwen3 的模型层、视觉组件、模型拓扑与 KV/采样入口均采用此模式；Model Runner 入口还需要显式提供其模型 alias、inference.yml 和 network.yml 环境变量。模块入口使用小型合成配置，前向默认使用 CUDA，仅在 CUDA 不可用时使用 CPU；它们不读取完整模型权重，也不启动服务。VideoMamba 的入口在可选依赖 mamba_ssm 缺失时会明确跳过对应分支。

Mage-VL 的拆分视觉层、特征提取、多模态注入和 StreamMind 前处理入口会动态导入本机固定 revision 快照中的 `modeling_mage_vl.py` 作参考，并以随机小型模型参数对齐 key 后比较前向结果；它们不载入 checkpoint tensor。Qwen3 的层、模型和 Runner 入口对照 Transformers 官方 Qwen3 实现。新增计算模块应在自身 `__main__` 中写直接前向、打印 engine/baseline 并断言结果，不将对照包进外部测试函数。

真实模型性能和 Gateway 链路使用上面的逐模型三项测试。测试目录、配置字段及能力覆盖要求见 [新模型接入指南](../../docs/Specifications/inference/model-integration.md)，设计约束见 [Inference Specifications](../../docs/Specifications/inference/)。
