# VideoStreamAPI

本项目提供面向视频理解业务的模型推理 API 与独立的 Go API Gateway。Gateway 负责外部接入、认证、租户限流、模型路由和请求转发；Inference 负责模型加载、输入处理、生成及 GPU 资源管理。两者分别配置、启动和验收。

## 当前模型

| Gateway 公开 alias | 模型包目录 | 当前能力 | 实现方式 |
| --- | --- | --- | --- |
| mage-vl-4b | src/inference/models/mage_vl_4b/ | 图片聊天、文字 SSE、视频上传、Live | Mage-VL Transformers Runtime |
| qwen3-0.6b | src/inference/models/qwen3_0_6b/ | 纯文字聊天、文字 SSE、标准工具调用 | 模型包内 PyTorch 计算图与批次 Runner |

模型包目录名是 Python 包名，公开 alias 是客户端选择模型的名称，二者不必相同。上游模型 ID 则记录在模型 inference.yml 的 model.id 中。Gateway 使用模型 alias 接收请求，并将请求映射到 Inference API 的 api.model_id。

Inference 的 API、Engine Core、Scheduler、Executor 和 Worker 由两个模型共用；模型 Runtime、Processor、拓扑、权重载入和 Cache 行为留在模型包。项目参考 vLLM 与 nano-vLLM 的职责划分，但没有 vLLM backend、自定义 PagedAttention kernel、prefix cache 或其他未登记模型。

## 启动与检查

Inference 使用 `src/inference/pyproject.toml` 注册为 `inference` Python 包，发行名称和导入命名空间都为 `inference`。开发时在所选模型环境中安装 editable package 和对应模型 extra；基础安装不会安装所有模型依赖：

```bash
# Mage-VL 环境
python -m pip install -e "src/inference[mage-vl]"

# Qwen3 环境（先安装与 CUDA 匹配的 PyTorch）
python -m pip install -e "src/inference[qwen3]"
```

PyTorch/CUDA wheel 不由 model extra 自动选择，需按目标机器安装。两个 extra 分别读取模型目录内的 `requirements.txt`。安装后可用 `inference list`、`inference validate --model mage-vl-4b` 检查，再用 `inference start --model mage-vl-4b` 启动。开发 editable 安装会以仓库根目录作为项目根；部署 wheel 时通过 `INFERENCE_PROJECT_ROOT` 指定 checkpoints 和运行设定所在目录。当前 alias 为 mage-vl-4b 和 qwen3-0.6b。Inference API 位址由各自模型的 configs/network.yml 决定：Mage-VL 默认监听 127.0.0.1:8001，Qwen3 默认监听 127.0.0.1:8011；Gateway 默认监听 127.0.0.1:8000。
推理优先使用 CUDA；仅当 CUDA 不可用时回退到 CPU。CPU 模式下模型量化会停用，运行速度和可容纳模型大小会受主机内存限制。

模型服务参数放在对应模型的 configs/inference.yml。Inference 服务启动、设定说明和独立 MediaMTX 部署方法见 [Inference README](src/inference/README.md)。

## Qwen3 标准工具调用

Qwen3 的 `POST /v1/chat/completions` 支援 OpenAI 风格 `tools`、`tool_choice: "auto"/"none"`、assistant `tool_calls`、tool 结果消息及 SSE `delta.tool_calls`。Inference 解析模型原生输出并回传标准工具调用；与 OpenAI 兼容部署方式一致，工具由客户端／Agent 应用执行，再将 `role: "tool"` 结果通过同一 Chat Completions 接口送回模型。Gateway 只负责认证、租户政策、能力检查与转送，不执行工具。

工具调用部署只需启动 Qwen3 Inference（`127.0.0.1:8011`）和 Gateway（`127.0.0.1:8000`）。最简 OpenAI SDK 对话、客户端工具执行循环及现有视频理解 API 示例见 [tests/client.py](tests/client.py)；Inference 与 Gateway 启动说明见 [Inference README](src/inference/README.md) 和 [Gateway README](src/gateway/README.md)。

## 新模型接入

新模型以一个模型包为主体接入，不复制 API Server、Engine Core 或 Worker。建议按以下顺序准备：

1. 固定上游模型 revision，确认输入模态、Tokenizer/Processor、Cache、输出方式和设备需求。
2. 新建模型目录；填写 configs/model.yml、configs/inference.yml 和 configs/network.yml。
3. 新增 runtime.py 工厂，并按需要增加 processor.py、utils/loader.py、model/、layers/、components/ 和 requirements.txt。
4. 在 tests/{package_name}/ 准备 test.yml、test_components.py、test_performance.py 和 test_end2end.py。
5. 需要经 Gateway 对外开放时，再更新 src/gateway/configs/gateway.yml；使用 Live 时另行配置 deploy/mediamtx/。

完整的文件清单、各字段职责、能力限制、测试次序和 Gateway/MediaMTX 条件见 [新模型接入指南](docs/Specifications/inference/model-integration.md)。本文不提供模型实现代码示例。

## API 与测试

Gateway 提供 OpenAI 风格聊天 API、视频上传理解与 Live 会话 API。现有路由能力依模型 manifest 和 Gateway 配置限制；聊天文字 SSE 不代表视频输入必须逐帧或逐 token 返回。RTSP/RTMP 媒体流由发布端直连 MediaMTX，不通过 Gateway 转发。

逐模型的三个测试入口位于 tests/mage_vl_4b/ 和 tests/qwen3_0_6b/，分别检查模型结构与组件、比较真实 Inference API 性能、通过 Gateway 验证端到端调用。Qwen3 组件入口可集成运行模块级前向对照，并以可选 CUDA Profiler 分析 Attention；Qwen3 算子分析、优化和验收工作流见 `analysis/qwen3_0_6b/analysis.md` 与 `analysis.xlsx`。测试方式见 [Tests README](tests/README.md)。这些测试不做模型数据集精度评估。
