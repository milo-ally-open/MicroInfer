# VideoStreamAPI

本项目提供面向视频理解业务的模型推理 API 与独立的 Go API Gateway。Gateway 负责外部接入、认证、租户限流、模型路由和请求转发；Inference 负责模型加载、输入处理、生成及 GPU 资源管理。两者分别配置、启动和验收。

## 当前模型

| Gateway 公开 alias | 模型包目录 | 当前能力 | 实现方式 |
| --- | --- | --- | --- |
| mage-vl-4b | src/inference/models/mage_vl_4b/ | 图片聊天、文字 SSE、视频上传、Live | Mage-VL Transformers Runtime |
| qwen3-0.6b | src/inference/models/qwen3_0_6b/ | 纯文字聊天、文字 SSE | 模型包内 PyTorch 计算图与批次 Runner |

模型包目录名是 Python 包名，公开 alias 是客户端选择模型的名称，二者不必相同。上游模型 ID 则记录在模型 inference.yml 的 model.id 中。Gateway 使用模型 alias 接收请求，并将请求映射到 Inference API 的 api.model_id。

Inference 的 API、Engine Core、Scheduler、Executor 和 Worker 由两个模型共用；模型 Runtime、Processor、拓扑、权重载入和 Cache 行为留在模型包。项目参考 vLLM 与 nano-vLLM 的职责划分，但没有 vLLM backend、自定义 PagedAttention kernel、prefix cache 或其他未登记模型。

## 启动与检查

从项目根目录运行 Launcher 的 list、validate 或 start 命令，并在启动时用 --model 明确选择模型。当前 alias 为 mage-vl-4b 和 qwen3-0.6b。Inference API 位址由各自模型的 configs/network.yml 决定：Mage-VL 默认监听 127.0.0.1:8001，Qwen3 默认监听 127.0.0.1:8011；Gateway 默认监听 127.0.0.1:8000。

模型服务参数放在对应模型的 configs/inference.yml。Inference 服务启动、设定说明和独立 MediaMTX 部署方法见 [Inference README](src/inference/README.md)。

## 新模型接入

新模型以一个模型包为主体接入，不复制 API Server、Engine Core 或 Worker。建议按以下顺序准备：

1. 固定上游模型 revision，确认输入模态、Tokenizer/Processor、Cache、输出方式和设备需求。
2. 新建模型目录；填写 configs/model.yaml、configs/inference.yml 和 configs/network.yml。
3. 新增 runtime.py 工厂，并按需要增加 processor.py、utils/loader.py、model/、layers/、components/ 和 requirements.txt。
4. 在 tests/{package_name}/ 准备 test.yaml、test_components.py、test_performance.py 和 test_end2end.py。
5. 需要经 Gateway 对外开放时，再更新 src/gateway/configs/gateway.yml；使用 Live 时另行配置 deploy/mediamtx/。

完整的文件清单、各字段职责、能力限制、测试次序和 Gateway/MediaMTX 条件见 [新模型接入指南](docs/Specifications/inference/model-integration.md)。本文不提供模型实现代码示例。

## API 与测试

Gateway 提供 OpenAI 风格聊天 API、视频上传理解与 Live 会话 API。现有路由能力依模型 manifest 和 Gateway 配置限制；聊天文字 SSE 不代表视频输入必须逐帧或逐 token 返回。RTSP/RTMP 媒体流由发布端直连 MediaMTX，不通过 Gateway 转发。

逐模型的三个测试入口位于 tests/mage_vl_4b/ 和 tests/qwen3_0_6b/，分别检查模型结构与组件、比较真实 Inference API 性能、通过 Gateway 验证端到端调用。测试方式见 [Tests README](tests/README.md)。这些测试不做模型数据集精度评估。checkpoints/val.md 和 checkpoints/val.xlsx 是受保护的验收工作流文件。
