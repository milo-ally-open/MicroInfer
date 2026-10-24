# VideoStreamAPI

本专案提供模型专属推论服务与独立的 Go API Gateway。Gateway 负责外部 API、认证与转送；Inference 负责模型载入、生成与 GPU 资源管理。两者可独立部署及测试。

## 推论模型与启动

模型别名、能力、Runtime 工厂及设定档由各模型目录下的 `configs/model.yaml` 登记；启动时以 `--model` 明确选择模型，不维护全域模型登录档。模型程式、设定及依赖集中于 `src/inference/models/<model_alias>/`。模型接入步骤见[新模型指南](docs/Specifications/inference/model-integration.md)。

目前登记模型：

- `mage-vl-4b`：Mage-VL-4B 多模态服务，使用 Transformers；支援图片聊天、文字 SSE、影片上传及 Live 工作阶段。
- `qwen3-0.6b`：Qwen3-0.6B 纯文字服务，使用模型包内的 PyTorch 计算图、批次 prefill/decode 和实体 KV page pool。

两者共用 API、Engine Core、Scheduler、Executor 与 Worker 装配；模型 Runtime、Processor、模型拓扑和权重载入各自留在模型包中。此架构参考 vLLM／nano-vLLM 的职责拆分，但没有 vLLM backend、自订 PagedAttention kernel、prefix cache 或其他未登记模型。

```bash
python src/inference/launcher.py list
python src/inference/launcher.py validate
python src/inference/launcher.py start --model mage-vl-4b
python src/inference/launcher.py start --model qwen3-0.6b
```

推论 API 预设位址由各模型 `configs/network.yml` 决定：Mage-VL 为 `127.0.0.1:8001`，Qwen3 为 `127.0.0.1:8011`。模型参数放在对应 `configs/inference.yml`；模型 YAML 指向的依赖清单位于模型根目录。启动与服务细节见 [`src/inference/README.md`](src/inference/README.md)。

## API 与测试

- `GET /health`、`GET /v1/models`
- `POST /v1/chat/completions`：OpenAI 相容文字／图片聊天；支援模型能力宣告的文字 SSE。
- `POST /v1/videos/understand`：Mage-VL 影片上传理解。
- `/v1/live/*`：Mage-VL 即时影片工作阶段 API；RTSP／RTMP 媒体流由 MediaMTX sidecar 处理。

Gateway 预设监听 `127.0.0.1:8000`。测试按模型整理于 `tests/qwen3_0_6b/` 和 `tests/mage_vl_4b/`，每个模型包含计算图对照、推理性能比较和 Gateway 端到端三项测试；细节见 [`tests/README.md`](tests/README.md)。`checkpoints/val.md` 和 `checkpoints/val.xlsx` 是受保护的既有验收记录。
