# Go API 闸道

Go Gateway v1 已实作，负责外部客户端 API Key、租户限流、模型别名、HTTP 请求政策、上游错误转换、SSE／视讯代理与请求观测。推论服务仍独立管理模型权重、GPU、生成及视讯前处理；RTSP／RTMP 媒体封包由客户端直接传至 MediaMTX。

## 目前能力

- Gateway 目前登录 `mage-vl-4b`（上游 `microsoft/Mage-VL`）及 `qwen3-0.6b`（上游 `qwen3-0.6b`），分别连接各自的推论服务。
- 能力依模型路由设定：Mage-VL 接受图片聊天、文字 SSE、影片上传和 Live；Qwen3-0.6B 接受纯文字聊天、文字 SSE 和标准 function tool calling。Gateway 会在上游转送前拒绝不支援的输入／路由。
- `GET /healthz` 仅检查 Gateway 存活；`GET /readyz` 检查每个已配置 Inference 服务 `GET /health` 的 `status=ok`。
- 需要 Bearer API Key 的客户端 API：`GET /v1/models`、`POST /v1/chat/completions`、`POST /v1/videos/understand`、`POST /v1/live/sessions`、live events SSE 与 `DELETE` 停止路由。
- 聊天 SSE 逐事件转送，并将分块中的上游 model 栏位改写为专案别名。视讯 multipart 本文以有界串流送往推论服务，不整段载入 Gateway 记忆体。
- `/v1/chat/completions` 将 Qwen3 `tools` 请求转送给 Inference，回传标准 `tool_calls`／`delta.tool_calls`，但不执行工具。客户端／Agent 应用负责校验并执行获准工具，再把 `role: "tool"` 结果送回同一个接口。
- 聊天 JSON 的 `model` 使用公开别名；影片上传可用 `?model=<alias>` 选择已登记模型，预设由 `defaults.video_model` 指定。Live 建立本文可带 `model`，预设使用 `defaults.live_model`；签名 session handle 绑定建立时的模型路由。
- Live 工作阶段控制 ID 是绑定租户、建立时模型别名、上游 session ID 和有效期的 HMAC 句柄。`publish.rtsp`／`publish.rtmp` 维持直连 MediaMTX；Gateway 不代理媒体封包。
- OpenAI 风格错误、每租户程序内 token bucket、JSON 结构化日志，以及不含 API Key／租户标签的 Prometheus 格式 `/metrics`。

## 目录

```text
src/gateway/
├── cmd/gateway/main.go            # 设定、密钥载入与 HTTP server lifecycle
├── configs/gateway.yml            # 非敏感 Gateway 设定
├── gateway.go                     # 应用组装、认证政策、API 业务流程及错误回应
└── internal/
    ├── auth/                      # API Key、HMAC live 句柄
    ├── config/                    # 严格 YAML 载入与设定验证
    ├── middleware/                # 单实例租户 token bucket
    ├── observability/             # 请求与延迟 metrics
    ├── route/                     # API 路径与 HTTP method 分派
    ├── stream/                    # 聊天 SSE 改写、限界解析与即时转送
    ├── upstream/                  # 推论上游 HTTP transport 与请求建构
    └── videoproxy/                # multipart 请求长度检查及串流代理
```

`gateway.go` 建立并组合以上元件；路由表、上游 transport、SSE 转送及影片 multipart 代理分别由对应套件负责。Gateway 不解析模型 tool marker 或执行工具，只负责外部 API Key／租户政策、能力校验、转送及 live 句柄签章。

## 设定与密钥

所有服务位址、公开 URL、模型映射／能力、本文限制、逾时、连线池／串流 buffer、限流、指标和 listener 设定都在 [`configs/gateway.yml`](configs/gateway.yml)。预设 Gateway 绑定 `127.0.0.1:8000`，Mage-VL 上游预设 `http://127.0.0.1:8001`，Qwen3-0.6B 上游预设 `http://127.0.0.1:8011`。Gateway 限制不得高于推论端自己的 YAML 上限；推论端的 `Content-Length` 规则仍由推论服务负责。

API Key 以 JSON array 放在 Git 工作树以外的受保护档案，透过 `GATEWAY_API_KEYS_FILE` 指向该档。格式如下，请替换占位值并使用随机密钥：

```json
[
  {"id": "local-client", "tenant_id": "demo", "key": "<至少 16 字元的随机 API Key>"}
]
```

`GATEWAY_SESSION_HMAC_KEYS` 接受以逗号分隔的 base64url 密钥，解码后每把至少 32 bytes。第一把用于新句柄签章，后续密钥只用于验证旧句柄。轮替时，保留旧密钥至少至所有旧句柄过期；`handle_ttl_seconds` 预设 3660 秒，需与推论端 live session 最长期限保持相容。不得将 API Key 或 HMAC 密钥写进追踪中的 YAML／JSON。

## 启动

按顺序启动 Qwen3 Inference，再启动 Gateway。Gateway 可以在 Inference 尚未就绪时启动，但 `/readyz` 会回 `503`。

PowerShell 范例：

```powershell
$env:GATEWAY_API_KEYS_FILE = 'C:\secrets\gateway-api-keys.json'
$env:GATEWAY_SESSION_HMAC_KEYS = '<base64url-secret>[,<previous-base64url-secret>]'
Set-Location src/gateway
go run ./cmd/gateway --config configs/gateway.yml
```

Qwen3 工具调用经现有 `/v1/chat/completions` 转送至 `http://127.0.0.1:8011`；不需要额外 Agent endpoint 或内部共享 token。客户端工具循环示例见 [`tests/client.py`](../../tests/client.py)。

从专案根目录呼叫：

```powershell
curl.exe http://127.0.0.1:8000/healthz
curl.exe http://127.0.0.1:8000/readyz
```

`/readyz` 会检查每个不同的已配置 Inference 上游；只有全部回报 ready 才回成功。`/metrics` 预设启用，提供请求量、错误类别、延迟 histogram、SSE 首文字或工具调用增量／串流时间及视讯本文读取完成时间。Prometheus 可用 histogram buckets 计算 p50、p95、p99；metrics 不使用原始 API Key 或租户 ID 作为标签。限流状态保存在单一程序记忆体，重启后重设；首版不支援多实例共享限流。

## 测试与端到端验收

Gateway 和推论服务启动方式维持独立。逐模型的三项测试入口、依赖条件与命令见 [`tests/README.md`](../../tests/README.md)；每个测试都必须用 `--config` 指向模型专属 `test.yml`。Gateway 端到端测试位于 `tests/qwen3_0_6b/test_end2end.py` 和 `tests/mage_vl_4b/test_end2end.py`，前者验证文字 SSE，后者验证图片 SSE 和影片上传。

2026-10-22 已以低显存设定连接 Qwen3-0.6B 执行文字 JSON／SSE、双请求并发与客户端断流回收 smoke；同日也透过真实 Mage-VL 完成图片聊天 JSON／SSE、视讯上传与 Live／MediaMTX 合成 RTSP smoke。当前逐模型测试脚本将 JSON 摘要输出到标准输出，不会自动创建结果文件；需要归档时由执行者保存到对应模型目录的 results/。Mage-VL Gateway 验收记录见 [`analysis/mage_vl_4b/analysis.md`](../../analysis/mage_vl_4b/analysis.md)。Live 仅有单次功能验证；正式容量、高并发和媒体授权边界仍未测。

目前仓库不再包含根目录的 example.py 或 test.jpg 范例。客户端可直接使用 OpenAI 兼容的聊天 API；API Key 由 VIDEO_API_KEY 环境变量提供。三项逐模型验收脚本及其设定方式见 [Tests README](../../tests/README.md)。

Inference 性能比较位于各模型的 `test_performance.py`，Gateway 端到端功能验收则位于 `test_end2end.py`；两者使用不同路径，分别量测推理 API 与 Gateway 完整链路。三项测试不做模型精度评测；既有 smoke 样本数不足以作为稳定容量基线或 SLA。

## 部署与回复

初始部署保持推论 API 绑定 loopback，只让受控前置入口连接 Gateway。TLS 可由前置 Proxy 终止，并将 `public_base_url` 设为客户端使用的 HTTPS URL。回复时先停止外部入口并停止作用中的 live session，再停止 Gateway 或回复 `gateway.yml`；推论服务和 MediaMTX 不需随 Gateway 一起回滚。
