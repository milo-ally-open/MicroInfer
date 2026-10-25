# Go API Gateway

Gateway 是独立的 Go HTTP 服务，负责客户端认证、租户限流、模型路由、请求政策、上游转发和观测。模型加载与推理由独立的 Inference 服务负责；RTSP/RTMP 媒体流由客户端直接连接 MediaMTX。

## 当前能力

- API Key Bearer 认证和基于租户的进程内 token bucket 限流。
- `/v1/models`、`/v1/chat/completions`、视频上传理解及 Live session 控制 API。
- 按模型能力拒绝不支持的输入和路由；当前模型与映射见 [`configs/gateway.yml`](configs/gateway.yml)。
- 有界转发聊天 SSE 和视频 multipart 请求；聊天回应中的模型字段改为公开别名。
- Qwen3 标准 function tool 请求和回应透传。Gateway 不解析模型标记，也不执行工具。
- `/healthz` 检查 Gateway 存活；`/readyz` 检查配置中的全部 Inference 上游；`/metrics` 提供 Prometheus 格式指标。
- Live session handle 使用 HMAC 签名并绑定租户和模型路由。发布端 RTSP/RTMP 地址仍由客户端直连 MediaMTX。

## 模型能力

| 对外别名 | 上游 Inference | 能力 |
| --- | --- | --- |
| `mage-vl-4b` | `http://127.0.0.1:8001` | 图片聊天、文本 SSE、视频上传、Live |
| `qwen3-0.6b` | `http://127.0.0.1:8011` | 文本聊天、文本 SSE、function tool calling |

模型上游地址、公开 URL、请求限制、逾时、连接池、限流、指标和 listener 配置位于 `configs/gateway.yml`。部署前应核对这些值与对应 Inference 模型的 YAML 限制相容。

## 启动

需要可用的 Go 工具链。从仓库根目录进入 Gateway 模块：

```bash
cd src/gateway
go run ./cmd/gateway --config configs/gateway.yml
```

Gateway 默认监听 `127.0.0.1:8000`。配置包含 Mage-VL 和 Qwen3 两个上游；若要让 `/readyz` 报告就绪，所有已配置的 Inference 上游都必须运行并通过 `GET /health` 检查。Inference 的安装和启动方式见 [Inference README](../inference/README.md)。

检查存活和就绪状态：

```bash
curl http://127.0.0.1:8000/healthz
curl http://127.0.0.1:8000/readyz
curl http://127.0.0.1:8000/metrics
```

## 认证密钥

通过仓库外的 JSON 文件配置 API Key，并以环境变量提供文件路径。示例文件内容如下：

```json
[
  {"id": "local-client", "tenant_id": "demo", "key": "<至少 16 字符的随机密钥>"}
]
```

Bash：

```bash
export GATEWAY_API_KEYS_FILE=/secure/path/gateway-api-keys.json
export GATEWAY_SESSION_HMAC_KEYS='<base64url-key>[,<previous-base64url-key>]'
```

PowerShell：

```powershell
$env:GATEWAY_API_KEYS_FILE = 'C:\secrets\gateway-api-keys.json'
$env:GATEWAY_SESSION_HMAC_KEYS = '<base64url-key>[,<previous-base64url-key>]'
```

每个 HMAC key 解码后至少 32 bytes。第一个 key 用于签署新句柄，后续 key 用于验证轮换前的句柄；旧 key 至少保留到对应句柄过期。默认句柄期限为 3660 秒，应与 Inference Live session 时长保持相容。不要将密钥写入 YAML、示例文件或 Git。

## API 行为与边界

- 聊天 JSON 的 `model` 使用 Gateway 公开别名。视频上传可用 `?model=<alias>` 选择模型；Live 建立请求可传 `model`。默认视频和 Live 模型由 Gateway YAML 指定。
- 聊天 SSE 逐事件转发；视频 multipart 采用有界读取，不将整个请求文件载入 Gateway 内存。
- `/readyz` 查询所有已配置 Inference 上游；任一上游未就绪时返回非就绪状态。
- 限流状态保存在单个 Gateway 进程内，重启后重置。当前实现不提供多实例共享限流状态。
- `/metrics` 不使用原始 API Key 或租户 ID 作为标签。指标包括请求、错误类别、延迟和流式请求观测。
- TLS 可由受控的前置反向代理终止；此时应将 `public_base_url` 设置为客户端实际访问的 HTTPS 地址。
- Gateway 不承载 RTSP/RTMP 封包，不执行工具，也不负责模型权重或 GPU 生命周期。

## 测试与验收

逐模型端到端测试见 [Tests README](../../tests/README.md)，测试脚本位于 `tests/<model_alias>/test_end2end.py`。运行前需启动所需 Gateway 与 Inference 服务、设置 `VIDEO_API_KEY`，并准备测试 YAML 指定的素材。测试结果输出到终端；需留档时，保存到对应模型的 `results/` 目录。

功能 smoke 不代表高并发容量测试或 SLA。Live 的正式容量、长期媒体稳定性和多实例部署尚未作为本 README 所述的既有验收结论。
