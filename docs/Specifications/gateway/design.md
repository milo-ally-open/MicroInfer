# 技术设计：Go API 网关

> 关联规格：[spec.md](spec.md)<br>
> 关联提案：[proposal.md](proposal.md)<br>
> 任务追踪：[tasks.md](tasks.md)<br>
> 状态：Go Gateway v1 已实现；双模型能力路由、Qwen 标准 tool capability 及仓库无关 module path 依 REQ-13 至 REQ-15 增补；验收进度见 tasks.md。

## 当前实现与目标

- 当前代码位置／行为：src/gateway/ 已有 Go module、HTTP handler、配置 YAML、认证／配置／限流／观测包及 server lifecycle；实际文件见 src/gateway/README.md。
- 现有推理服务：src/inference/network/service.py 依模型登记组装 /health、/v1/models、/v1/chat/completions，以及模型已启用的视频／Live 路由。
- 现有模型路由：`mage-vl-4b` → `microsoft/Mage-VL`（port 8001），`qwen3-0.6b` → `qwen3-0.6b`（port 8011）；各模型能力在 Inference Registry 与 Gateway YAML 对应登记。
- 目标边界：Gateway 处理外部 HTTP/API Key/租户政策/限流/路由/转发/可观测性；推理服务处理模型与推理；MediaMTX 处理 RTSP／RTMP 媒体流。

## Gateway 代码模块

| 模块 | 责任 |
|---|---|
| `gateway.go` | 组装应用组件、执行认证与限流政策、实现聊天／实时工作会话业务流程及 OpenAI 风格错误 |
| `internal/route/` | 将公开 API 路径和 HTTP method 分派至应用 handler，提供稳定的 route label |
| `internal/upstream/` | 创建共用 HTTP transport，转发请求正文、`Accept`、`Last-Event-ID` 和 request ID，返回上游延迟 |
| `internal/stream/` | 有界读取 SSE 事件、改写聊天模型别名、逐事件 flush、追踪首 token／闲置与取消 |
| `internal/videoproxy/` | 验证 multipart `Content-Length`，以流式方式转发 request body，观测请求正文读取完成时间 |

上述包只处理 Gateway 的 HTTP 边界，不加载模型或代理 RTSP／RTMP 媒体数据包。

Gateway Go module 使用稳定的中性 module path `example.com/videostreamapi/gateway`，内部 import 均以该路径为前缀。模块路径不从 Git remote、代码托管账号或仓库名称生成；迁移仓库时保持不变。该路径只属于 Go 编译／包识别，不构成公开 HTTP API、模型 alias 或部署端点。

## 组件与数据流

1. 客户端对 Gateway 发送 HTTP API 请求并提供 Bearer API Key。
2. Gateway 验证 API Key，取得服务器端 tenant_id，再执行模型、请求大小、限流及超时政策。
3. Gateway 依 src/gateway/configs/gateway.yml 将 API 请求送至推理服务；推理地址不可散落为代码常数。
4. 一般 JSON 响应由 Gateway 返回；聊天 SSE 逐事件转发并刷新，视频 multipart 以有界流式转发。
5. 实时工作会话控制 API 经 Gateway 转发。Gateway 把推理服务的内部 session_id 包装为租户绑定的签名句柄，并将事件／停止 URL 改写为 Gateway URL。
6. 客户端使用推理服务返回的 RTSP／RTMP publish URL 直接连接 MediaMTX。MediaMTX 的授权回调仍直接连接推理服务；Gateway 不承载媒体数据包，也不更改回调拓扑。

## 对外 HTTP 契约

### 系统端点

| 方法与路径 | 认证 | 行为 |
|---|---|---|
| GET /healthz | 不需要 | Gateway 程序存活；不检查模型是否就绪 |
| GET /readyz | 不需要 | 每个不同的已配置推理 URL 之 GET /health 都回报 `status=ok` 才成功；否则回 503 |

### 客户端 API

| 方法与路径 | 认证 | 上游路径 | 设计要点 |
|---|---|---|---|
| GET /v1/models | Bearer API Key | GET /v1/models | 对外只列出配置的项目别名 |
| POST /v1/chat/completions | Bearer API Key | POST /v1/chat/completions | 以公开 model alias 路由；按 text/image/stream capability 验证后改写 upstream model ID |
| POST /v1/videos/understand?model=`{model_alias}` | Bearer API Key | POST /v1/videos/understand | 选用 query 选模型，省略则使用 YAML 默认；保留 multipart 原文流式 |
| POST /v1/live/sessions | Bearer API Key | POST /v1/live/sessions | JSON 选用 `model`；改写工作会话控制标识符及 URL；publish URL 仍指向 MediaMTX |
| GET /v1/live/sessions/{session_id}/events | Bearer API Key | GET /v1/live/sessions/{upstream_session_id}/events | 验证租户句柄；保留 Last-Event-ID 与 SSE 语义 |
| DELETE /v1/live/sessions/{session_id} | Bearer API Key | DELETE /v1/live/sessions/{upstream_session_id} | 验证租户句柄后停止上游工作会话 |

/v1/live/mediamtx/auth 是推理服务提供给 MediaMTX 的内部回调，不作为 Gateway 客户端路由。

## 配置设计

Gateway 非敏感配置位于 src/gateway/configs/gateway.yml。配置至少涵盖：

| 类别 | 配置责任 |
|---|---|
| listener | host、port；默认 127.0.0.1:8000 |
| upstream | 共用推理服务 fallback URL；本地默认值 http://127.0.0.1:8001 |
| defaults | 视频及 Live 的默认模型别名 |
| model routes | 每模型公开别名、上游模型 ID、上游 URL 和 text/image/stream/tool/video/live 能力；目前登记 Mage-VL 与 Qwen3-0.6B |
| request limits | 聊天、视频 multipart、实时工作会话请求正文上限 |
| timeouts | 上游连接、一般响应、SSE 闲置及整体超时 |
| tenant policies | 每租户请求速率、突发容量及必要的并行限制 |
| observability | 日志等级、request ID、指标输出配置 |
| secret references | API Key、签名密钥的环境变量名称或挂载文件路径；不可包含秘密值 |
| transport／buffers | 上游连接池、keepalive、Expect-Continue 及请求标头／流式 buffer 上限 |

所有可调整政策由 YAML 提供；必要字段缺失或范围无效时，Gateway 启动失败。Gateway 的视频请求正文上限不可高于推理端 inference.yml 的 max_video_request_bytes；聊天及实时工作会话亦依其上游请求正文限制配置。推理配置仍由推理服务自己的 YAML 管理，不搬入 Gateway。

## 认证与租户

- Authorization: Bearer 提供 API Key。Gateway 从安全注入的来源加载有效密钥及其 tenant_id 对应。
- 客户端提交的租户标头不具权限来源；Gateway 不采信其覆盖已验证身份。
- 比较密钥时采用适当的常数时间比较；日志只记录安全的租户代号或 Key ID，不记录原始 API Key。
- /healthz 与 /readyz 不要求客户端 API Key；所有 /v1/* 客户端 API 都要求有效密钥。
- 首版只支持单实例；租户限流器位于程序内存，重启后计数重置。多实例共享状态不在首版范围。

## 模型路由与响应改写

- Gateway 启动时加载每模型路由。聊天请求以 `model` 选公开别名，未配置或未知 alias 在调用上游前以 HTTP 400 拒绝；依 capability 拒绝纯文字 Mage 请求及含图片的 Qwen 请求，两者都支持文字 SSE。
- Qwen route另有 `tool_calling=true`；标准 Chat tools 请求由 Gateway透传到 Inference，不解读模型参数或执行工具。调用方透过同一个 `/v1/chat/completions` endpoint 发送 tool result message。
- 聊天上游依路由使用 `microsoft/Mage-VL` 或 `qwen3-0.6b`，响应 JSON／SSE 的 model 字段改回该请求的项目 alias。
- GET /v1/models 仅返回公开别名，不暴露上游 repo_id。
- SSE parser 以事件为单位处理 data 字段，只改写合法 JSON 的 model 字段；注解、keep-alive、[DONE] 及非 JSON 控制数据保持不变。每个事件实时刷新，不聚合整条流式。

视频路由使用可选 query `model`，以便不读取或重建 multipart 请求正文也能选择上游；默认模型由 `defaults.video_model` 配置。Live 创建 JSON 使用可选 `model`（默认 `defaults.live_model`）。Gateway 在转发前检查 `video_upload`／`live_video`，不支持时拒绝。Inference route capability 不在 Gateway 内推导；模型端到端测试验证 Gateway 公开 alias 与实际请求路径，部署端点和 capability 需人工维护同步。

不新增 Agent Chat route 或额外上游。工具调用与普通聊天共用 `/v1/chat/completions` 的认证、租户限流、alias 路由、请求上限和 SSE 转发；工具执行由客户端／Agent 应用自行负责。

## 视频请求正文与 Content-Length

- 目前推理 API 在读取聊天 JSON、视频 multipart、实时工作会话 POST 请求正文前要求 Content-Length；缺少时回 HTTP 411。
- Gateway 在转发前检查 Content-Length 格式及 YAML 上限；长度无效回 400，请求正文过大回 413。
- Gateway 以有界流式方式转发 multipart，不把完整视频写入内存，也不改写 multipart boundary。
- Gateway 对流式位元组数另设上限，避免只依赖宣告长度；到达上限时中止上游请求并回报请求过大。
- 推理端当前上限是配置基线而非 Gateway 常数：聊天请求正文 15,728,640 bytes、视频请求正文 210,763,776 bytes、视频文件 209,715,200 bytes、实时创建请求正文 65,536 bytes。部署时应让 Gateway 上限不高于相应推理上限；推理服务仍负责检查视频文件本身大小。

## SSE、超时与错误

- SSE 响应配置适当的 text/event-stream、Cache-Control: no-cache，关闭代理缓冲并逐事件 flush。
- Gateway 保留上游事件顺序及必要流式标头；客户端取消时取消上游 HTTP 请求。遵守配置的闲置与整体超时。
- 在 SSE 标头送出前发生连接错误或超时，分别回 HTTP 502 或 504 的 OpenAI 风格 JSON 错误；流式开始后发生错误则记录 request ID 并结束流式。
- Gateway 自行产生的错误至少使用 error.message、error.type、error.code；HTTP 状态依认证、请求、限流及上游失败种类配置。上游错误保留有意义的 HTTP 状态并正规化错误请求正文，不返回内部堆叠或内部网络地址。
- 限流回 HTTP 429 并提供 Retry-After；请求正文过大回 413；缺少 Content-Length 回 411。

## 实时工作会话句柄

句柄采 URL 安全编码，签章涵盖版本、tenant_id、上游 session_id、创建时的模型 alias 与到期时间。Gateway 不需要数据库或程序内的租户工作会话表即可验证句柄并还原上游路由；旧版未含 alias 的 v1 句柄使用 `defaults.live_model` 兼容处理。

- 创建工作会话后，Gateway 将对外 id 改写为签名句柄；events_url 与 stop_url 使用 Gateway 主机及公开句柄。
- Gateway 验证 Bearer Key、句柄签章、到期时间及 tenant_id 相符后，才将句柄中的上游 session_id 转为推理服务路径。
- 无效、过期或其他租户的句柄一致回 HTTP 404，避免暴露工作会话存在性。
- RTSP／RTMP publish URL 保留推理服务生成的 MediaMTX 路径。该 URL 是能力凭证，取得者可在工作会话有效期间使用 MediaMTX 对应路径；工作会话停止或自然结束后失效。
- 密钥轮替时保留旧验证密钥，至少涵盖旧句柄最长有效期。目前 inference.yml 的实时工作会话最长时间为 3600 秒；部署值改变时须同步签名句柄 TTL。

## 可观测性、容量与敏感数据

- 每个请求带有或产生 request ID，日志记录路由、HTTP 状态、租户安全代号、上游耗时、总耗时及可用的请求正文位元组数。
- 指标可观测吞吐、错误率、延迟分位数、SSE 首 token 延迟与流式时间、视频请求完成时间及上游连接／超时错误。
- 不记录 API Key、签名密钥、聊天／图片／视频请求正文、生成内容或完整 RTSP／RTMP 能力 URL。
- 指标不得以原始 API Key 为维度；租户标签应受控以避免高基数及敏感信息暴露。

## 兼容性、回复与测试

- 推理 API 保持可独立启动与测试；Gateway 只使用配置的 HTTP 上游，不加载推理包或模型。
- 目前 MediaMTX 授权回调维持直接连接推理 API，避免把内部回调错误暴露为客户端路由。
- 回复只需停止 Gateway listener、撤下外部入口并还原 Gateway YAML；推理服务与 MediaMTX 不需反向修改。切换前应停止活动中的实时工作会话。若程序意外停止，推理服务依自身 max_duration_seconds 结束工作会话。
- 逐模型维护 `tests/{model_alias}/test_components.py`、`test_performance.py`、`test_end2end.py` 三项验证；性能与端到端测试须在 Gateway、推理 API 和模型所需 MediaMTX 可用时另行实测。三项测试不做精度评估。
- 首版性能基线记录吞吐量、错误率、p50/p95/p99、SSE 首 token 延迟／流式时间及视频上传完成时间，不设硬性 SLA，不做模型精度评测。

## 取舍

| 方案 | 优点 | 放弃原因／代价 |
|---|---|---|
| 单实例、程序内限流、租户签名句柄 | 无需外部状态服务；句柄验证可跨 Gateway 重启 | 多实例需新增共享限流方案 |
| API Key 对应租户并以 HMAC 句柄授权控制 API | SSE／停止操作具租户隔离且不需数据库 | 媒体 URL 本身仍是能力凭证，须由客户端保密 |
| Gateway 代理 RTSP／RTMP 媒体 | 集中网络入口 | 增加长连接与频宽压力，且违反 MediaMTX sidecar 边界，故不采用 |
| 透过推理 HTTP API 转发 SSE 与视频 | 保持推理与网关服务分离 | 需正确处理刷新、取消、Content-Length 及流式上限 |

## 未决项

无待确认的产品范围项目。部署者需提供实际限流值、请求正文限制、超时、上游 URL、密钥注入来源及 TLS 前置代理配置。

## 变更记录

| 日期 | 变更 |
|---|---|
| 2026-10-20 | 创建 Gateway 技术设计草稿 |
| 2026-10-25 | 增加 Qwen tool capability；按 OpenAI 兼容模式使用现有 Chat route，工具由调用端执行 |
| 2026-10-25 | 将 Go module 与内部 imports 改为仓库无关的稳定命名空间，避免编译及诊断信息绑定代码托管地址 |

