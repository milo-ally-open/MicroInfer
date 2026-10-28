# 行为规格：Go API 网关

> 关联提案：[proposal.md](proposal.md)<br>
> 技术设计：[design.md](design.md)<br>
> 实施追踪：[tasks.md](tasks.md)<br>
> 状态：Go Gateway v1 已实现；REQ-13 记录双模型能力路由，REQ-14 将 Qwen tool capability 纳入既有 Chat Completions route，REQ-15 约束 Gateway module path 与代码仓库无关；验收状态见 tasks.md。
> 来源：使用者确认的 Go 网关目标、根目录 AGENTS.md、README.md、src/gateway/README.md 与 src/inference/README.md。

## 范围和术语

本规格涵盖外部客户端至 Go Gateway、Gateway 至推理 API 的 HTTP 请求，以及实时工作会话对 MediaMTX 的媒体数据流。Gateway 与推理服务是分离组件。媒体数据流由客户端直接连接 MediaMTX。

- 项目别名：`mage-vl-4b` 和 `qwen3-0.6b`，供客户端使用。
- 上游模型 ID：`microsoft/Mage-VL` 与 `qwen3-0.6b`，分别用于各模型推理服务路由。
- 租户：由已验证 API Key 对应的客户或调用方。
- 工作会话句柄：Gateway 签发、绑定租户与上游工作会话的 URL 安全标识符。
- OpenAI 风格错误：JSON 请求正文包含 error 物件，至少含 message、type、code；param 可为 null。

## 需求与验收条件

### REQ-01：公开模型别名与上游路由

**需求**：Gateway 应以 YAML 登记公开模型别名及其上游服务映射，只公开已登记的模型别名。

**验收条件**

- **AC-01.1** Given API Key 有效，When 客户端调用 GET /v1/models，Then 响应清单包含目前登记的 mage-vl-4b 和 qwen3-0.6b，且不暴露上游标识符。
- **AC-01.2** Given 客户端请求指定 model 为 mage-vl-4b，When Gateway 转发聊天请求，Then 上游请求使用 microsoft/Mage-VL，非流式响应及每个 SSE JSON 分块的 model 字段均使用 mage-vl-4b。
- **AC-01.3** Given 请求未指定 model 或指定未登记的名称，When Gateway 验证聊天请求，Then 响应 HTTP 400 与 OpenAI 风格错误，且不调用上游。

**验证方式**：Gateway API 整合测试；验证请求与响应的模型字段。

### REQ-02：API Key 验证与租户识别

**需求**：除健康及就绪端点外，所有公开 /v1/* API 都应以 Bearer API Key 验证调用者，并以服务器端对应结果决定租户。

**验收条件**

- **AC-02.1** Given /v1/* 请求带有有效 Bearer API Key，When Gateway 验证请求，Then 使用该密钥配置所对应的租户执行政策判定。
- **AC-02.2** Given API Key 缺漏或无效，When 调用任一公开 /v1/* API，Then 响应 HTTP 401 与 OpenAI 风格错误，且不调用上游。
- **AC-02.3** Given 请求另带客户端自定义租户标头，When Gateway 创建调用者身份，Then 不以客户端标头覆盖服务器端租户对应。

**验证方式**：认证中介层测试及 API 整合测试。

### REQ-03：一般聊天请求

**需求**：Gateway 应依选定模型已登记的输入能力，将文字或图片／图文聊天请求转发至正确的推理服务，并保留兼容的聊天字段。Mage-VL 要求图片输入；Qwen3-0.6B 接受纯文字，两者都不接受对方未宣告的输入模态。

**验收条件**

- **AC-03.1** Given 聊天请求通过 API Key、模型及请求正文限制检查，When stream 为 false 或未启用，Then Gateway 转发请求并返回推理服务的 JSON 完成结果，仅改写公开 model 字段及 Gateway 规定的错误信息。
- **AC-03.2** Given 上游响应包含非 hop-by-hop 的内容型别及必要标头，When Gateway 返回结果，Then 保留与客户端契约兼容的标头及状态码，且不泄漏上游内部地址或服务器堆叠信息。

**验证方式**：Gateway API 整合测试，涵盖文字及图片请求；不执行模型精度评估。

### REQ-04：聊天 SSE 流式

**需求**：Gateway 应转发聊天 SSE token 流式，保持事件边界、顺序、实时送出、取消及完成标记语义。

**验收条件**

- **AC-04.1** Given 聊天请求指定 stream 为 true，When 上游产生 SSE，Then Gateway 逐事件转发，刷新响应而不等待完整生成，并保留 data: [DONE] 结尾。
- **AC-04.2** Given 上游 SSE JSON 分块包含 model 字段，When Gateway 对外转发该分块，Then 将字段改写为请求选定的公开别名；SSE 注解、keep-alive、非 JSON 控制数据及 [DONE] 不被误解析为完成分块。
- **AC-04.3** Given 客户端在流式中断线或取消，When Gateway 收到取消讯号，Then 取消上游请求并释放相关连接与工作资源。

**验证方式**：流式测试，检查首事件延迟、事件顺序、反向压力及客户端取消。

### REQ-05：视频上传转发

**需求**：Gateway 应将视频理解 multipart 请求转发至推理服务，不在 Gateway 将完整文件读入内存。

**验收条件**

- **AC-05.1** Given multipart 请求包含 video、question 及目前推理 API 支持的选用字段，When Gateway 转发请求，Then 保留 multipart 字段与边界语义并以流式方式送至 POST /v1/videos/understand。
- **AC-05.2** Given 请求缺少 Content-Length、长度格式错误或超过 Gateway 配置上限，When Gateway 在读取大型请求正文前检查请求，Then 分别响应 HTTP 411、HTTP 400 或 HTTP 413，使用 OpenAI 风格错误且不调用上游。
- **AC-05.3** Given Gateway 与推理服务的视频请求正文上限已配置，When 部署配置被验证，Then Gateway 对应上限不得高于推理端 inference.yml 的限制；Gateway 另以有界流式计数避免错报长度造成无界转发。

**验证方式**：multipart API 整合测试及大请求正文流式测试；不下载模型权重。

### REQ-06：实时工作会话与租户隔离

**需求**：Gateway 应代理实时工作会话控制 API，并确保只有创建工作会话的租户可读取事件或停止该工作会话。

**验收条件**

- **AC-06.1** Given 有效租户创建工作会话，When Gateway 调用推理服务 POST /v1/live/sessions，Then 对外响应使用该租户绑定的 URL 安全签名句柄，events_url 与 stop_url 指向 Gateway，publish RTSP／RTMP URL 仍直连 MediaMTX。
- **AC-06.2** Given 同租户使用有效 API Key 与签名句柄调用事件或停止路由，When Gateway 验证句柄签章、租户及有效期，Then 以句柄内的模型别名和上游标识符代理 GET /v1/live/sessions/{session_id}/events 或 DELETE /v1/live/sessions/{session_id}。
- **AC-06.3** Given 句柄无效、过期或属于其他租户，When 客户端调用事件或停止路由，Then 响应一致的 HTTP 404 OpenAI 风格错误，不揭露工作会话是否存在。
- **AC-06.4** Given 客户端透过 publish URL 连接 MediaMTX，When MediaMTX 授权回调检查路径，Then 仅活动中的工作会话可发布或读取；媒体 URL 视同能力凭证，客户端应保密，且工作会话终止后失效。

**验证方式**：租户隔离测试、签章与有效期测试、SSE Last-Event-ID 续接测试，以及受控 MediaMTX 整合测试。

### REQ-07：请求大小、长度与租户限流

**需求**：Gateway 应在转发前应用 YAML 配置的请求正文限制及每租户限流，单实例首版的限流状态由程序内维护。

**验收条件**

- **AC-07.1** Given POST 聊天、视频上传或工作会话创建请求未带 Content-Length，When Gateway 验证请求，Then 响应 HTTP 411；格式错误长度响应 HTTP 400；超出配置上限响应 HTTP 413。
- **AC-07.2** Given 租户超出 YAML 配置的请求速率或突发容量，When 该租户再发送请求，Then Gateway 响应 HTTP 429 与 Retry-After，且其他租户不受该租户计数器直接影响。
- **AC-07.3** Given Gateway 程序重新启动，When 新程序开始接受流量，Then 程序内限流状态重新初始化；此行为须在运维文件中明确记载。

**验证方式**：请求正文边界、Content-Length、租户隔离及限流测试。

### REQ-08：错误格式与上游失败

**需求**：Gateway 自身及正规化后的上游错误应使用一致的 OpenAI 风格 JSON，且不得泄漏密钥、内部地址或堆叠。

**验收条件**

- **AC-08.1** Given Gateway 拒绝认证、模型、请求正文、限流或工作会话请求，When 响应错误，Then JSON error 物件包含 message、type、code，并搭配适当 HTTP 状态码。
- **AC-08.2** Given 上游返回 HTTP 错误，When Gateway 响应客户端，Then 保留有意义的 HTTP 状态并将错误请求正文正规化；连接失败回 HTTP 502，超时回 HTTP 504。
- **AC-08.3** Given SSE 标头已送出后发生上游错误，When Gateway 无法再更改 HTTP 状态，Then 终止流式并记录含 request ID 的错误，不把一般 JSON 错误插入 SSE token 数据。

**验证方式**：错误契约测试、上游模拟器测试及流式中断测试。

### REQ-09：健康与就绪

**需求**：Gateway 应分别提供程序存活检查与上游就绪检查。

**验收条件**

- **AC-09.1** Given Gateway 程序运行，When 调用 GET /healthz，Then 响应成功且不要求 API Key；此端点只表示 Gateway 程序仍存活。
- **AC-09.2** Given Gateway 配置有效且每个不同推理 URL 的 GET /health 均回报模型状态为 ok，When 调用 GET /readyz，Then 响应成功且不要求 API Key。
- **AC-09.3** Given 配置未加载或推理上游不可用／未就绪，When 调用 GET /readyz，Then 响应 HTTP 503，不宣称模型 API 可服务。

**验证方式**：健康端点测试及上游健康状态模拟测试。

### REQ-10：可观测性与敏感数据保护

**需求**：Gateway 应提供可依请求追踪的结构化日志与服务指标，且不记录敏感请求正文或凭证。

**验收条件**

- **AC-10.1** Given 每个 API 请求进入 Gateway，When 请求完成或失败，Then 日志可关联 request ID、路由、租户标识符或安全的租户代号、HTTP 状态、上游延迟与总延迟。
- **AC-10.2** Given 任何认证、聊天、图片、视频、实时工作会话或媒体 URL，When Gateway 记录请求，Then 不输出 API Key、签名密钥、请求正文、图片／视频内容或完整媒体能力 URL。
- **AC-10.3** Given Gateway 提供性能观测数据，When 管理者查看基线，Then 可取得请求吞吐量、错误率、p50/p95/p99 延迟、SSE 首 token 延迟／流式时间及视频上传完成时间，且不以原始 API Key 作为维度。

**验证方式**：日志敏感字段测试、指标检查及端到端基线报告。

### REQ-11：YAML 配置与密钥注入

**需求**：Gateway 的可调整服务参数应由 src/gateway/configs/gateway.yml 加载；秘密值只由环境变量或受保护的挂载文件注入。

**验收条件**

- **AC-11.1** Given Gateway 启动，When 加载配置，Then 绑定地址、端口、推理上游、模型路由、请求正文限制、超时、租户限流与日志／指标参数均从 YAML 取得；默认绑定 127.0.0.1:8000，上游预留默认值为 http://127.0.0.1:8001。
- **AC-11.2** Given YAML 缺少必要字段、格式错误或存在无效上限，When Gateway 启动，Then 明确失败且不以未记录的代码常数静默替代必要配置。
- **AC-11.3** Given API Key、工作会话 HMAC 密钥或其他凭证，When Gateway 加载秘密，Then 从环境变量或安全挂载文件读取，且不要求在追踪中的 YAML 填入明文秘密。
- **AC-11.4** Given工作会话签名密钥轮替，When 新密钥开始签发句柄，Then 旧验证密钥至少保留至其签发句柄最长有效期结束；目前推理端工作会话最长期间配置为 3600 秒，Gateway 必须依实际部署配置保持一致。

**验证方式**：配置加载、无效配置拒绝、密钥来源及轮替兼容测试。

### REQ-12：兼容、回复与测试分层

**需求**：Gateway 的实现与验收应维持既有组件边界、明确的测试分层及可回复部署方式。

**验收条件**

- **AC-12.1** Given 新增 Gateway，When 项目查看组件责任，Then 推理服务仍负责模型生命周期、视频理解与 GPU；MediaMTX 仍负责 RTSP／RTMP 媒体流；Gateway 不加载模型、不管理 GPU，也不代理媒体数据包。
- **AC-12.2** Given 已登记模型，When 维护测试，Then 每个模型在 `tests/{model_alias}/` 提供组件对照、Inference 性能和 Gateway 端到端三项入口；Inference 性能测试直接调用推理 API，端到端测试经 Gateway，两条路径彼此分明。
- **AC-12.3** Given Gateway、推理服务及所需 MediaMTX 可用，When 执行端到端性能基线，Then 报告吞吐量、错误率、延迟分位数、SSE 首 token 延迟及视频上传完成时间；首版不配置硬性 SLA、不执行数据集精度测试。
- **AC-12.4** Given Gateway 发布需回复，When 部署者撤下 Gateway 入口并还原其 YAML，Then 推理服务与 MediaMTX 不需回滚变更；切换前应停止活动中的实时工作会话。

**验证方式**：包／责任边界查看、分层测试、回复演练及性能基线报告。

### REQ-13：依模型能力选择推理上游并拒绝不支持功能

**需求**：Gateway 必须依公开模型别名、每模型上游 URL 及能力配置路由请求；Gateway 配置需与 Inference Registry 的已登记 alias、上游标识符和能力保持一致。

**验收条件**

- **AC-13.1** Given `mage-vl-4b` 或 `qwen3-0.6b` 聊天请求，When Gateway 解析 JSON `model`，Then 转发至该模型 YAML 上游 URL，改写成相应上游模型 ID，并将 JSON／SSE 响应模型字段改回该项目别名。
- **AC-13.2** Given 聊天请求正文含有不受所选模型支持的输入模态或流式模式，When Gateway 执行 capability policy，Then 响应 HTTP 400 `unsupported_model_capability`，且不调用上游。
- **AC-13.3** Given视频 multipart 请求以选用 query `model` 指定别名，When Gateway 验证 `video_upload`，Then 以所选模型上游转发；省略时使用 YAML `defaults.video_model`；unsupported capability 在读取或转发请求正文前拒绝；转发保留原 multipart body、boundary 及 `Content-Length`。
- **AC-13.4** Given Live 创建 JSON 请求正文含选用 `model`，When Gateway 验证 `live_video`，Then 使用该模型上游并在句柄签章中绑定其 alias；省略时使用 `defaults.live_model`；events SSE 与停止请求必须使用句柄绑定的同一上游。句柄仍须验证签章、有效期及 API Key tenant。
- **AC-13.5** Given Gateway 配置多个不同推理 URL，When 客户端调用 `/readyz`，Then 每个不同上游都回报 ready 才返回成功。
- **AC-13.6** Given Gateway 与 Inference 各自保存模型路由，When 执行各模型 `test_end2end.py`，Then 验证已配置 alias 可由 Gateway 路由并返回预期公开模型名；Gateway 连接地址由测试 YAML 指定。

### REQ-14：在既有 Chat Completions route 转发标准 tool calls

**需求**：Gateway依每模型已登记的 `tool_calling` capability 拒绝或转发标准工具调用请求；响应由既有 Chat Completions route 返回。Gateway负责外部认证／tenant／限流，不解析模型 tool marker、不执行工具，也不新增专用 Agent endpoint。

**验收条件**

- **AC-14.1** Given `POST /v1/chat/completions` 携带标准 `tools` 与 `tool_choice`，When model route `tool_calling=true`，Then 转发指定 Inference URL及上游模型 ID，并将公开请求 model 改为 Inference model ID。
- **AC-14.2** Given Inference 返回非流式 tool call 或 SSE tool-call deltas，When Gateway 转发响应，Then 保留 `assistant.tool_calls`、`delta.tool_calls`、`finish_reason=tool_calls` 及 arguments，并将 response model 改回公开 alias。
- **AC-14.3** Given model route `tool_calling=false` 且请求包含 tools，When Gateway 处理请求，Then 不转发并返回明确 capability error。
- **AC-14.4** Given请求配置 `parallel_tool_calls=true`，When Gateway 处理请求，Then 于转发前明确拒绝；一般工具调用继续使用既有 Chat body limit、timeout、认证及租户政策。
- **AC-14.5** Given客户端收到完整有效的 `tool_calls`，When客户端执行工具并返回 `role=tool` 历史消息，Then 继续调用同一个 `/v1/chat/completions` route，不依赖额外 Agent endpoint。
- **AC-14.6** Given `/readyz` 与多模型配置，When Gateway执行 readiness，Then 只检查所有唯一 Inference URL；Gateway YAML 逐模型明确宣告 `tool_calling`，Mage false、Qwen true。

**验证方式**：Go config validation、既有 Chat route 的 mock／SSE alias 检查及 Qwen 经 Gateway 的 JSON／SSE tool-call 端到端请求；不另设 Agent route。

### REQ-15：Gateway 模块路径与代码仓库无关

**需求**：Gateway 的 Go module 和内部 import 不得包含代码托管平台、账号或具体仓库路径；仓库迁移不应改变公开 API、模型 alias 或部署端点。

**验收条件**

- **AC-15.1** Given Gateway 从当前仓库目录构建，When 检查 `go.mod` 与 Gateway Go source imports，Then module path 使用稳定的中性命名空间，且不含代码托管平台、账号或仓库 URL。
- **AC-15.2** Given Gateway 仓库迁移到不同 Git remote，When 按既有命令构建 Gateway，Then 不需要修改 module/import 路径；公开 HTTP API、模型 alias 和运行配置保持不变。

**验证方式**：搜索 Gateway 源码与 module metadata 中的旧仓库地址，并执行 `go build ./...` 检查内部 package imports。

**验证方式**：逐模型端到端脚本覆盖可执行的文字／图片 SSE 与视频上传流程；Live capability、授权拒绝矩阵及静态双配置文件一致性不属于目前三项模型测试入口的实机覆盖范围。

## 不变项与兼容要求

- 项目别名 mage-vl-4b 与上游 repo_id microsoft/Mage-VL 是不同标识符，不可混用。
- 推理服务默认为 127.0.0.1:8001；Gateway 默认预留 127.0.0.1:8000；实际地址由各自 YAML 控制。
- 推理层目前提供聊天文字／图片与 SSE、视频上传单次响应及实时工作会话控制；SSE 表示文字输出流式，不表示帧逐帧返回。
- 推理 API 目前要求相关 POST 请求提供 Content-Length；Gateway 必须兼容且限制内存使用。
- 图片与上传视频不依赖 MediaMTX；实时媒体流才需 MediaMTX。

## 数据处理和保留

- Gateway 不持久化聊天、图片、视频请求正文或推理输出。
- Gateway 不需数据库；首版租户限流状态为程序内状态。工作会话控制权由签名句柄承载，不在 Gateway 创建持久工作会话数据表。
- 日志只保留诊断所需的安全中继数据；实际保留期限由部署环境的日志政策配置。
- API Key 与 HMAC 密钥不记录、不放入 Git，也不放入普通 YAML。

## 明确不做

- 不提供模型精度或数据集评测功能。
- 不把 SGLang、vLLM、第二个模型、多实例共享状态或 Redis 列为首版已完成能力。
- 不经 Gateway 转发 RTSP／RTMP 媒体数据包。
- 不在 Gateway 实现模型推理、权重下载、GPU 资源调度或 MediaMTX 管理。
- 不在 Gateway 或 Inference Model Worker 执行 function tools；调用端在校验工具名称、参数及权限后负责执行。

## 需求追踪

| 上游条款／决定 | REQ | 模块／服务 | 任务 ID | AC | 验收证据／状态 |
|---|---|---|---|---|---|
| 已登记模型别名与上游路由 | REQ-01 | internal/route/、internal/upstream/、gateway.go | T-03、T-12、T-13 | AC-01.1、AC-01.2、AC-01.3、AC-13.1 | Mage／Qwen mock 路由已测；真实 Gateway smoke 已覆盖 Qwen 文字 JSON／SSE、Mage 图片 JSON／SSE／视频上传及 Live／MediaMTX 合成 RTSP |
| API Key 与租户识别 | REQ-02 | internal/auth/、gateway.go | T-02 | AC-02.1、AC-02.2、AC-02.3 | 已实现；mock 测试 |
| 既有聊天 JSON API | REQ-03 | gateway.go、internal/upstream/ | T-03、T-12 | AC-03.1、AC-03.2 | 已实现；mock 测试 |
| 既有聊天 SSE | REQ-04 | internal/stream/、internal/upstream/、gateway.go | T-04、T-12 | AC-04.1、AC-04.2、AC-04.3 | 已实现；取消专项测试待补 |
| 视频 multipart 与 Content-Length 限制 | REQ-05 | internal/videoproxy/、internal/upstream/、YAML | T-05、T-12 | AC-05.1、AC-05.2、AC-05.3 | 已实现；mock／静态兼容测试 |
| 实时 API、媒体直连与租户绑定句柄 | REQ-06 | gateway.go、internal/auth/ | T-06 | AC-06.1、AC-06.2、AC-06.3、AC-06.4 | 控制 API mock 测试通过；已完成一次 Gateway／MediaMTX 合成 RTSP 端到端 smoke，过期／未授权媒体访问仍待验证 |
| 单实例、每租户限流及请求正文上限 | REQ-07 | internal/middleware/、internal/config/ | T-07 | AC-07.1、AC-07.2、AC-07.3 | 已实现；mock 测试 |
| OpenAI 风格错误及上游超时 | REQ-08 | gateway.go、internal/upstream/、internal/stream/ | T-08、T-12 | AC-08.1、AC-08.2、AC-08.3 | 已实现；502 已测，504／中断 case 未完整测 |
| 健康与就绪语义 | REQ-09 | gateway.go、internal/upstream/ | T-09、T-12 | AC-09.1、AC-09.2、AC-09.3 | 已实现；mock 上游测试 |
| 日志、指标与敏感数据保护 | REQ-10 | internal/observability/、gateway.go | T-10 | AC-10.1、AC-10.2、AC-10.3 | 已实现；mock 指标／遮蔽测试，真实性能未执行 |
| YAML 配置及安全密钥注入 | REQ-11 | internal/config/、internal/auth/ | T-01、T-02 | AC-11.1、AC-11.2、AC-11.3、AC-11.4 | 已实现；checked-in YAML 与 mock 密钥测试 |
| 组件边界、测试分层、性能基线与回复 | REQ-12 | Gateway 包、tests/ | T-11、T-12 | AC-12.1、AC-12.2、AC-12.3、AC-12.4 | 包责任与测试分层已查看；实际性能与回复演练未执行 |
| 模型能力政策、双模型路由与配置一致性 | REQ-13 | internal/config/、gateway.go、模型目录端到端测试 | T-13 | AC-13.1、AC-13.2、AC-13.3、AC-13.4、AC-13.5、AC-13.6 | 历史 Gateway mock／跨配置测试已移出目前精简测试入口；真实 Gateway smoke 报告按模型保留，路由负向矩阵待另行安排 |
| Qwen tool capability 与 Chat route 转发边界 | REQ-14 | internal/config/、internal/route/、internal/upstream/、gateway.go、gateway.yml | T-15 | AC-14.1–AC-14.6 | 标准工具请求走既有 route；端到端证据记录于 tasks.md |
| Go module 与仓库迁移独立性 | REQ-15 | go.mod、Gateway Go imports | T-16 | AC-15.1、AC-15.2 | 中性 module path；源码旧仓库地址搜索及 Go build 状态见 tasks.md |

## 假设与未决问题

- [假设] TLS 由外部入口或反向代理终止；首版默认只在 127.0.0.1 绑定。部署直接对外暴露前需确认 TLS 保护。
- [假设] Gateway 和推理服务由同一部署管理者配置，能维持 Gateway 与推理端请求正文限制、实时工作会话期限一致。
- 没有未解决的产品范围决策；限流值、请求限制、超时与观测后端是部署配置，不是固定产品 SLA。

## 变更记录

| 日期 | 变更 |
|---|---|
| 2026-10-20 | 创建可追踪的 Gateway 行为规格与验收条件 |
| 2026-10-25 | 新增 Gateway Go module 与代码仓库无关的 REQ-15 |

