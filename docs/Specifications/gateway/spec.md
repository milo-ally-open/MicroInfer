# 行为规格：Go API 闸道

> 关联提案：[proposal.md](proposal.md)<br>
> 技术设计：[design.md](design.md)<br>
> 实施追踪：[tasks.md](tasks.md)<br>
> 状态：Go Gateway v1 已实作；REQ-13 记录双模型能力路由，REQ-14 将 Qwen tool capability 纳入既有 Chat Completions route，REQ-15 约束 Gateway module path 与代码仓库无关；验收状态见 tasks.md。
> 来源：使用者确认的 Go 闸道目标、根目录 AGENTS.md、README.md、src/gateway/README.md 与 src/inference/README.md。

## 范围和术语

本规格涵盖外部客户端至 Go Gateway、Gateway 至推论 API 的 HTTP 请求，以及即时工作阶段对 MediaMTX 的媒体资料流。Gateway 与推论服务是分离元件。媒体资料流由客户端直接连接 MediaMTX。

- 专案别名：`mage-vl-4b` 和 `qwen3-0.6b`，供客户端使用。
- 上游模型识别码：`microsoft/Mage-VL` 与 `qwen3-0.6b`，分别用于各模型推论服务路由。
- 租户：由已验证 API Key 对应的客户或呼叫方。
- 工作阶段句柄：Gateway 签发、绑定租户与上游工作阶段的 URL 安全识别码。
- OpenAI 风格错误：JSON 本文包含 error 物件，至少含 message、type、code；param 可为 null。

## 需求与验收条件

### REQ-01：公开模型别名与上游路由

**需求**：Gateway 应以 YAML 登录公开模型别名及其上游服务映射，只公开已登录的模型别名。

**验收条件**

- **AC-01.1** Given API Key 有效，When 客户端呼叫 GET /v1/models，Then 回应清单包含目前登录的 mage-vl-4b 和 qwen3-0.6b，且不暴露上游识别码。
- **AC-01.2** Given 客户端请求指定 model 为 mage-vl-4b，When Gateway 转送聊天请求，Then 上游请求使用 microsoft/Mage-VL，非串流回应及每个 SSE JSON 分块的 model 栏位均使用 mage-vl-4b。
- **AC-01.3** Given 请求未指定 model 或指定未登录的名称，When Gateway 验证聊天请求，Then 回应 HTTP 400 与 OpenAI 风格错误，且不呼叫上游。

**验证方式**：Gateway API 整合测试；验证请求与回应的模型栏位。

### REQ-02：API Key 验证与租户识别

**需求**：除健康及就绪端点外，所有公开 /v1/* API 都应以 Bearer API Key 验证呼叫者，并以伺服器端对应结果决定租户。

**验收条件**

- **AC-02.1** Given /v1/* 请求带有有效 Bearer API Key，When Gateway 验证请求，Then 使用该密钥设定所对应的租户执行政策判定。
- **AC-02.2** Given API Key 缺漏或无效，When 呼叫任一公开 /v1/* API，Then 回应 HTTP 401 与 OpenAI 风格错误，且不呼叫上游。
- **AC-02.3** Given 请求另带客户端自订租户标头，When Gateway 建立呼叫者身分，Then 不以客户端标头覆盖伺服器端租户对应。

**验证方式**：认证中介层测试及 API 整合测试。

### REQ-03：一般聊天请求

**需求**：Gateway 应依选定模型已登记的输入能力，将文字或图片／图文聊天请求转送至正确的推论服务，并保留相容的聊天栏位。Mage-VL 要求图片输入；Qwen3-0.6B 接受纯文字，两者都不接受对方未宣告的输入模态。

**验收条件**

- **AC-03.1** Given 聊天请求通过 API Key、模型及本文限制检查，When stream 为 false 或未启用，Then Gateway 转送请求并回传推论服务的 JSON 完成结果，仅改写公开 model 栏位及 Gateway 规定的错误资讯。
- **AC-03.2** Given 上游回应包含非 hop-by-hop 的内容型别及必要标头，When Gateway 回传结果，Then 保留与客户端契约相容的标头及状态码，且不泄漏上游内部位址或伺服器堆叠资讯。

**验证方式**：Gateway API 整合测试，涵盖文字及图片请求；不执行模型精度评估。

### REQ-04：聊天 SSE 串流

**需求**：Gateway 应转送聊天 SSE token 串流，保持事件边界、顺序、即时送出、取消及完成标记语义。

**验收条件**

- **AC-04.1** Given 聊天请求指定 stream 为 true，When 上游产生 SSE，Then Gateway 逐事件转送，刷新回应而不等待完整生成，并保留 data: [DONE] 结尾。
- **AC-04.2** Given 上游 SSE JSON 分块包含 model 栏位，When Gateway 对外转送该分块，Then 将栏位改写为请求选定的公开别名；SSE 注解、keep-alive、非 JSON 控制资料及 [DONE] 不被误解析为完成分块。
- **AC-04.3** Given 客户端在串流中断线或取消，When Gateway 收到取消讯号，Then 取消上游请求并释放相关连线与工作资源。

**验证方式**：串流测试，检查首事件延迟、事件顺序、反向压力及客户端取消。

### REQ-05：视讯上传转送

**需求**：Gateway 应将视讯理解 multipart 请求转送至推论服务，不在 Gateway 将完整档案读入记忆体。

**验收条件**

- **AC-05.1** Given multipart 请求包含 video、question 及目前推论 API 支援的选用栏位，When Gateway 转送请求，Then 保留 multipart 栏位与边界语义并以串流方式送至 POST /v1/videos/understand。
- **AC-05.2** Given 请求缺少 Content-Length、长度格式错误或超过 Gateway 设定上限，When Gateway 在读取大型本文前检查请求，Then 分别回应 HTTP 411、HTTP 400 或 HTTP 413，使用 OpenAI 风格错误且不呼叫上游。
- **AC-05.3** Given Gateway 与推论服务的视讯本文上限已设定，When 部署设定被验证，Then Gateway 对应上限不得高于推论端 inference.yml 的限制；Gateway 另以有界串流计数避免错报长度造成无界转送。

**验证方式**：multipart API 整合测试及大本文串流测试；不下载模型权重。

### REQ-06：即时工作阶段与租户隔离

**需求**：Gateway 应代理即时工作阶段控制 API，并确保只有建立工作阶段的租户可读取事件或停止该工作阶段。

**验收条件**

- **AC-06.1** Given 有效租户建立工作阶段，When Gateway 呼叫推论服务 POST /v1/live/sessions，Then 对外回应使用该租户绑定的 URL 安全签名句柄，events_url 与 stop_url 指向 Gateway，publish RTSP／RTMP URL 仍直连 MediaMTX。
- **AC-06.2** Given 同租户使用有效 API Key 与签名句柄呼叫事件或停止路由，When Gateway 验证句柄签章、租户及有效期，Then 以句柄内的模型别名和上游识别码代理 GET /v1/live/sessions/{session_id}/events 或 DELETE /v1/live/sessions/{session_id}。
- **AC-06.3** Given 句柄无效、过期或属于其他租户，When 客户端呼叫事件或停止路由，Then 回应一致的 HTTP 404 OpenAI 风格错误，不揭露工作阶段是否存在。
- **AC-06.4** Given 客户端透过 publish URL 连接 MediaMTX，When MediaMTX 授权回呼检查路径，Then 仅作用中的工作阶段可发布或读取；媒体 URL 视同能力凭证，客户端应保密，且工作阶段终止后失效。

**验证方式**：租户隔离测试、签章与有效期测试、SSE Last-Event-ID 续接测试，以及受控 MediaMTX 整合测试。

### REQ-07：请求大小、长度与租户限流

**需求**：Gateway 应在转送前套用 YAML 设定的本文限制及每租户限流，单实例首版的限流状态由程序内维护。

**验收条件**

- **AC-07.1** Given POST 聊天、视讯上传或工作阶段建立请求未带 Content-Length，When Gateway 验证请求，Then 回应 HTTP 411；格式错误长度回应 HTTP 400；超出设定上限回应 HTTP 413。
- **AC-07.2** Given 租户超出 YAML 设定的请求速率或突发容量，When 该租户再发送请求，Then Gateway 回应 HTTP 429 与 Retry-After，且其他租户不受该租户计数器直接影响。
- **AC-07.3** Given Gateway 程序重新启动，When 新程序开始接受流量，Then 程序内限流状态重新初始化；此行为须在运维文件中明确记载。

**验证方式**：本文边界、Content-Length、租户隔离及限流测试。

### REQ-08：错误格式与上游失败

**需求**：Gateway 自身及正规化后的上游错误应使用一致的 OpenAI 风格 JSON，且不得泄漏密钥、内部位址或堆叠。

**验收条件**

- **AC-08.1** Given Gateway 拒绝认证、模型、本文、限流或工作阶段请求，When 回应错误，Then JSON error 物件包含 message、type、code，并搭配适当 HTTP 状态码。
- **AC-08.2** Given 上游回传 HTTP 错误，When Gateway 回应客户端，Then 保留有意义的 HTTP 状态并将错误本文正规化；连线失败回 HTTP 502，逾时回 HTTP 504。
- **AC-08.3** Given SSE 标头已送出后发生上游错误，When Gateway 无法再更改 HTTP 状态，Then 终止串流并记录含 request ID 的错误，不把一般 JSON 错误插入 SSE token 资料。

**验证方式**：错误契约测试、上游模拟器测试及串流中断测试。

### REQ-09：健康与就绪

**需求**：Gateway 应分别提供程序存活检查与上游就绪检查。

**验收条件**

- **AC-09.1** Given Gateway 程序运行，When 呼叫 GET /healthz，Then 回应成功且不要求 API Key；此端点只表示 Gateway 程序仍存活。
- **AC-09.2** Given Gateway 设定有效且每个不同推论 URL 的 GET /health 均回报模型状态为 ok，When 呼叫 GET /readyz，Then 回应成功且不要求 API Key。
- **AC-09.3** Given 设定未载入或推论上游不可用／未就绪，When 呼叫 GET /readyz，Then 回应 HTTP 503，不宣称模型 API 可服务。

**验证方式**：健康端点测试及上游健康状态模拟测试。

### REQ-10：可观测性与敏感资料保护

**需求**：Gateway 应提供可依请求追踪的结构化日志与服务指标，且不记录敏感本文或凭证。

**验收条件**

- **AC-10.1** Given 每个 API 请求进入 Gateway，When 请求完成或失败，Then 日志可关联 request ID、路由、租户识别码或安全的租户代号、HTTP 状态、上游延迟与总延迟。
- **AC-10.2** Given 任何认证、聊天、图片、视讯、即时工作阶段或媒体 URL，When Gateway 记录请求，Then 不输出 API Key、签名密钥、请求本文、图片／视讯内容或完整媒体能力 URL。
- **AC-10.3** Given Gateway 提供性能观测资料，When 管理者检视基线，Then 可取得请求吞吐量、错误率、p50/p95/p99 延迟、SSE 首 token 延迟／串流时间及视讯上传完成时间，且不以原始 API Key 作为维度。

**验证方式**：日志敏感栏位测试、指标检查及端到端基线报告。

### REQ-11：YAML 设定与密钥注入

**需求**：Gateway 的可调整服务参数应由 src/gateway/configs/gateway.yml 载入；秘密值只由环境变数或受保护的挂载档案注入。

**验收条件**

- **AC-11.1** Given Gateway 启动，When 载入设定，Then 绑定位址、连接埠、推论上游、模型路由、本文限制、逾时、租户限流与日志／指标参数均从 YAML 取得；预设绑定 127.0.0.1:8000，上游预留预设值为 http://127.0.0.1:8001。
- **AC-11.2** Given YAML 缺少必要栏位、格式错误或存在无效上限，When Gateway 启动，Then 明确失败且不以未记录的程式常数静默替代必要设定。
- **AC-11.3** Given API Key、工作阶段 HMAC 密钥或其他凭证，When Gateway 载入秘密，Then 从环境变数或安全挂载档案读取，且不要求在追踪中的 YAML 填入明文秘密。
- **AC-11.4** Given工作阶段签名密钥轮替，When 新密钥开始签发句柄，Then 旧验证密钥至少保留至其签发句柄最长有效期结束；目前推论端工作阶段最长期间设定为 3600 秒，Gateway 必须依实际部署设定保持一致。

**验证方式**：设定载入、无效设定拒绝、密钥来源及轮替相容测试。

### REQ-12：相容、回复与测试分层

**需求**：Gateway 的实作与验收应维持既有元件边界、明确的测试分层及可回复部署方式。

**验收条件**

- **AC-12.1** Given 新增 Gateway，When 专案检视元件责任，Then 推论服务仍负责模型生命周期、视讯理解与 GPU；MediaMTX 仍负责 RTSP／RTMP 媒体流；Gateway 不载入模型、不管理 GPU，也不代理媒体封包。
- **AC-12.2** Given 已登录模型，When 维护测试，Then 每个模型在 `tests/{model_alias}/` 提供元件对照、Inference 性能和 Gateway 端到端三项入口；Inference 性能测试直接呼叫推论 API，端到端测试经 Gateway，两条路径彼此分明。
- **AC-12.3** Given Gateway、推论服务及所需 MediaMTX 可用，When 执行端到端性能基线，Then 报告吞吐量、错误率、延迟分位数、SSE 首 token 延迟及视讯上传完成时间；首版不设定硬性 SLA、不执行资料集精度测试。
- **AC-12.4** Given Gateway 发布需回复，When 部署者撤下 Gateway 入口并还原其 YAML，Then 推论服务与 MediaMTX 不需回滚变更；切换前应停止作用中的即时工作阶段。

**验证方式**：套件／责任边界检视、分层测试、回复演练及性能基线报告。

### REQ-13：依模型能力选择推论上游并拒绝不支援功能

**需求**：Gateway 必须依公开模型别名、每模型上游 URL 及能力设定路由请求；Gateway 设定需与 Inference Registry 的已登录 alias、上游识别码和能力保持一致。

**验收条件**

- **AC-13.1** Given `mage-vl-4b` 或 `qwen3-0.6b` 聊天请求，When Gateway 解析 JSON `model`，Then 转送至该模型 YAML 上游 URL，改写成相应上游模型识别码，并将 JSON／SSE 回应模型栏位改回该专案别名。
- **AC-13.2** Given 聊天本文含有不受所选模型支援的输入模态或串流模式，When Gateway 执行 capability policy，Then 回应 HTTP 400 `unsupported_model_capability`，且不呼叫上游。
- **AC-13.3** Given影片 multipart 请求以选用 query `model` 指定别名，When Gateway 验证 `video_upload`，Then 以所选模型上游转送；省略时使用 YAML `defaults.video_model`；unsupported capability 在读取或转送本文前拒绝；转送保留原 multipart body、boundary 及 `Content-Length`。
- **AC-13.4** Given Live 建立 JSON 本文含选用 `model`，When Gateway 验证 `live_video`，Then 使用该模型上游并在句柄签章中绑定其 alias；省略时使用 `defaults.live_model`；events SSE 与停止请求必须使用句柄绑定的同一上游。句柄仍须验证签章、有效期及 API Key tenant。
- **AC-13.5** Given Gateway 配置多个不同推论 URL，When 客户端呼叫 `/readyz`，Then 每个不同上游都回报 ready 才返回成功。
- **AC-13.6** Given Gateway 与 Inference 各自保存模型路由，When 执行各模型 `test_end2end.py`，Then 验证已设定 alias 可由 Gateway 路由并回传预期公开模型名；Gateway 连线位址由测试 YAML 指定。

### REQ-14：在既有 Chat Completions route 转送标准 tool calls

**需求**：Gateway依每模型已登记的 `tool_calling` capability 拒绝或转送标准工具调用请求；回应由既有 Chat Completions route 回传。Gateway负责外部认证／tenant／限流，不解析模型 tool marker、不执行工具，也不新增专用 Agent endpoint。

**验收条件**

- **AC-14.1** Given `POST /v1/chat/completions` 携带标准 `tools` 与 `tool_choice`，When model route `tool_calling=true`，Then 转送指定 Inference URL及上游模型 ID，并将公开请求 model 改为 Inference model ID。
- **AC-14.2** Given Inference 回传非串流 tool call 或 SSE tool-call deltas，When Gateway 转送回应，Then 保留 `assistant.tool_calls`、`delta.tool_calls`、`finish_reason=tool_calls` 及 arguments，并将 response model 改回公开 alias。
- **AC-14.3** Given model route `tool_calling=false` 且请求包含 tools，When Gateway 处理请求，Then 不转送并返回明确 capability error。
- **AC-14.4** Given请求设定 `parallel_tool_calls=true`，When Gateway 处理请求，Then 于转送前明确拒绝；一般工具调用继续使用既有 Chat body limit、timeout、认证及租户政策。
- **AC-14.5** Given客户端收到完整有效的 `tool_calls`，When客户端执行工具并回传 `role=tool` 历史消息，Then 继续呼叫同一个 `/v1/chat/completions` route，不依赖额外 Agent endpoint。
- **AC-14.6** Given `/readyz` 与多模型配置，When Gateway执行 readiness，Then 只检查所有唯一 Inference URL；Gateway YAML 逐模型明确宣告 `tool_calling`，Mage false、Qwen true。

**验证方式**：Go config validation、既有 Chat route 的 mock／SSE alias 检查及 Qwen 经 Gateway 的 JSON／SSE tool-call 端到端请求；不另设 Agent route。

### REQ-15：Gateway 模组路径与代码仓库无关

**需求**：Gateway 的 Go module 和内部 import 不得包含代码托管平台、账号或具体仓库路径；仓库迁移不应改变公开 API、模型 alias 或部署端点。

**验收条件**

- **AC-15.1** Given Gateway 从当前仓库目录构建，When 检查 `go.mod` 与 Gateway Go source imports，Then module path 使用稳定的中性命名空间，且不含代码托管平台、账号或仓库 URL。
- **AC-15.2** Given Gateway 仓库迁移到不同 Git remote，When 按既有命令构建 Gateway，Then 不需要修改 module/import 路径；公开 HTTP API、模型 alias 和运行设定保持不变。

**验证方式**：搜索 Gateway 源码与 module metadata 中的旧仓库地址，并执行 `go build ./...` 检查内部 package imports。

**验证方式**：逐模型端到端脚本覆盖可执行的文字／图片 SSE 与影片上传流程；Live capability、授权拒绝矩阵及静态双设定档一致性不属于目前三项模型测试入口的实机覆盖范围。

## 不变项与相容要求

- 专案别名 mage-vl-4b 与上游 repo_id microsoft/Mage-VL 是不同识别码，不可混用。
- 推论服务预设为 127.0.0.1:8001；Gateway 预设预留 127.0.0.1:8000；实际位址由各自 YAML 控制。
- 推论层目前提供聊天文字／图片与 SSE、视讯上传单次回应及即时工作阶段控制；SSE 表示文字输出串流，不表示影格逐帧回传。
- 推论 API 目前要求相关 POST 请求提供 Content-Length；Gateway 必须兼容且限制记忆体使用。
- 图片与上传视讯不依赖 MediaMTX；即时媒体流才需 MediaMTX。

## 资料处理和保留

- Gateway 不持久化聊天、图片、视讯本文或推论输出。
- Gateway 不需资料库；首版租户限流状态为程序内状态。工作阶段控制权由签名句柄承载，不在 Gateway 建立持久工作阶段资料表。
- 日志只保留诊断所需的安全中继资料；实际保留期限由部署环境的日志政策设定。
- API Key 与 HMAC 密钥不记录、不放入 Git，也不放入普通 YAML。

## 明确不做

- 不提供模型精度或资料集评测功能。
- 不把 SGLang、vLLM、第二个模型、多实例共享状态或 Redis 列为首版已完成能力。
- 不经 Gateway 转送 RTSP／RTMP 媒体封包。
- 不在 Gateway 实作模型推论、权重下载、GPU 资源排程或 MediaMTX 管理。
- 不在 Gateway 或 Inference Model Worker 执行 function tools；调用端在校验工具名称、参数及权限后负责执行。

## 需求追踪

| 上游条款／决定 | REQ | 模组／服务 | 任务 ID | AC | 验收证据／状态 |
|---|---|---|---|---|---|
| 已登录模型别名与上游路由 | REQ-01 | internal/route/、internal/upstream/、gateway.go | T-03、T-12、T-13 | AC-01.1、AC-01.2、AC-01.3、AC-13.1 | Mage／Qwen mock 路由已测；真实 Gateway smoke 已覆盖 Qwen 文字 JSON／SSE、Mage 图片 JSON／SSE／视讯上传及 Live／MediaMTX 合成 RTSP |
| API Key 与租户识别 | REQ-02 | internal/auth/、gateway.go | T-02 | AC-02.1、AC-02.2、AC-02.3 | 已实作；mock 测试 |
| 既有聊天 JSON API | REQ-03 | gateway.go、internal/upstream/ | T-03、T-12 | AC-03.1、AC-03.2 | 已实作；mock 测试 |
| 既有聊天 SSE | REQ-04 | internal/stream/、internal/upstream/、gateway.go | T-04、T-12 | AC-04.1、AC-04.2、AC-04.3 | 已实作；取消专项测试待补 |
| 视讯 multipart 与 Content-Length 限制 | REQ-05 | internal/videoproxy/、internal/upstream/、YAML | T-05、T-12 | AC-05.1、AC-05.2、AC-05.3 | 已实作；mock／静态相容测试 |
| 即时 API、媒体直连与租户绑定句柄 | REQ-06 | gateway.go、internal/auth/ | T-06 | AC-06.1、AC-06.2、AC-06.3、AC-06.4 | 控制 API mock 测试通过；已完成一次 Gateway／MediaMTX 合成 RTSP 端到端 smoke，过期／未授权媒体存取仍待验证 |
| 单实例、每租户限流及本文上限 | REQ-07 | internal/middleware/、internal/config/ | T-07 | AC-07.1、AC-07.2、AC-07.3 | 已实作；mock 测试 |
| OpenAI 风格错误及上游逾时 | REQ-08 | gateway.go、internal/upstream/、internal/stream/ | T-08、T-12 | AC-08.1、AC-08.2、AC-08.3 | 已实作；502 已测，504／中断 case 未完整测 |
| 健康与就绪语义 | REQ-09 | gateway.go、internal/upstream/ | T-09、T-12 | AC-09.1、AC-09.2、AC-09.3 | 已实作；mock 上游测试 |
| 日志、指标与敏感资料保护 | REQ-10 | internal/observability/、gateway.go | T-10 | AC-10.1、AC-10.2、AC-10.3 | 已实作；mock 指标／遮蔽测试，真实性能未执行 |
| YAML 设定及安全密钥注入 | REQ-11 | internal/config/、internal/auth/ | T-01、T-02 | AC-11.1、AC-11.2、AC-11.3、AC-11.4 | 已实作；checked-in YAML 与 mock 密钥测试 |
| 元件边界、测试分层、性能基线与回复 | REQ-12 | Gateway 套件、tests/ | T-11、T-12 | AC-12.1、AC-12.2、AC-12.3、AC-12.4 | 套件责任与测试分层已检视；实际性能与回复演练未执行 |
| 模型能力政策、双模型路由与设定一致性 | REQ-13 | internal/config/、gateway.go、模型目录端到端测试 | T-13 | AC-13.1、AC-13.2、AC-13.3、AC-13.4、AC-13.5、AC-13.6 | 历史 Gateway mock／跨设定测试已移出目前精简测试入口；真实 Gateway smoke 报告按模型保留，路由负向矩阵待另行安排 |
| Qwen tool capability 与 Chat route 转送边界 | REQ-14 | internal/config/、internal/route/、internal/upstream/、gateway.go、gateway.yml | T-15 | AC-14.1–AC-14.6 | 标准工具请求走既有 route；端到端证据记录于 tasks.md |
| Go module 与仓库迁移独立性 | REQ-15 | go.mod、Gateway Go imports | T-16 | AC-15.1、AC-15.2 | 中性 module path；源码旧仓库地址搜索及 Go build 状态见 tasks.md |

## 假设与未决问题

- [假设] TLS 由外部入口或反向代理终止；首版预设只在 127.0.0.1 绑定。部署直接对外暴露前需确认 TLS 保护。
- [假设] Gateway 和推论服务由同一部署管理者配置，能维持 Gateway 与推论端本文限制、即时工作阶段期限一致。
- 没有未解决的产品范围决策；限流值、请求限制、逾时与观测后端是部署设定，不是固定产品 SLA。

## 变更记录

| 日期 | 变更 |
|---|---|
| 2026-10-20 | 建立可追踪的 Gateway 行为规格与验收条件 |
| 2026-10-25 | 新增 Gateway Go module 与代码仓库无关的 REQ-15 |

