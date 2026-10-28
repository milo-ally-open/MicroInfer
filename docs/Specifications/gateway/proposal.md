# 提案：Go API 网关与 Qwen 标准工具调用

> 状态：Go Gateway v1 已实现；REQ-13 记录双模型 capability，REQ-14 将 Qwen 工具调用纳入既有 Chat Completions route，REQ-15 约束 Go module path 与代码仓库无关；验收进度见 tasks.md。
> 路径：跨边界高风险<br>
> 负责人：项目维护者<br>
> 上游依据：[项目规范](../../../AGENTS.md)、[项目 README](../../../README.md)、[网关目录说明](../../../src/gateway/README.md)、[推理服务说明](../../../src/inference/README.md)

## 背景与问题

创建本提案时，项目已有 Mage-VL-4B Transformers 推理 API，但尚无 Go 网关服务。外部 API 身份验证、租户识别、请求政策、限流、模型路由与 API 可观测性需要由独立网关承担，避免把客户端接入职责和模型加载、GPU 推理、模型专属处理混在一起。后续使用者明确要求依此规格实现 Gateway v1；目前实现及测试状态由 tasks.md 记录。

目前推理 API 默认监听 127.0.0.1:8001；Go Gateway v1 默认监听 127.0.0.1:8000。推理路由包含健康检查、模型清单、聊天完成、视频上传及实时工作会话 API。MediaMTX 是实时媒体流的 sidecar，不是网关或模型服务。

## 目标与成功观察

- 提供以 Go 实现、可由 YAML 配置的外部 API 网关，统一处理客户端验证、租户政策、模型路由、请求限制及可观测性。
- 在不承担模型生命周期或 GPU 工作的前提下，转发文字／图片聊天、SSE 文字输出、视频上传及实时工作会话控制 API。
- 依模型能力将 `mage-vl-4b`（`microsoft/Mage-VL`）与 `qwen3-0.6b`（`qwen3-0.6b`）路由至各自推理服务；拒绝未宣告的输入模态或 API。
- 创建 REQ、AC、T 任务及验收证据之间可追踪的规格流程。

## 范围

### 本次包含

- Gateway 的 HTTP API、Bearer API Key 与租户识别、单实例限流、错误契约、健康／就绪检查、上游路由、日志与指标需求。
- 聊天一般响应与 SSE、视频文件上传、实时工作会话创建／SSE 事件／停止操作的转发规格。
- MediaMTX 实时媒体流与推理服务之间的责任界线、安全要求、配置要求、失败行为、回复方式及后续测试分层。
- 在本规格组内维护提案、行为规格、技术设计与实施任务。

### 本次不包含／保持不变

- 原始规格阶段只创建文件，不实现 Go 服务；后续实现任务依本规格创建 Go Gateway，并只为将 live 请求正文限制移入 YAML 而调整推理配置／middleware。此提案不代表 SGLang、vLLM 或第二个模型已实现。
- SGLang 与 vLLM Gateway／Inference backend 未实现；本次不增加第三个模型，也不改写已登记模型的推理计算图。
- 不把模型权重、模型加载、生成、视频解析、GPU 管理或 MediaMTX 媒体数据包转发移入 Gateway。
- 不进行模型精度或数据集基准测试。端到端性能测试待 Gateway 完成后进行。
- 本 Gateway 工作项不新增、移动或清理模型验收素材；当前的算子分析/优化工作流文件由 Inference 规格组维护。

## 现状依据与已确认决策

### 代码与文件现况

- src/gateway/README.md 描述已实现的 Go Gateway v1 及各包职责；历史任务记录称 2026-10-22 曾透过真实服务完成 Qwen 文字 JSON／SSE、Mage 图片 JSON／SSE／视频上传，以及 Mage Live／MediaMTX 合成 RTSP 功能 smoke。对应的原始 Live JSON 报告不在当前仓库；正式容量、授权失败与部署演练仍待验收。
- 推理服务由 `models/*/configs/model.yml` 自我登记 Mage-VL-4B 与 Qwen3-0.6B；Launcher 自动发现模型并要求启动命令明确指定 alias，两者各自位于模型包内。
- Mage 推理配置中的模型 ID是 microsoft/Mage-VL；Qwen 的 API model ID 是 qwen3-0.6b。两者皆以项目别名作 Gateway 路由键，标识符不可混淆。
- 推理服务目前提供 GET /health、GET /v1/models、POST /v1/chat/completions、POST /v1/videos/understand，以及 /v1/live/* 工作会话 API。
- POST 聊天、视频上传及实时工作会话创建目前要求 Content-Length。当前推理配置的聊天请求正文上限为 15,728,640 bytes、视频请求正文上限为 210,763,776 bytes、单一视频文件上限为 209,715,200 bytes；实时工作会话创建上限目前为 65,536 bytes。
- 实时工作会话响应包含 RTSP／RTMP 发布 URL、事件 URL、停止 URL 及上游工作会话 ID。MediaMTX 授权回调目前直接调用推理服务，且只允许活动中的工作会话路径。

### 已确认决策

- 首版涵盖聊天／SSE、视频上传与实时工作会话控制；RTSP／RTMP 媒体数据流直接连接 MediaMTX。
- Gateway 依模型别名及明确 capability 路由两个已登记的 Inference URL；响应只公开项目别名，不暴露上游 repo ID。
- 首版以单一 Gateway 实例为目标，租户限流状态可保存在程序内存中。
- 客户端 API Key 与租户对应由环境变量或挂载的安全文件提供；密钥不得提交至 Git。
- Gateway 错误使用 OpenAI 风格 JSON。
- 实时工作会话控制 API 使用租户绑定的签名工作会话句柄；RTSP／RTMP URL 是随工作会话失效的能力凭证。
- 端到端性能测试先创建实测基线，不在首版写入硬性 SLA。

### 技术假设

- Gateway 默认监听 127.0.0.1:8000；监听地址及端口由 Gateway YAML 配置，推理上游地址亦由该配置加载。
- 部署时由外部入口或反向代理终止 TLS；若日后要求 Gateway 直接对公网提供 TLS，应另行补充部署与凭证规格。
- 推理服务的请求正文限制仍由其 inference.yml 管理；Gateway 配置的请求正文上限不得超过推理端相应上限。

## 方案与取舍

| 方案 | 收益 | 成本／风险 |
|---|---|---|
| 单实例 Gateway、程序内限流、租户绑定签名句柄 | 不需新增共享数据库；符合目前部署与规模假设 | 多实例时须改用共享限流与工作会话状态方案 |
| 首版使用共享状态服务 | 可直接扩展多实例 | 增加服务依赖、部署及维运工作，目前没有需求依据 |
| 让 RTSP／RTMP 媒体数据包经 Gateway | 可集中处理网络入口 | 改变 MediaMTX 架构并增加长连接与频宽负担，不符合既定组件边界 |

建议采用第一列。这是已确认的首版部署取舍，不代表项目已具备多实例能力。

## 影响面、依赖与风险

- 主要影响：src/gateway/ 的后续 Go 实现、src/gateway/configs/gateway.yml、Gateway API 使用者文件及 tests/ 下的分层测试。
- 上游依赖：目前的 Mage-VL-4B API 与 MediaMTX；Gateway 不管理推理服务或 MediaMTX 的启动。
- 安全风险：API Key、签名密钥、请求正文、图片、视频及含媒体 URL 的完整日志都可能暴露敏感数据；设计明确禁止记录这些内容。
- 兼容风险：推理端目前要求 Content-Length；Gateway 不应为了支持未知长度流式而将大型视频整段缓存在内存。
- 运维风险：程序内限流在 Gateway 重启后重置；单实例不是高可用部署。
- 回复方式：停止或撤下 Gateway 入口并还原 Gateway YAML；推理服务与 MediaMTX 不需改动。仍活动中的媒体工作会话依推理服务的工作会话生命周期结束；正常切换前应先调用停止 API。

## 未决问题

没有阻挡规格实现的产品决策。部署者需在 Gateway YAML 中提供 API Key 来源、签名密钥、请求限制、限流值、超时及上游地址；性能目标待基线结果出炉后再决定。

## 变更记录

| 日期 | 内容 | 决定／依据 |
|---|---|---|
| 2026-10-20 | 创建 Gateway 跨边界规格草稿 | 使用者已确认首版 API 范围、模型别名、单实例、错误格式、密钥来源、性能基线与实时流授权边界 |
| 2026-10-20 | 将路由、SSE、上游 HTTP 及视频 multipart 代理拆至 internal/route、stream、upstream、videoproxy | 让实现模块与设计责任一致；透过既有 Gateway mock 整合测试确认外部行为 |
| 2026-10-25 | 为 Qwen3 宣告工具能力并透传标准 Chat Completions 工具字段 | 按使用者最新要求模仿 OpenAI 兼容模型部署；不新增 Agent route，工具循环由客户端／Agent 应用执行 |
| 2026-10-25 | 移除 Gateway Go module 对当前代码托管仓库路径的耦合 | 使用稳定的中性模块命名空间；迁移代码仓库时不需要重写 Gateway imports |

## Qwen 标准 Tool Calling 增补

Qwen3 的工具定义、调用选择和历史 tool messages 通过既有 `/v1/chat/completions` 转给 Inference。Gateway 只按模型 `tool_calling` capability 拒绝或转发请求，并保留标准 `tool_calls`／SSE `delta.tool_calls` 响应。

Gateway 不解析模型原生工具标记，不执行工具，也不新增 Agent 专用 route。客户端／Agent 应用收到完整有效的 tool call 后，依自身工具注册表和权限策略执行，再把 `role: "tool"` 消息回送同一 endpoint。`tests/client.py` 提供最小 OpenAI SDK 端到端示例。

## 仓库迁移独立性增补

Gateway 的 Go module 与内部 import 使用稳定、中性的模块命名空间，不包含当前代码托管平台、账号或仓库路径。更换 Git 仓库时，Gateway 的 API、模型 alias 与部署配置保持不变。

