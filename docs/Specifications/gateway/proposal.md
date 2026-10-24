# 提案：Go API 闸道

> 状态：Go Gateway v1 已实作；本次补充两个已登录模型的能力路由，实际验收进度见 tasks.md。
> 路径：跨边界高风险<br>
> 负责人：专案维护者<br>
> 上游依据：[专案规范](../../../AGENTS.md)、[专案 README](../../../README.md)、[闸道目录说明](../../../src/gateway/README.md)、[推论服务说明](../../../src/inference/README.md)

## 背景与问题

建立本提案时，专案已有 Mage-VL-4B Transformers 推论 API，但尚无 Go 闸道服务。外部 API 身分验证、租户识别、请求政策、限流、模型路由与 API 可观测性需要由独立闸道承担，避免把客户端接入职责和模型载入、GPU 推论、模型专属处理混在一起。后续使用者明确要求依此规格实作 Gateway v1；目前实作及测试状态由 tasks.md 记录。

目前推论 API 预设监听 127.0.0.1:8001；Go Gateway v1 预设监听 127.0.0.1:8000。推论路由包含健康检查、模型清单、聊天完成、视讯上传及即时工作阶段 API。MediaMTX 是即时媒体流的 sidecar，不是闸道或模型服务。

## 目标与成功观察

- 提供以 Go 实作、可由 YAML 设定的外部 API 闸道，统一处理客户端验证、租户政策、模型路由、请求限制及可观测性。
- 在不承担模型生命周期或 GPU 工作的前提下，转送文字／图片聊天、SSE 文字输出、视讯上传及即时工作阶段控制 API。
- 依模型能力将 `mage-vl-4b`（`microsoft/Mage-VL`）与 `qwen3-0.6b`（`qwen3-0.6b`）路由至各自推论服务；拒绝未宣告的输入模态或 API。
- 建立 REQ、AC、T 任务及验收证据之间可追踪的规格流程。

## 范围

### 本次包含

- Gateway 的 HTTP API、Bearer API Key 与租户识别、单实例限流、错误契约、健康／就绪检查、上游路由、日志与指标需求。
- 聊天一般回应与 SSE、视讯档案上传、即时工作阶段建立／SSE 事件／停止操作的转送规格。
- MediaMTX 即时媒体流与推论服务之间的责任界线、安全要求、设定要求、失败行为、回复方式及后续测试分层。
- 在本规格组内维护提案、行为规格、技术设计与实施任务。

### 本次不包含／保持不变

- 原始规格阶段只建立文件，不实作 Go 服务；后续实作任务依本规格建立 Go Gateway，并只为将 live 本文限制移入 YAML 而调整推论设定／middleware。此提案不代表 SGLang、vLLM 或第二个模型已实作。
- SGLang 与 vLLM Gateway／Inference backend 未实作；本次不增加第三个模型，也不改写已登录模型的推论计算图。
- 不把模型权重、模型载入、生成、视讯解析、GPU 管理或 MediaMTX 媒体封包转送移入 Gateway。
- 不进行模型精度或资料集基准测试。端到端性能测试待 Gateway 完成后进行。
- 不修改 checkpoints/val.md、checkpoints/val.xlsx，也不移动或清理既有工作树内容。

## 现状依据与已确认决策

### 程式与文件现况

- src/gateway/README.md 描述已实作的 Go Gateway v1 及各套件职责；2026-10-22 已透过真实服务完成 Qwen 文字 JSON／SSE、Mage 图片 JSON／SSE／视讯上传，以及 Mage Live／MediaMTX 合成 RTSP 功能 smoke。Live 单次报告见 [`tests/mage_vl_4b/results/mage-vl-4b-live-smoke-position-fix-2026-10-22.json`](../../../tests/mage_vl_4b/results/mage-vl-4b-live-smoke-position-fix-2026-10-22.json)；正式容量、授权失败与部署回复仍待验收。
- 推论服务由 `models/*/configs/model.yaml` 自我登记 Mage-VL-4B 与 Qwen3-0.6B；Launcher 自动发现模型并要求启动命令明确指定 alias，两者各自位于模型包内。
- Mage 推论设定中的模型识别码是 microsoft/Mage-VL；Qwen 的 API model ID 是 qwen3-0.6b。两者皆以专案别名作 Gateway 路由键，识别码不可混淆。
- 推论服务目前提供 GET /health、GET /v1/models、POST /v1/chat/completions、POST /v1/videos/understand，以及 /v1/live/* 工作阶段 API。
- POST 聊天、视讯上传及即时工作阶段建立目前要求 Content-Length。当前推论设定的聊天本文上限为 15,728,640 bytes、视讯请求本文上限为 210,763,776 bytes、单一视讯档上限为 209,715,200 bytes；即时工作阶段建立上限目前为 65,536 bytes。
- 即时工作阶段回应包含 RTSP／RTMP 发布 URL、事件 URL、停止 URL 及上游工作阶段识别码。MediaMTX 授权回呼目前直接呼叫推论服务，且只允许作用中的工作阶段路径。

### 已确认决策

- 首版涵盖聊天／SSE、视讯上传与即时工作阶段控制；RTSP／RTMP 媒体资料流直接连接 MediaMTX。
- Gateway 依模型别名及明确 capability 路由两个已登录的 Inference URL；回应只公开专案别名，不暴露上游 repo ID。
- 首版以单一 Gateway 实例为目标，租户限流状态可保存在程序记忆体中。
- 客户端 API Key 与租户对应由环境变数或挂载的安全档案提供；密钥不得提交至 Git。
- Gateway 错误使用 OpenAI 风格 JSON。
- 即时工作阶段控制 API 使用租户绑定的签名工作阶段句柄；RTSP／RTMP URL 是随工作阶段失效的能力凭证。
- 端到端性能测试先建立实测基线，不在首版写入硬性 SLA。

### 技术假设

- Gateway 预设监听 127.0.0.1:8000；监听位址及连接埠由 Gateway YAML 设定，推论上游位址亦由该设定载入。
- 部署时由外部入口或反向代理终止 TLS；若日后要求 Gateway 直接对公网提供 TLS，应另行补充部署与凭证规格。
- 推论服务的本文限制仍由其 inference.yml 管理；Gateway 设定的请求本文上限不得超过推论端相应上限。

## 方案与取舍

| 方案 | 收益 | 成本／风险 |
|---|---|---|
| 单实例 Gateway、程序内限流、租户绑定签名句柄 | 不需新增共享资料库；符合目前部署与规模假设 | 多实例时须改用共享限流与工作阶段状态方案 |
| 首版使用共享状态服务 | 可直接扩展多实例 | 增加服务依赖、部署及维运工作，目前没有需求依据 |
| 让 RTSP／RTMP 媒体封包经 Gateway | 可集中处理网路入口 | 改变 MediaMTX 架构并增加长连线与频宽负担，不符合既定元件边界 |

建议采用第一列。这是已确认的首版部署取舍，不代表专案已具备多实例能力。

## 影响面、依赖与风险

- 主要影响：src/gateway/ 的后续 Go 实作、src/gateway/configs/gateway.yml、Gateway API 使用者文件及 tests/ 下的分层测试。
- 上游依赖：目前的 Mage-VL-4B API 与 MediaMTX；Gateway 不管理推论服务或 MediaMTX 的启动。
- 安全风险：API Key、签名密钥、请求本文、图片、视讯及含媒体 URL 的完整日志都可能暴露敏感资料；设计明确禁止记录这些内容。
- 兼容风险：推论端目前要求 Content-Length；Gateway 不应为了支援未知长度串流而将大型视讯整段缓存在记忆体。
- 运维风险：程序内限流在 Gateway 重启后重置；单实例不是高可用部署。
- 回复方式：停止或撤下 Gateway 入口并还原 Gateway YAML；推论服务与 MediaMTX 不需改动。仍作用中的媒体工作阶段依推论服务的工作阶段生命周期结束；正常切换前应先呼叫停止 API。

## 未决问题

没有阻挡规格实作的产品决策。部署者需在 Gateway YAML 中提供 API Key 来源、签名密钥、请求限制、限流值、逾时及上游位址；性能目标待基线结果出炉后再决定。

## 变更记录

| 日期 | 内容 | 决定／依据 |
|---|---|---|
| 2026-10-20 | 建立 Gateway 跨边界规格草稿 | 使用者已确认首版 API 范围、模型别名、单实例、错误格式、密钥来源、性能基线与即时流授权边界 |
| 2026-10-20 | 将路由、SSE、上游 HTTP 及影片 multipart 代理拆至 internal/route、stream、upstream、videoproxy | 让实作模组与设计责任一致；透过既有 Gateway mock 整合测试确认外部行为 |

