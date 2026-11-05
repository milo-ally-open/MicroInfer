# 提案：控制面后端接入 cloud-vlm kit

> 状态：待评审<br>
> 路径：跨边界高风险<br>
> 负责人：平台后端维护者；MicroInfer 维护者负责推理服务接口确认<br>
> 上游合同：LYG AI Platform Intern Kit v1.9.9 的 `docs/01-cloud-vlm.md`、`docs/00-sandbox.md` 与 `contracts/openapi.yaml`。当前副本位于本机忽略目录 `repos/lyg_ai_platform_intern_kit-master/`，不随 MicroInfer 仓库提交。

## 背景与问题

`lyg_ai_platform_intern_kit` 是 Vue 3 前端开发沙箱，不含产品后端源码；`packages/cloud-vlm/src/views/CloudModelsView.vue` 当前仍是占位页。云端模型手册已冻结面向前端的 `/api/cloud-models/*` 契约，并指明产品契约应进入 `backend-go/api/openapi.yaml`，由产品后端接既有镜头快照、告警中心与部署好的 MageVL。

MicroInfer 当前由 Python FastAPI 直接提供 `/v1/models`、`/v1/chat/completions` 与图片输入能力，API Key 鉴权位于 Inference Network 层。模型服务对外 ID 是 `mage-vl-4b`；kit 的业务模型 ID 是 `magevl-4b`。两个 ID 必须由控制面配置映射，不能要求浏览器知道推理服务内部命名。

本工作区没有产品 `backend-go`、数据库、登录/JWT、中控镜头 API 或既有告警实现，因此方案能确定职责与合同，但不能据此认定产品已有包路径、数据表或 Worker 框架。

## 目标与成功观察

- 浏览器继续只访问同源 `/api/*`，使用平台现有 Bearer JWT；浏览器不接触推理 API Key、推理服务地址或摄像头 RTSP 地址。
- 在既有 Go 产品后端内实现云端模型管理、任务持久化与授权；由后端 Worker 取得最新 JPEG、调用 MicroInfer FastAPI、保存最近结果，并通过现有告警领域入口去重写入告警。
- 前端能够按 kit 的字段读取模型、任务和执行状态；推理模型调用异步进行，UI 轮询任务状态，不让浏览器同步等待 GPU 推理。
- MicroInfer 继续作为模型推理服务；本方案不恢复已移除的 Go Gateway，不修改 Inference Engine、模型 Runtime 或推理 API。

## 范围

### 本次包含

- 设计产品后端路由、身份/租户权限、模型 ID 映射、任务状态机、持久化调度、图片适配、Inference HTTP Client、告警幂等、运行观测与验收计划。
- 说明 kit 的开发 Mock 契约如何映射到产品既有认证、设备/镜头、告警和推理服务。
- 列出开始实现前需要从产品后端仓库和平台负责人确认的接口与数据决策。

### 本次不包含 / 保持不变

- 不修改或合并 `lyg_ai_platform_intern_kit`；不改它的 `contracts/`、`mock/` 或验收夹具。
- 不在 MicroInfer 仓库实现控制面、用户登录、设备管理、任务数据库、摄像头取流或告警中心。
- 不新增公网 API Gateway、专用告警数据库、WebSocket、摄像头 RTSP 转发服务或模型权重容器。
- 不把 inference API Key、模型地址、帧内容或 RTSP/RTMP 凭据返回浏览器。

## 现状依据与约束

- kit v1.9.9：`docs/01-cloud-vlm.md` 定义模型、最新画面、任务、告警和错误语义；`docs/00-sandbox.md` 要求产品侧契约落在 `backend-go/api/`，前端由产品现有 `apiFetch` 调用。
- kit 当前 Mock：同一时刻最多 4 个运行任务；重复的设备/镜头/模型组合拒绝；持续命中时复用活动告警 ID；Mock 的固定账号/token 和 `X-Mock-*` header 仅供沙箱验收。
- MicroInfer：Mage-VL 接受 `image_url` 多模态消息；Inference `ChatCompletionRequest` 没有 `response_format` 字段，未知字段会被忽略。因此首版若要求 JSON 输出，必须由后端提示词约束、严格解析和校验，不能假定 JSON Schema 约束解码已支持。
- MicroInfer Mage-VL 当前图片限制为单张、最大边长 896、最大图片 10 MiB，Chat 请求正文上限 15 MiB；对接时需限图、压缩并在上传前拒绝过大输入。
- `[假设]` 产品后端使用现有 Go HTTP 服务及其认证/数据库/告警组件；kit 手册明确提到 `backend-go`，但其源代码未在此工作区。
- `[待确认]` Mage-VL 返回结构化 `hit/confidence/labels` 的 JSON 遵从率和 confidence 校准证据尚未提供。

## 备选方案与建议

| 方案 | 收益 | 成本/风险 |
|---|---|---|
| 在现有 `backend-go` 内增加云模型领域模块和独立 Worker 运行角色 | 复用平台 JWT、租户、设备、告警、数据库与部署；不增加浏览器到推理服务的新边界 | 需先检查现有 Go 项目的目录、任务框架、数据库和部署能力 |
| 新建另一个公开控制面微服务 | 可以独立发布 | 复制身份、权限和设备/告警数据；增加服务间鉴权、部署、迁移和跨服务事务；与 kit 的 `backend-go` 收回指引不符 |
| 把职责放进 MicroInfer FastAPI | 离模型服务近 | 混合用户控制面与 GPU 推理，复用不了平台既有设备/告警/JWT；违反当前 Inference 与控制面的边界 |

建议采用第一种：产品后端继续是控制面 API；Worker 可与 API 使用同一 Go 模块/镜像但以不同运行角色启动。若现有后端已有可靠的定时任务/队列，应复用；否则先采用数据库持久化的单 Worker 调度，不引入 Redis/Celery/Temporal 等新基础设施。并发和硬件容量必须用真实模型测量决定。

## 影响面、依赖与风险

- 产品范围/API：沿用 kit `/api/cloud-models/*`；生产告警页按 kit 收回说明改接既有告警 API，不创建第二套告警记录或推送通道。
- Inference 服务：仅作为私网/受控网络中的上游 `POST /v1/chat/completions`；服务端单独持有 Bearer API Key。API Key 不等同于用户 JWT。
- 数据/任务：任务和最近结果必须可恢复；多 Worker/多副本时使用行锁/租约和幂等键防止重复执行。
- GPU 容量：最多 4 个“运行中”任务是业务配额，不代表可并发执行 4 次 4B 图像推理。首发建议将同一模型的推理并发设为 1，超时/负载情况下跳过过期 tick，不能无限排队。
- 模型输出：VLM 自报的 `confidence` 不是经校准的概率；未完成带标注样本的校准前，不应将该数值用作安全告警阈值或产品准确率承诺。
- 回滚：关闭 cloud-vlm 功能开关并停止 Worker；保留任务/告警数据和数据库迁移，不回滚或重启 MicroInfer 模型服务。

## 未决问题

| 问题 | 影响 | 负责人/验证方式 | 状态 |
|---|---|---|---|
| 产品 `backend-go` 的真实源代码、数据库和后台任务框架是什么？ | 确定模块落点、事务和 Worker 实现 | 产品后端负责人提供只读仓库或架构导览 | `[待确认]` |
| 平台 JWT 中如何取得 user、tenant、role；任务/模型开关权限如何定义？ | 防止跨租户资源访问和越权启停 | 身份/权限负责人确认现有 middleware 与权限表 | `[待确认]` |
| 现有设备最新 JPEG API 如何鉴权；预览图 URL 如何给浏览器安全读取？ | 决定 FrameProvider 和预览兼容 | 设备/媒体服务负责人确认接口及大小限制 | `[待确认]` |
| 告警中心的活动告警 upsert/去重键与现有列表 API 是什么？ | 防止新建孤立告警库或重复报警 | 告警域负责人确认写入服务及来源字段 | `[待确认]` |
| `confidence` 是否已有校准定义，Mage-VL 能否稳定返回严格 JSON？ | 影响阈值语义、告警误报/漏报 | 模型负责人用标注样本做结构化输出与校准评估 | `[待确认]` |
| Inference API Key 如何安全注入 Go Worker；网络是否能访问模型 FastAPI？ | 决定密钥轮换与服务网络配置 | 平台运维与 MicroInfer 维护者确认 Secret 管理及端口 | `[待确认]` |

## 变更记录

| 日期 | 内容 | 决定/依据 |
|---|---|---|
| 2026-11-03 | 初稿 | 基于 kit v1.9.9、本工作区 MicroInfer API 与已知平台边界；产品后端实现待检查 |
