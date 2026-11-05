# 行为规格：控制面后端接入 cloud-vlm kit

> 关联提案：[proposal.md](proposal.md)；设计：[design.md](design.md)。
> 本文规定未来独立产品控制面的目标行为，不代表 MicroInfer 已实现这些接口。
> 上游依据为本机忽略目录中的 LYG AI Platform Intern Kit v1.9.9；该快照不随仓库提交。

## 范围与术语

- 控制面：产品业务后端，复用现有平台认证、租户/设备权限、任务存储、摄像头快照和告警中心；在产品后端仓库实现。
- Inference：本仓库 Python FastAPI 推理服务，负责模型 API Key、限流和推理；不负责用户 JWT、设备、任务或告警。
- kit 合同：contracts/openapi.yaml 和 docs/01-cloud-vlm.md 定义的 /api/cloud-models/*；kit 是 Vue 开发沙箱与 Mock，不是生产后端。
- 模型 ID：前端业务 ID magevl-4b 映射到 Inference ID mage-vl-4b，映射由服务端配置维护。

## 需求与验收条件

### REQ-01：复用平台身份并实施对象级授权

**需求**：浏览器使用平台 Bearer JWT；控制面校验角色、租户及模型、任务、设备和摄像头的对象权限。kit Mock 登录和固定 Token 只用于开发。

**验收条件**

- **AC-01.1** Given 有效平台 JWT，When 用户访问其租户授权的资源，Then 返回数据且不要求 kit Mock 凭据。
- **AC-01.2** Given 用户访问其他租户或无权对象，When 读取或修改，Then 拒绝请求且不泄露资源详情。

验证方式：API 集成测试覆盖角色、租户和对象归属。

### REQ-02：模型目录和 ID 映射

**需求**：提供 kit 模型目录、启用状态，并将业务 ID 映射到部署配置中的 Inference ID。

**验收条件**

- **AC-02.1** Given Mage-VL 已部署且启用，When 请求 GET /api/cloud-models，Then 返回 magevl-4b 及显示名、类型和状态，调用推理时使用 mage-vl-4b。
- **AC-02.2** Given 模型未启用或上游不可用，When 创建或启动任务，Then 按合同拒绝且不把业务 ID 直接发送给 Inference。

验证方式：目录 API 契约与上游预检测试。

### REQ-03：适配既有摄像头快照

**需求**：通过产品已有设备/媒体服务取得授权 JPEG；浏览器和模型请求均不得控制 RTSP 或任意 URL。

**验收条件**

- **AC-03.1** Given 用户有摄像头权限，When 请求 GET /api/cloud-models/frames/latest，Then 返回 kit 帧元数据及受控预览地址。
- **AC-03.2** Given Worker 获取图像，When 发送给 Inference，Then 校验格式、大小与尺寸，满足当前单图、最大边 896、图像 10 MiB、请求体 15 MiB 限制。

### REQ-04：任务字段校验与持久化

**需求**：按 kit 保存 device_id、camera_id、model_id、prompt、interval_ms、min_confidence、severity、enabled、status、last_error 和 last_result。

**验收条件**

- **AC-04.1** Given 合法请求，When 创建或更新停止态任务，Then 校验 prompt 1–200 字符、间隔 1000–60000 ms、阈值 0–1、severity 枚举及资源归属。
- **AC-04.2** Given 同设备/摄像头/模型组合已存在或任务正在运行，When 重复创建、更新或删除，Then 按 kit 合同拒绝；组合冲突由持久层约束。

### REQ-05：原子任务生命周期

**需求**：start/stop 修改持久状态并快速返回完整任务；推理由后台执行。running 任务上限 4，配额和转换须原子化。

**验收条件**

- **AC-05.1** Given 停止任务、可用模型和剩余配额，When start，Then 快速返回 running，Worker 异步执行，浏览器轮询结果。
- **AC-05.2** Given 已有 4 个 running 任务，When 并发 start，Then 配额外请求返回 429；stop 后在途请求不得复活任务。

### REQ-06：可恢复且有界的调度

**需求**：Worker 按间隔执行，同一任务不可并发重入；重启可恢复，错过的 tick 合并或跳过。

**验收条件**

- **AC-06.1** Given 多 Worker 竞争到期任务，When 领取工作，Then 队列或原子租约保证只有一个有效执行者。
- **AC-06.2** Given 重启、超时或调度落后，When 恢复运行，Then 租约可恢复、重试有界、过期 tick 不积压且失败写入 last_error。

### REQ-07：通过 FastAPI 调用并校验模型输出

**需求**：Worker 使用私网地址和服务端 API Key 调用 POST /v1/chat/completions，非流式提交单张图像。模型输出只是建议，不得绕过业务授权执行动作。

**验收条件**

- **AC-07.1** Given 有效任务和图片，When 调用 Inference，Then 使用 mage-vl-4b、独立 Bearer key 和有界超时；平台 JWT 与推理 key 不下发浏览器。
- **AC-07.2** Given 模型返回文本，When 解析 hit、summary、confidence、labels，Then 严格检查类型/范围；无效输出不算命中。

验证方式：Client 契约、错误输出及固定模型版本验收。当前 Inference 不支持 response_format 约束解码，不能假设提示词必然产生合法 JSON。

### REQ-08：保存结果并复用告警中心

**需求**：保存时间、frame_id、hit、summary、confidence、latency_ms 和 labels；合格命中经既有告警服务幂等更新，不新建告警存储。

**验收条件**

- **AC-08.1** Given 合法结果，When 一次推理完成，Then 更新任务 last_result 并记录耗时与来源帧。
- **AC-08.2** Given hit=true 且达到阈值，When 同一活动事件持续命中，Then 更新同一活动告警；非命中或无效输出不产生告警。

模型自报 confidence 未经校准前不得视为概率。

### REQ-09：安全、可观测和资源有界

**需求**：保护密钥、帧和提示词；限制请求速率、图片大小、并发、超时、重试和日志，并提供运行指标。

**验收条件**

- **AC-09.1** Given 源码、浏览器和服务日志，When 检查密钥边界，Then 推理 key 不出现于源码/浏览器/普通日志，原始帧不写入常规日志。
- **AC-09.2** Given 超时、限流、过大图像或上游故障，When 请求失败，Then 资源有界、错误可关联且不无限重试或重复告警。

### REQ-10：保持 kit HTTP 合同和轮询行为

**需求**：保持 kit 路径、字段、错误和生命周期语义；浏览器只通过同源 /api/* 使用平台认证。

**验收条件**

- **AC-10.1** Given kit 定义请求，When 调用模型、帧、任务、start/stop 和兼容告警路径，Then OpenAPI 字段/状态语义一致。
- **AC-10.2** Given 长任务，When 页面每秒轮询或离开页面，Then 返回持久状态且不依赖 WebSocket 或浏览器同步等待 GPU。

### REQ-11：支持安全灰度与回滚

**需求**：功能可独立开关；停用不影响平台告警、摄像头和 MicroInfer。

**验收条件**

- **AC-11.1** Given cloud-vlm 未启用，When 部署后端版本，Then 旧业务不变且 Worker 不领取此类任务。
- **AC-11.2** Given 需要回滚，When 关闭开关并停止 Worker，Then 不再发起推理，任务/告警数据保留且既有服务可用。

## 不变项、数据和不做事项

- MicroInfer 保持 Python FastAPI；不恢复旧 Go Gateway，也不在本仓库新增 Go 控制面源码。
- 用户 JWT 与 Inference API Key 分属不同信任边界；kit 的 magevl-4b 与推理 mage-vl-4b ID 不变。
- 任务持久化；last_result 仅保存最近结果。默认不长期存储原始帧或完整模型输入，具体保留期遵循产品治理。
- 不在 MicroInfer 实现用户、租户、任务数据库、摄像头、告警中心；不新增公网 Gateway、WebSocket 或 RTSP 转发。
- 不把自报 confidence 当作校准概率，也不直接执行模型提出的外部动作。

## 需求追踪

| 上游条款 | REQ | 模块 | 任务 | AC | 状态 |
|---|---|---|---|---|---|
| 平台身份与对象授权 | REQ-01 | 产品认证/授权 | T-02 | AC-01.1–AC-01.2 | 待实施 |
| 模型目录与 ID 映射 | REQ-02 | 模型目录/Inference Client | T-03、T-06 | AC-02.1–AC-02.2 | 待实施 |
| 最新帧 API | REQ-03 | FrameProvider | T-03 | AC-03.1–AC-03.2 | 待实施 |
| 任务合同与冲突 | REQ-04 | Task API/Repository | T-04 | AC-04.1–AC-04.2 | 待实施 |
| 启停与四任务配额 | REQ-05 | Task Service/Worker | T-04、T-05 | AC-05.1–AC-05.2 | 待实施 |
| 有界后台执行 | REQ-06 | Worker/Scheduler | T-05 | AC-06.1–AC-06.2 | 待实施 |
| FastAPI 图像推理 | REQ-07 | Inference Client | T-06 | AC-07.1–AC-07.2 | 待实施 |
| 告警与最近结果 | REQ-08 | Alert adapter | T-07 | AC-08.1–AC-08.2 | 待实施 |
| 安全与资源限制 | REQ-09 | 全服务/运维 | T-09 | AC-09.1–AC-09.2 | 待实施 |
| kit OpenAPI 与轮询 | REQ-10 | API/前端联调 | T-08 | AC-10.1–AC-10.2 | 待实施 |
| 灰度与恢复 | REQ-11 | 部署/运维 | T-10 | AC-11.1–AC-11.2 | 待实施 |

## 假设与未决问题

- [假设] 产品现有 Go 后端具有可复用的认证、数据库、设备和告警模块，须在独立产品仓库核实。
- [待确认] JWT claims/RBAC、任务 schema、scheduler、快照服务和活动告警幂等接口尚未检查。
- [待确认] Mage-VL JSON 遵从率与 confidence 校准数据尚未提供；须由模型负责人使用标注数据评估。

## 变更记录

| 日期 | 变更 |
|---|---|
| 2026-11-03 | 根据 kit v1.9.9 与 MicroInfer 当前 FastAPI 能力建立跨边界行为规格。 |
