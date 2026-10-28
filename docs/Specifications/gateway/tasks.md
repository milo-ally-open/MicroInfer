# 实施任务：Go API 网关

> 关联规格：[spec.md](spec.md)；关联设计：[design.md](design.md)；提案：[proposal.md](proposal.md)<br>
> 状态：T-01 至 T-14 记录 Gateway v1 基线；REQ-14/T-15 将 Qwen tool capability 接入既有 Chat route，REQ-15/T-16 已移除 module path 对当前代码托管位置的依赖。新增验收状态见下表。
> Go Gateway v1 已实现，不代表已部署或已连接真实推理服务。

## 任务清单

### 1. 配置与服务骨架

- [x] **T-01 创建 Gateway 配置加载与服务骨架** — 交付：`src/gateway/go.mod`、`internal/config/`、`cmd/gateway/main.go`、`configs/gateway.yml`；严格加载 YAML、拒绝无效配置、启动／停止 HTTP server；覆盖：REQ-09 / AC-09.1；REQ-11 / AC-11.1、AC-11.2；状态：完成。

### 2. 认证、路由与转发

- [x] **T-02 实现 API Key、租户识别与秘密加载** — 交付：`internal/auth/` 的 Bearer API Key、tenant mapping、常数时间杂凑比对及轮替 HMAC 句柄；API Key 由挂载 JSON 文件加载，签章密钥由环境变量注入；覆盖：REQ-02 / AC-02.1、AC-02.2、AC-02.3；REQ-11 / AC-11.3、AC-11.4；状态：完成。
- [x] **T-03 实现模型别名、模型清单与一般聊天代理（初始 Mage 路由）** — 交付：创建 `mage-vl-4b` → `microsoft/Mage-VL` 的起始路由；双模型 capability 路由由 T-13 扩充；覆盖：REQ-01 / AC-01.1、AC-01.2、AC-01.3；REQ-03 / AC-03.1、AC-03.2；状态：完成初始基线。
- [x] **T-04 实现聊天 SSE 转发** — 交付：有界 SSE event parser、model 别名改写、逐事件 flush、完成标记、取消与闲置／总超时处理；覆盖：REQ-04 / AC-04.1、AC-04.2、AC-04.3；REQ-08 / AC-08.3；状态：完成。
- [x] **T-05 实现视频 multipart 有界代理** — 交付：`Content-Length` 检查、YAML 请求正文上限、有界流式转发及 multipart boundary 保留；覆盖：REQ-05 / AC-05.1、AC-05.2、AC-05.3；REQ-07 / AC-07.1；状态：完成。
- [x] **T-06 实现实时工作会话签名句柄** — 交付：租户绑定 HMAC 句柄、事件／停止路由验证、`Last-Event-ID` 转发与 Gateway URL 改写；MediaMTX publish URL 保持原值；覆盖：REQ-06 / AC-06.1、AC-06.2、AC-06.3、AC-06.4；REQ-11 / AC-11.4；状态：完成。
- [x] **T-07 实现租户限流与请求正文政策** — 交付：YAML 速率／突发值、程序内 tenant token bucket、`Retry-After`、请求正文长度与位元组上限检查；覆盖：REQ-07 / AC-07.1、AC-07.2、AC-07.3；状态：完成。
- [x] **T-08 实现错误契约、上游超时及失败处理** — 交付：OpenAI 风格错误、上游 4xx/5xx 正规化、连接失败 502、超时 504、流式中断日志；覆盖：REQ-08 / AC-08.1、AC-08.2、AC-08.3；状态：完成。

### 3. 健康检查、观测与验收

- [x] **T-09 实现健康与就绪端点** — 交付：免认证 `/healthz` 存活与 `/readyz` 上游模型就绪检查；覆盖：REQ-09 / AC-09.1、AC-09.2、AC-09.3；状态：完成。
- [x] **T-10 加入安全日志与服务指标** — 交付：request ID、路由／租户杂凑／状态／上游与总延迟日志，Prometheus request/error/histogram 指标，不输出 API Key、请求正文或租户标签；覆盖：REQ-10 / AC-10.1、AC-10.2、AC-10.3；状态：完成。
- [x] **T-11 创建分层测试、回复流程与文件同步** — 交付：`tests/gateway/` mock API 测试、`tests/inference/` 静态契约测试、`tests/end2end/` 客户端与配置兼容测试、端到端性能工具、启动／回复文件及 README 同步；覆盖：REQ-12 / AC-12.1、AC-12.2、AC-12.3、AC-12.4；状态：工具及文件完成；Qwen 与 Mage-VL 已有低样本真实服务 smoke，正式性能基线及部署回复演练未完成。
- [x] **T-12 将 Gateway HTTP 能力拆分至职责包** — 交付：`internal/route/` 路径分派、`internal/upstream/` HTTP transport、`internal/stream/` SSE 转发、`internal/videoproxy/` multipart 流式代理；由 `gateway.go` 组合各组件，不改变公开 API 契约；覆盖：REQ-01 / AC-01.1、AC-01.2；REQ-03 / AC-03.1、AC-03.2；REQ-04 / AC-04.1、AC-04.2；REQ-05 / AC-05.1、AC-05.2、AC-05.3；REQ-08 / AC-08.1、AC-08.2、AC-08.3；REQ-09 / AC-09.2；REQ-12 / AC-12.1、AC-12.2；状态：完成，`tests/gateway/` 整合测试通过。
- [x] **T-13 登记两模型 capability 并路由至各自推理服务** — 交付：Gateway YAML 的 Mage／Qwen route、输入与路由能力检查、视频／Live 模型选择、Live 句柄模型绑定、多上游 readiness；覆盖：REQ-01 / AC-01.1、AC-01.2；REQ-03 / AC-03.1；REQ-04 / AC-04.2；REQ-06 / AC-06.2；REQ-09 / AC-09.2；REQ-13 / AC-13.1、AC-13.2、AC-13.3、AC-13.4、AC-13.5、AC-13.6；状态：Gateway mock 测试、跨配置测试及 Spec 结构检查通过；另有 Qwen 文字 JSON／SSE、Mage 图片 JSON／SSE／视频上传、Mage Live／MediaMTX 合成 RTSP 真实 Gateway smoke。Live 与媒体授权仅完成单次功能检查，未验证失败授权与到期边界。
- [x] **T-14 将 Gateway 验收入口收敛至逐模型测试** — 交付：`tests/{model_alias}/test_end2end.py` 验证各模型 Gateway SSE／媒体路由，并于 Gateway README／Spec 指明 Inference 性能比较与 Gateway 端到端测试的区别；清理原有多目录测试树并将历史 JSON 按模型归文件；覆盖：REQ-12 / AC-12.2；REQ-13 / AC-13.6；状态：两个模型脚本已创建；未启动 Gateway／Inference 实机服务。
- [x] **T-15 将标准 tool capability 接入既有 Chat route** — 交付：模型 route `tool_calling` capability、工具请求拒绝／转发，保留标准 JSON/SSE tool_calls 与现有认证／限制／readiness；不新增 Agent endpoint；覆盖：REQ-14 / AC-14.1、AC-14.2、AC-14.3、AC-14.4、AC-14.5、AC-14.6；依赖：T-13；状态：实现与 `go build ./...` 通过；Qwen 直接 Inference 的客户端工具循环及 SSE 已验收。Windows Application Control 阻止启动本次 Gateway 可执行文件，完整 Gateway 端到端待可运行的受信任环境验收。
- [x] **T-16 移除 Gateway 对当前代码托管路径的耦合** — 交付：Go module 与内部 imports 改用稳定中性命名空间；不改变 HTTP API、模型 alias 或部署配置；覆盖：REQ-15 / AC-15.1、AC-15.2；状态：`go list -m` 显示中性 module path，`go build ./...` 通过，Gateway 源码不再包含旧托管地址；HTTP API 与 YAML 未修改。

## 验收记录

| AC | 检查方法 | 结果／证据 | 状态 |
|---|---|---|---|
| AC-01.1、AC-01.2、AC-01.3；AC-02.1、AC-02.2；AC-03.1、AC-03.2 | `go test ./...`（`tests/gateway/`） | models alias、认证、上游模型映射、一般响应改写及不转发 API Key 测试 | 通过 mock 测试 |
| AC-04.1、AC-04.2 | `TestChatSSERewritesAliasAndPreservesDone` | SSE 分块别名与 `[DONE]` 保留 | 通过 mock 测试 |
| AC-04.3 | 代码查看 | request context 取消会关闭上游请求；没有专门客户端断线测试 | 代码已实现，runtime case 待补 |
| AC-05.1、AC-05.2；AC-07.1、AC-07.2 | multipart、Content-Length、token bucket 整合测试 | boundary／请求正文相同、缺少长度 411、请求正文过大 413、速率超限 429 | 通过 mock 测试 |
| AC-05.3 | `tests/end2end/test_config_compatibility.py` | gateway chat/video/live request limits 均不高于 inference YAML 对应值 | 通过静态配置检查 |
| AC-06.1、AC-06.2、AC-06.3 | live handle 测试 | URL 改写、租户隔离、密钥轮替和签章窜改拒绝 | 通过 mock 测试 |
| AC-06.4 | Mage Live／MediaMTX 合成 RTSP 单次 smoke | 历史记录称 Gateway 创建、事件与停止均 HTTP 200；FFmpeg 发布的帧经 MediaMTX 读取并触发 `analysis`。原始 JSON 报告不在当前仓库；句柄过期、非活动 session 拒绝及未授权 publish/read 未测 | 部分验证；授权边界待测 |
| AC-07.3、AC-08.1、AC-08.2、AC-09.1、AC-09.2 | Go handler 测试与代码查看 | 程序内限流、OpenAI error、502、ready／health 有 mock 验证；504 专门延迟 case 未执行 | 部分通过 |
| AC-08.3、AC-10.1、AC-10.2 | 日志测试及 SSE 实现查看 | request ID、上游延迟字段存在；API Key／请求正文不出现在日志 | 通过 mock 测试 |
| AC-10.3 | Gateway `/metrics` 与历史真实服务 smoke | Qwen text-sse 有 3 个请求，另有 JSON／SSE 各 1 次；Mage 图片 JSON／SSE、视频上传及 Live／MediaMTX 合成 RTSP 各 1 次；Mage 配置见 [`analysis/mage_vl_4b/analysis.md`](../../../analysis/mage_vl_4b/analysis.md)，JSON 见 `tests/qwen3_0_6b/results/`、`tests/mage_vl_4b/results/`；样本不足以代表容量 |
| AC-11.1、AC-11.2、AC-11.3、AC-11.4 | YAML 加载、配置兼容检查、mock handler 测试 | 已加载 checked-in YAML，测试 API key／HMAC 以代码注入；未以部署密钥启动服务 | 静态／mock 通过；部署验证待执行 |
| AC-12.1、AC-12.2 | 原始码与逐模型测试目录查看 | Gateway 不汇入推理包；每个模型以一个端到端入口经 Gateway 验证，与 Inference API 性能测试分开 | 通过查看；实机测试依赖服务状态 |
| AC-12.3、AC-12.4 | 性能工具／回复文件查看及真实服务 smoke | 已提供性能工具与回复步骤；Qwen text-sse、Mage 图片／视频低样本 smoke 已连 GPU，Mage Live／MediaMTX 完成一次合成 RTSP smoke；未执行部署回复演练 | 工具／文件完成；功能 smoke 完成，正式容量及回复演练待执行 |
| T-12 涉及之 API／流式／上游验收条件 | `go test -count=1 ./...`（`tests/gateway/`）及 `go test -count=1 ./...`（`src/gateway/`） | 新增模块编译；既有 mock handler 测试涵盖模型路由、SSE 别名、multipart boundary／请求正文、上游错误及健康检查 | 通过本地 mock 测试；未连真实推理服务 |
| AC-13.1 至 AC-13.5 | 旧 Go Gateway mock 与真实服务 smoke 纪录 | 精简测试前的 mock cases 与真实链路已验证 Qwen JSON／SSE、Mage 图片 JSON／SSE／视频上传及 Live／MediaMTX RTSP；报告见 `tests/qwen3_0_6b/results/`、`tests/mage_vl_4b/results/` 与 [`analysis/mage_vl_4b/analysis.md`](../../../analysis/mage_vl_4b/analysis.md) | mock 结果为历史纪录；真实 smoke 样本少，Live 不代表授权边界或容量 |
| AC-13.6 | 历史 `test_config_compatibility.py`（已移除）；现行模型 `test_end2end.py` | 精简前 10 项静态配置测试通过；现行脚本验证公开 alias、Gateway 聊天 SSE 与各模型媒体路径 | 历史配置检查通过；新脚本已创建，待实机执行 |
| REQ-13 规格追踪 | `validate_spec_group.py docs/Specifications/gateway` | 13 REQ、44 AC、13 T 的需求／验收／任务映射完整 | 通过 |
| REQ-14 Qwen tool calling／既有 Chat route | `go build ./...`、本地 Qwen JSON/SSE smoke、Gateway HTTP 端到端 | `go build ./...` 通过；Qwen Inference 非流式工具调用往返和 SSE `delta.tool_calls`／`finish_reason=tool_calls` 通过；Gateway 两种运行目录下启动均被 Windows Application Control 拦截，未向 Gateway 发出端到端请求 | Inference 通过；Gateway 整合未验收 |
| AC-15.1、AC-15.2 Gateway 仓库无关 module path | Gateway 源码搜索、`go list -m`、`go build ./...`、差异查看 | 搜索未找到旧托管地址；module path 为 `example.com/videostreamapi/gateway`；编译通过；变更只涉及 Go module/import 和规格文档 | 本地检查通过；未实际迁移 Git remote |

## 实施偏差与决定

| 日期 | 偏差／新发现 | 规格／设计更新 | 依据 |
|---|---|---|---|
| 2026-10-20 | 原推理 middleware 将 live 创建及 MediaMTX callback 的 65,536 bytes 上限写在 Python 常数 | 新增 `inference.yml` 的 `api.max_live_request_bytes`，两路由共用该值；Gateway 配置兼容测试同步检查三类请求正文上限 | 符合所有可调推理参数留在推理 YAML 的项目规范，默认行为不变 |
| 2026-10-20 | Gateway v1 的路由、SSE、上游 HTTP 与视频 multipart 实现原先集中在 `src/gateway/gateway.go` | 已拆分至 `internal/route/`、`internal/stream/`、`internal/upstream/`、`internal/videoproxy/`，并同步 README 与技术设计 | 以职责边界拆分；`gateway.go` 保留应用组装与跨包业务流程，既有 API mock 测试通过 |
| 2026-10-20 | 推理代码要求聊天至少含一个 `image_url`，但原 Gateway 规格概称文字／图片聊天 | README 与规格补注现行推理契约；Gateway 不改变此推理端行为 | 以 `service.py` 实际输入整理逻辑为准，纯文字契约仍待维护者决定 |

## 完成摘要

- 已完成：Go Gateway v1、YAML 配置、API Key／租户、模型别名、聊天 JSON／SSE、视频流式上传、live 控制句柄、限流、错误、health/readiness、日志／metrics、分层测试及客户端／性能工具。
- 已完成包拆分：`route`、`stream`、`upstream`、`videoproxy` 各自承接路由分派、SSE、上游 transport 与视频 multipart 代理。
- 已执行：Go Gateway 包编译与单元测试；`tests/inference/` 静态来源检查；`tests/end2end/` mock client、性能统计与 YAML 兼容性检查。具体结果在本次交付摘要记录。
- 已执行：Qwen3-0.6B 真实 CUDA API 经 Gateway 的非流式／SSE smoke 与 3 个串行 text-sse 性能样本；报告见 `tests/qwen3_0_6b/results/qwen3-0.6b-gateway-smoke-2026-10-22.json`。
- 已执行：Qwen3-0.6B Gateway JSON／SSE；Mage-VL Gateway 图片 JSON／SSE／视频上传；Mage Live／MediaMTX 合成 RTSP 功能 smoke。Mage 图片／视频配置与量测见 [`analysis/mage_vl_4b/analysis.md`](../../../analysis/mage_vl_4b/analysis.md)，JSON 与 Live 报告分别归文件于 `tests/mage_vl_4b/results/`、`tests/qwen3_0_6b/results/`。仍未执行：未授权媒体访问／过期句柄测试、正式容量／高并发性能与部署回复演练；本次未下载模型权重。
- 当时 Gateway 任务未新增或修改算子报告；目前各模型报告位于根目录 `analysis/<model_alias>/`。
- 不代表：已选定第二个模型，或 SGLang／vLLM 已实现。
- REQ-14/T-15 仅在既有 Chat route 增加 Qwen tool-call capability；不提供专用 Agent HTTP endpoint。真实 Qwen Inference JSON／SSE 与客户端执行循环通过；Gateway 运行验收受 Application Control 阻挡，需在允许执行本地 Gateway 的环境补测。
