# 文档导航

本文按读者任务整理仓库文档，并标明各文档的用途与事实来源。改动代码或配置后，应同步更新描述该行为的文档；不要只改索引而留下内容不一致的操作说明。

## 从哪里开始

| 需求 | 首选文档 | 内容 |
| --- | --- | --- |
| 了解项目并启动服务 | [根 README](../README.md) | 项目边界、已登记模型、安装、启动和文档入口 |
| 从 Hugging Face 接入一个新模型 | [新模型接入流程](../TUTORIAL.md) | 固定 revision、下载快照、模型包目录、依赖、配置、Runtime 及分层验收 |
| 贡献代码或修改文档 | [仓库维护约定](../AGENTS.md) | 架构边界、注释和文档写法、验证证据与安全要求 |
| 直接运行 Python 推理 API | [Inference README](../src/inference/README.md) | FastAPI、模型插件、安装和服务端用法 |
| 查看当前 CI/CD 和 Runner 边界 | [CI/CD 架构与操作说明](Operations/CI_CD_Runner_Architecture_Plan.md) | Python CPU 校验、GitHub Release 制品、手动 GPU 验收及未来共享 Runner 边界 |
| 运行模型组件、性能或端到端验收 | 对应模型的 `tests/<model_name>/README.md` | 该模型的测试前置条件、运行命令、产物和验收要求；入口链接见根目录 README |
| 部署实时媒体 sidecar | [MediaMTX README](../deploy/mediamtx/README.md) | 独立部署、回调配置和模型网络端点的对应关系 |

## 设计与接口规格

`Specifications/` 保存架构决策、接口要求和实施记录，供需要理解约束由来或审查变更时使用。

| 子系统 | 设计文档 | 适用范围 |
| --- | --- | --- |
| Inference | [提案](Specifications/inference/proposal.md)、[技术设计](Specifications/inference/design.md)、[接口规格](Specifications/inference/spec.md)、[实施记录](Specifications/inference/tasks.md) | Engine、模型 Runtime、API 契约和能力声明；新模型接入按根目录 [流程文档](../TUTORIAL.md) 执行 |
| Cloud VLM 控制面（规划） | [提案](Specifications/control-plane/proposal.md)、[行为规格](Specifications/control-plane/spec.md)、[技术设计](Specifications/control-plane/design.md)、[实施任务](Specifications/control-plane/tasks.md) | 未来独立产品后端与 MicroInfer 的对接；不包含在本仓库实现 |

这些规格包含历史方案和阶段性验收记录。它们解释决策与当时的验证结果，不自动代表当前实现；涉及端口、字段、能力和限制时，先核对当前代码与 YAML。历史测试结果应保留其模型 revision、硬件、配置和采集日期，不要当作当前性能承诺。

## 实测分析

| 模型 | 报告 | 证据表 |
| --- | --- | --- |
| Qwen3-VL-2B | [analysis.md](../analysis/qwen3_vl_2b/analysis.md) | [analysis.xlsx](../analysis/qwen3_vl_2b/analysis.xlsx) |
| Mage-VL-4B | [analysis.md](../analysis/mage_vl_4b/analysis.md) | [analysis.xlsx](../analysis/mage_vl_4b/analysis.xlsx) |

Markdown 报告说明问题、解决方案和效果；XLSX 保存由脚本产生的原始数值证据。报告中的历史实验、被撤销的实现和当前可用能力应明确区分。

## 维护约定

- `README.md` 负责快速识别项目、前置条件和最短启动路径；详细流程放到对应专题文档。
- 操作说明要给出执行目录、命令、配置入口、必需环境变量及成功判据。模型权重、密钥和媒体素材不进入仓库。
- 设计规格记录边界和原因；当前行为以代码与配置为准，测试结果以实际运行证据为准。
- 相对链接优先指向仓库内文件。移动或重命名文档时，检查根 README、专题 README、索引和教程中的引用。
