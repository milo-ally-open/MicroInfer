# 贡献指南

欢迎提交缺陷修复、模型接入、性能分析和文档改进。提交前先确认改动所属边界，并提供能复核的行为说明和验证证据。

## 开始修改前

1. 阅读[项目 README](README.md)和[文档导航](docs/README.md)，再查看相关子系统的 README、模型配置及现有实现。
2. 确认修改属于 Gateway、Inference 共用层、单一模型包、部署配置还是分析工具。优先在拥有该职责的目录中实现，避免把模型专属逻辑放入公共 Engine 或 Gateway。
3. 检查工作树中的现有修改，保留与当前任务无关的用户改动。

## 工程边界

- Gateway 负责认证、租户策略、能力检查、模型路由和 HTTP/SSE 转发；它不加载模型、不执行工具，也不转发 RTSP/RTMP 媒体。
- Inference 共用层负责 API、请求生命周期、调度和 Worker 通信。Tokenizer、模型拓扑、权重适配和专属 Cache 放在 `src/inference/models/<model>/`。
- 每个模型通过 `configs/model.yml` 声明真实能力，通过自己的 `inference.yml` 和 `network.yml` 配置运行参数。依赖按模型 extra 安装，不把其他模型的依赖带入安装集合。
- 新增能力时同步更新实现、模型 manifest、必要的 Gateway 路由、测试和对应文档。声明为支持的行为必须有与该层匹配的验证证据。
- 推理和模型前向优先使用 CUDA；仅当 CUDA 不可用时才回退到 CPU。不要以 CPU 运行结果声称 GPU 性能。
- 不提交权重、模型缓存、密钥、Token、用户图片或视频。日志和报告不得包含凭据或不必要的请求内容。

## 文档与注释

- 面向开发者的仓库文档、Python docstring、代码注释和 CLI 中文说明使用简体中文；标识符、命令、API 字段、模型名和上游术语保留原文。
- 根 README 只承载项目概览和快速开始。按主题维护 Inference、Gateway、Tests、Scripts 和部署说明；模型接入步骤集中在根目录 `TUTORIAL.md`，文档用途见 [`docs/README.md`](docs/README.md)。
- 操作步骤要可复制：标出执行目录、配置路径、依赖、所需环境变量和预期结果。配置字段、端口、别名和 API 能力必须与当前代码一致。
- 对外提供的 Python 模块、类和函数使用准确、简洁的 docstring。按 PEP 257 的格式组织；说明调用者需要知道的行为、输入/输出约束、异常、资源生命周期或副作用，不重复函数名和实现步骤。内部小型辅助函数和数据字段不写重复性说明。
- 行内注释解释不明显的原因、不变量、张量维度/布局、缓存所有权或安全边界；不要逐行复述代码。算子注释应注明 shape、dtype、device 和必要的数值假设。
- Go 导出标识符使用紧邻声明的文档注释，并以标识符名称开头。YAML 注释只解释单位、默认值、约束或设计原因；配置值本身仍由配置和代码定义。
- 修改接口或配置契约时，同步维护规格和使用文档；记录尚未验证的内容，不把历史测试、计划或局部 Profile 写成当前已验收能力。

## 验证与提交说明

根据改动范围运行相应验证。以下命令从仓库根目录执行：

```bash
inference list
inference validate --model qwen3-0.6b
python tests/qwen3_0_6b/test_components.py --config tests/qwen3_0_6b/test.yml
```

`validate` 是不加载权重的静态预检；组件、性能和端到端测试覆盖不同层次，不能相互替代。性能测试需要本地权重、可用 GPU 和运行中的 Inference 服务；端到端测试还需要 Gateway、API Key 和相应素材。Go 变更应在 `src/gateway/` 下运行 `go test ./...`。若本次没有运行某项验证，提交说明中应明确写出。

提交说明建议包含：

- 改动目的和影响的子系统/模型。
- 配置、API 或用户可见行为的变化。
- 实际执行的验证命令和结果，以及未验证的范围。
- 对性能数据提供模型 revision、GPU、软件版本、配置和样本数。

不得把 smoke、单层算子 Profile 或少量计时样本表述为数据集精度、容量保证或 SLA。

## 参考规范

- [GitHub README 指南](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-readmes)：让 README 说明项目用途、价值、快速开始和帮助入口。
- [GitHub 贡献指南](https://docs.github.com/en/communities/setting-up-your-project-for-healthy-contributions/setting-guidelines-for-repository-contributors)：把贡献流程放在仓库可发现的位置。
- [PEP 257](https://peps.python.org/pep-0257/)：Python docstring 的结构与约定。
- [Go Doc Comments](https://go.dev/doc/comment)：Go 包和导出声明的文档注释规范。
- [Google Python Style Guide：Comments and Docstrings](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)：注释应提供实现本身看不出的上下文。
