# CI/CD Runner 架构与落地计划

- **状态**：规划阶段；未安装、注册或启动服务器 Runner
- **适用项目**：MicroInfer 首发试点；后续供组内项目复用
- **目标平台**：GitHub Actions 与 GitLab CI/CD
- **目标硬件**：BISG 4090 服务器
- **原则**：先建立可信代码通道和隔离边界，再接入共享 GPU；不把 Runner 注册凭证、应用密钥或模型权重放入 Git。

## 1. 目标和当前判断

本项目先验证 Go Gateway 的 GitHub CI/CD：Pull Request 由 GitHub 托管 Runner 执行检查；版本 Tag 触发 Gateway 镜像发布；部署仍由维护者人工执行。该链路不需要自建 Runner，也不需要 4090。

Python 模型组件测试以及模型级 Test、Bench、Profile 由工程师在已分配的 GPU 时段手动运行，明确不进入 GitHub/GitLab Runner。其原因是权重和 CUDA 环境依赖本机资源，且需要与在线推理服务及其他实验协调 GPU/显存，不能由通用 CI 队列自动并发调度。具体操作见 [`tests/README.md`](../../tests/README.md)。

4090 Runner 是独立的共享基础设施。MicroInfer 的 Python 模型测试不会交给它调度；如未来组内其他非模型任务要使用该 Runner，仍须单独满足可信代码、权限隔离和资源控制要求。它应放在团队控制的基础设施范围内，不应和公开应用仓库的 Pull Request 工作流直接绑定。

当前 MicroInfer 仓库在 GitHub 上公开。GitHub 官方说明，公开仓库的 Pull Request 可能运行攻击者提交的任意工作流代码；持久化自托管 Runner 可能被留存式攻陷。因此：

- MicroInfer 的公开 PR 和版本发布继续使用 GitHub 托管 Runner。
- 不在 MicroInfer 公开仓库配置 4090 自托管 Runner，不给公开 PR 暴露 Runner 标签。
- 4090 GPU 自动化先采用团队组织下的私有控制仓库或其他私有可信入口，只运行经过评审的固定提交 SHA。
- 若无法建立可信入口和隔离执行环境，GPU 验收保持人工操作，不以放宽安全边界换取自动化。

## 2. Runner、平台和执行器的边界

| 对象 | 作用 | 本计划中的实现 |
| --- | --- | --- |
| GitHub Actions Runner | 执行 GitHub Workflow Job 的代理程序 | GitHub 托管 Runner 承担公开项目 CI；4090 若启用，使用独立受限的 GitHub 自托管 Runner 池。 |
| GitLab Runner | 领取 GitLab CI Job 并调用 Executor 执行 | 如团队要求 GitLab 也用 4090，单独安装、注册和维护 GitLab Runner。 |
| Executor | 创建任务实际运行环境的方式 | 首选一次性容器或虚拟机；不让普通 Job 直接在宿主机 Shell 执行。 |
| 工作流 | 声明要执行的检查、构建和发布步骤 | GitHub 使用 `.github/workflows/*.yml`；GitLab 使用 `.gitlab-ci.yml`。 |
| BISG 4090 主机 | GPU 执行资源 | MicroInfer Python 模型测试由工程师手动排期，不进入 Runner 队列；若其他获批任务共用主机，GitHub 与 GitLab 队列必须共享主机级互斥锁。 |

GitHub Runner 和 GitLab Runner 是两种不同的代理程序，不能共用注册配置或认证令牌。可以共用同一台服务器和 GPU，但必须隔离服务账号、配置目录、工作目录、日志和令牌；还要用主机级并发控制，避免两个平台同时占用显卡。

## 3. 安全架构

### 3.1 代码信任边界

1. **公开代码验证**：MicroInfer 的普通 PR 运行在 GitHub 托管的临时机器上，只给只读 `GITHUB_TOKEN`，不注入仓库密钥或服务器凭证。
2. **4090 任务入口**：使用团队 GitHub Organization 下的私有控制仓库或等价私有 CI 项目管理 GPU Job。Runner Group 仅授权该控制仓库和经过批准的受信任项目。
3. **固定提交**：GPU Job 输入必须记录并验证仓库地址、完整提交 SHA、分支保护状态和审批记录；禁止直接执行任意 PR 分支名或外部 fork 代码。
4. **执行审批**：生产权重或高权限资源的任务放在需审核的环境中；审批只控制是否开始，不能替代容器隔离和宿主机清理。
5. **最小权限**：Job 不获得 Runner 管理令牌、GitHub App 私钥、GitLab Runner 身份令牌、服务器 SSH 私钥或 Docker 管理 socket。

私有仓库能缩小攻击面，但不自动使代码可信。任何允许触发 Runner 的写权限人员和工作流修改者都属于信任边界。

### 3.2 4090 执行隔离

- Runner 管理服务与 Job 容器使用不同身份；Job 容器不挂载 `/var/run/docker.sock`，不使用 `privileged`，不共享 Runner 管理目录。
- 如未来获批非模型 GPU Job，使用一次性工作目录和容器；完成、失败、超时或取消后清除临时容器和该 Job 工作区。清理范围必须固定在专用 Runner 根目录内。
- Runner Job 默认不挂载模型权重、Inference 缓存或服务密钥；MicroInfer 模型验证在 CI 外由工程师按资源预约手动执行。
- 如其他获批 GPU Job 使用该主机，任务并发设为 1；GitHub 与 GitLab Runner 通过同一个宿主机锁协调，不能各自仅依赖平台内的 `concurrency`。
- 只开放必要网络：访问对应 Git 服务、依赖镜像与受控制品源；阻止访问宿主机管理端口及不相关内网服务。
- 限制 Job 的运行时长、CPU、内存、临时磁盘和日志大小；不接受任意 Docker image、任意服务容器或由用户传入的宿主机路径。
- 主机日志、Runner 作业日志和审计事件保留在管理员控制的位置；不得在日志中打印认证令牌、密钥或模型输入敏感内容。

隔离强度要根据服务器实现能力验证。若无法保证一次性执行环境、工作区清理和网络限制，4090 不接收来自共享项目的自动化 Job。

### 3.3 Runner 凭证与更新

- Runner 注册令牌只用于注册，存放在服务器受限目录或密钥管理系统，不进入仓库、Compose 文件、Shell 历史和日志。
- GitHub 侧优先使用短期、限权的组织 Runner 注册流程；若使用 JIT Runner，还需保护负责创建 JIT 配置的管理凭证。
- GitLab 侧按当前 GitLab 版本创建 Runner，并使用新的 Runner authentication token；旧式 registration token 已弃用，不照搬示范 SOP 的旧流程。
- 固定 Runner 软件版本和基础镜像版本，制定更新窗口、回滚方案及令牌轮换周期；禁止生产配置使用不受控的 `latest`。
- GitHub Actions 第三方 Action 固定到完整提交 SHA；优先使用 GitHub 官方 Action，并通过 Dependabot 或维护窗口审查升级。

## 4. 参考 SOP 的审查结论

参考材料 `repos/gitlab-ci-release-sop-dev/sop/GitLab_CICD实施SOP.md` 提供了从私有项目、Runner 注册、smoke job、Go 检查/构建/打包到 Tag Release 的逐步示例。它适合作为**文档组织方式**和**GitLab 13.8.8 环境演示**参考，但不能不经修改直接用于共享 4090：

| 参考内容 | 当前评估 | 落地要求 |
| --- | --- | --- |
| 使用 Go 最小项目演示 CI | 流程直观，适合学习 Stage、Job、Artifacts 和 Tag | MicroInfer 试点直接验证真实 Gateway，不增加无关 `2 + 3` 示例。 |
| Docker Compose 部署 Runner | 适用于自托管 GitLab Runner | 先审查 Docker socket、Executor 和工作区隔离，再用于共享 GPU 主机。 |
| `.env` 中的旧 registration token | 与 SOP 指定 GitLab 13.8.8 版本相关 | 当前 GitLab 使用新的 Runner 创建与认证流程；不要把旧令牌教程当作现代默认步骤。 |
| Runner 标签匹配 `.gitlab-ci.yml` | 有助于把 Job 路由到指定能力池 | 4090 标签仅授予私有可信项目；公开 PR 不得得到该标签。 |
| Tag 触发 Release | 适合作为交付演示 | 保护 `v*` Tag，发布前强制验证和评审；生产部署单独审批。 |
| 截图展示 GitLab 操作页面 | 对具体实例有帮助 | GitHub 与 GitLab 页面不同，MicroInfer SOP 使用各平台自己的路径和真实验收截图，不复用错误平台截图。 |

参考 Docker Compose 挂载了 Docker 管理 socket。该权限可让 Runner 管理服务控制宿主 Docker，因此不能让不受信任的 Job 共享这个接口。需要 Docker 构建时，应选取专门评估过的隔离构建方式，不默认给 GPU 执行 Job 提供宿主 Docker 权限。

## 5. 分阶段实施计划

| 阶段 | 工作 | 通过条件 | 不通过时 |
| --- | --- | --- | --- |
| 0. 基线与变更保护 | 记录 Git 状态、当前分支和 Runner 主机现况；备份配置；确认没有清理现有容器、权重或用户数据的动作 | 变更范围、回滚方式、主机负责人和维护窗口都明确 | 不改主机，不清理磁盘，不覆盖配置 |
| 1. MicroInfer GitHub CI | 固定 Go Gateway CI；通过 PR 验证格式、`go vet`、集成测试和 Docker build | PR 检查成功，权限只读，未接触自托管 Runner | 留在 GitHub 托管 Runner，修复后再进下一阶段 |
| 2. Gateway GHCR 交付 | 对受保护版本 Tag 执行相同 CI，发布版本与 SHA 镜像；部署继续人工执行 | 镜像来源、标签、权限和回滚版本均可追溯 | 撤销或修正发布流程，不自动部署服务器 |
| 3. 4090 主机只读盘点 | 核验系统版本、GPU/驱动、容器运行时、磁盘、网络、现存 GPU 任务、模型权重位置和备份状态 | 盘点不影响现有服务；资源与隔离方式满足限制 | 停止部署并提交风险项，不执行安装或重启 |
| 4. 私有 Runner 控制面 | 确认 GitHub Organization/Runner Group、GitLab 实例及可用 Runner 权限；配置仓库 Allowlist、分支/Tag 保护和审批 | Runner 只能被列明的私有可信项目触发；公开 MicroInfer PR 不可达 | 不注册共享 Runner |
| 5. 隔离执行器 | 为单独批准的 Runner 任务建立一次性工作区、网络/资源限制、日志和清理策略；不提供模型权重挂载 | 恶意 Job 模拟、超时、取消、重试和清理验收通过 | 4090 不承载自动化任务 |
| 6. 主机盘点验收 | 维护窗口内由管理员只读检查 `nvidia-smi`、驱动、磁盘、服务和 GPU 进程；不从 CI Runner 启动 Python/CUDA 测试 | 主机资源和服务状态有记录；确认无在途 GPU 任务后再制定后续安排 | 停止安装/注册，不影响现有服务 |
| 7. Go Gateway 首发验收 | GitHub 托管 CI 只完成 Go Gateway 检查和镜像构建；Python 模型测试按 `tests/README.md` 由工程师预约 GPU 后手动执行 | Go CI 和手动模型验证分别有独立、可追溯的验收记录 | Go CI 继续留在托管 Runner，模型测试保持人工排期 |
| 8. 组内推广 | 登记项目所有者、代码信任级别、运行镜像、资源配额和联系人；分项目逐个接入 | 每项目先过安全审查与 smoke test；跨平台 GPU 锁有效 | 不扩大 Runner Group Allowlist |

每一阶段只在前一阶段验收通过后推进。不得将“Runner 在线”作为安全验收；必须证明权限边界、恶意输入隔离、资源限制和任务后清理有效。

## 6. 4090 主机变更安全流程

### 6.1 变更前

1. 取得明确的服务器连接方式、操作账号、主机负责人和维护窗口；凭证通过受控渠道提供，不粘贴进仓库或文档。
2. 只读记录 `nvidia-smi`、操作系统、驱动、Docker/Container Toolkit 版本、磁盘空间、已运行容器、正在使用 GPU 的进程及相关服务状态。
3. 对已有 Runner、Docker 配置、模型权重和服务配置做可恢复备份；记录文件权限、所有者和校验值。
4. 明确待创建路径与服务名；通过绝对路径检查确认不覆盖当前服务或仓库数据。
5. 将变更步骤、回滚步骤和预期影响写入变更记录，安排维护窗口和通知对象。

### 6.2 变更中

- 先在隔离目录部署，不覆盖现存配置；先做配置校验，再启动单个 Runner 实例。
- 每次只做一项变更；记录命令、时间、退出码和日志位置。
- 不执行宽泛 `docker system prune`、递归删除、清空缓存、重装驱动或重启宿主机等动作，除非有单独审批、备份和影响评估。
- 不使用来源不明的容器镜像或安装脚本；使用官方仓库并固定可审计版本。
- 若观察到意外服务退出、GPU 任务被中断、权限扩大、异常外联或工作区清理超范围，立即停止后续步骤，保全日志并回滚已知安全变更。

### 6.3 变更后

- 验收平台在线状态、作业来源、Runner 标签、并发限制、任务隔离和清理结果；确认 Runner Job 默认看不到 GPU 设备、模型权重和 Inference 缓存。
- 确认现有推理服务、模型权重、容器和端口保持预期状态。
- 轮换试运行凭证，删除不再需要的注册材料，更新资产清单与维护人。
- 没有实际运行记录的验收项标记为“未执行”，不以配置文件存在代替上线验收。

## 7. 当前阻塞项与开工输入

在 4090 主机实际安装或注册 Runner 之前，必须确认以下信息。只需要提供非秘密的资源/组织信息；令牌和私钥通过服务器密钥管理渠道设置，不要发送到聊天或 Git：

| 信息 | 为什么需要 |
| --- | --- |
| GitHub 组织名称及组织管理员/Runner Group 管理人 | 当前 MicroInfer 属于公开个人空间；共享组织 Runner 需要明确组织边界和授权仓库。 |
| 是否建立私有 Runner 控制仓库（推荐） | 避免公开仓库 PR 将任意代码送到 4090。 |
| GitLab 实例版本、地址及 GitLab Runner 归属范围 | 决定使用新的 Runner 认证流程、组/项目范围和兼容 Executor。 |
| 4090 服务器可用连接入口、操作系统与负责人 | 当前仓库工作区没有服务器执行会话；无法安全验证现存服务和变更窗口。不要在聊天粘贴密码或私钥。 |
| GPU 权重所在目录、只读挂载需求、可用磁盘与网络策略 | 设计隔离 Job 容器且避免复制或误删权重。 |
| GitHub 与 GitLab 是否要共用同一 GPU 主机 | 若共用，必须实施跨平台互斥锁和资源配额。 |

未确认这些开工输入前，不生成或保存实际 Runner 注册凭证，不在服务器安装/重启服务，不配置公共仓库的自托管 Runner，也不清理服务器空间。

## 8. 当前项目的对应文件

| 文件 | 责任 |
| --- | --- |
| `.github/workflows/ci.yml` | MicroInfer Go Gateway 的托管 Runner 持续集成。 |
| `.github/workflows/release-gateway.yml` | 受保护 Tag 的 Gateway 镜像与 Release 交付。 |
| `docs/Operations/GitHub_GoGateway_CICD_SOP.md` | GitHub 页面设置、PR、GHCR 发布、手动部署和回滚实操。 |
| `deploy/docker/` | 仅部署 Go Gateway 的容器文件；Inference 与模型权重留在宿主机。 |
| `tests/integration/` | 独立 Go Gateway HTTP 集成测试。 |

Runner 主机的注册令牌、服务状态、机器级锁与运行日志属于基础设施资产，不应放在 MicroInfer 应用仓库。实际 Runner 部署包应进入团队私有的基础设施仓库，并由明确的资产所有者维护。

## 9. 官方资料

- [GitHub Actions 安全使用参考](https://docs.github.com/en/actions/reference/security/secure-use)
- [GitHub 自托管 Runner 访问控制](https://docs.github.com/en/actions/how-tos/manage-runners/self-hosted-runners/manage-access)
- [GitHub 自托管 Runner 参考与临时 Runner](https://docs.github.com/en/actions/reference/runners/self-hosted-runners)
- [GitLab Runner 新注册流程](https://docs.gitlab.com/runner/register/)
- [GitLab Token 与 Runner 认证](https://docs.gitlab.com/security/tokens/)
- [GitLab Docker Executor](https://docs.gitlab.com/runner/executors/docker/)

## 10. 修订记录

| 版本 | 日期 | 更新说明 |
| --- | --- | --- |
| v1.0.0 | 2026-11-02 | 汇总 GitHub/GitLab 与 BISG 4090 共享 Runner 的边界、阶段、威胁防护及开工条件。 |
