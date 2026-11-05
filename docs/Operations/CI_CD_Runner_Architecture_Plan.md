# CI/CD 架构与操作说明

- **适用仓库**：MicroInfer（Python FastAPI 推理服务）
- **自动化平台**：GitHub Actions；当前未配置 GitLab CI
- **执行资源**：GitHub 托管 Ubuntu Runner；不使用 4090 自托管 Runner
- **状态**：工作流定义在 `.github/workflows/ci.yml`；首次运行和仓库规则配置仍需在 GitHub 验收

## 1. 当前仓库的边界

旧 Go Gateway、Go 集成测试、Gateway Docker 镜像和专属发布流程已从工作区移除。当前对外推理入口是 Python FastAPI；模型权重和 GPU Runtime 安装在推理宿主机，不打包为推理 Docker 镜像。

Python 模型组件和模型级 Test、Bench、Profile 仍由工程师手动排期执行，因为这些工作依赖本地权重、CUDA 和共享显存。CI 仅运行无权重、无 GPU 的语法检查、模型 manifest/YAML 预检和 Python 包构建；不运行 pytest 或模型推理。

## 2. 流水线

| 流程 | 触发 | 工作内容 | 交付/结果 |
|---|---|---|---|
| CI | PR 到 main、push main、v* tag、手动 workflow_dispatch | Ubuntu 24.04；Python 3.10/3.12；compileall；launcher validate；构建 wheel 与 sdist | 两个 Python 版本均通过才可合并或发布 |
| CD（持续交付） | 推送 v* tag 且 CI 通过 | 检查 tag 与包版本一致；构建 wheel、sdist 和 SHA256SUMS | 创建 GitHub Release 并附上制品 |
| 生产部署 | 当前不自动触发 | 运维人员将已审查版本部署到指定推理主机 | 手动维护窗口、GPU/权重预检和服务健康验收 |

CI 的 `inference validate` 只读模型 manifest 与配置，不加载权重、不初始化 CUDA。模块级测试及模型级 Test/Bench/Profile 按各模型的 `tests/<model_name>/README.md` 手动执行。

CD 目前是持续交付而非持续部署：没有已确认的生产主机、制品仓库、部署凭证、服务管理方式或健康回滚接口，因此工作流不会登录服务器、重启推理服务或覆盖模型权重。确认目标环境后，再单独设计带审批、最小权限和回滚的部署 job。

## 3. GitHub 工作流安全

- 所有 Job 在 GitHub 托管 Runner 运行；公开 PR 不接触持久化自托管主机、GPU、模型权重、Secrets 或生产网络。
- 工作流仅监听 `pull_request`、`push` 和手动触发，不使用 `pull_request_target` 执行 PR 代码。
- 默认 `GITHUB_TOKEN` 仅有 `contents: read`；只有 Release Job 获得 `contents: write`。PR 校验不需要仓库密钥。
- 官方 Actions 固定到完整 commit SHA，checkout 禁止持久保存 Git 凭据；升级 SHA 时审查官方 release 和变更记录。
- Release 只接受版本号与包版本一致的 `vX.Y.Z` tag；tag 应由维护者从受保护的 main 发布。
- GitHub Release 只含 Python 源码包、wheel 与校验文件；不含模型权重、密钥、私有配置或运行日志。

这些设置遵循 GitHub 关于最小化 GITHUB_TOKEN 权限、固定 Actions SHA 和隔离公开仓库 Runner 的安全建议。

## 4. 仓库设置与发布操作

### 首次启用前

1. 在 Settings → Actions → General 中将 Workflow permissions 设为只读；不要允许 Actions 创建或批准 Pull Request。
2. 为 main 配置 Ruleset/Branch protection：要求 PR、至少一人审查，并要求 CI 的 Python 3.10 与 Python 3.12 两项检查通过后才能合并。
3. 为 `v*` tag 配置规则，仅允许维护者创建或更新；禁止删除已发布 tag。发布 tag 不可重用。
4. 确认仓库 Actions policy 允许本工作流中 SHA 固定的 GitHub 官方 Actions。

### 发布版本

1. 将版本号更新到 `src/inference/__init__.py` 的 `__version__`，并审查依赖、配置和变更记录。
2. 合并到 main，等待两个 Python 版本的 CI 检查通过。
3. 维护者在该提交创建匹配 tag，例如包版本为 0.1.0 时使用 `v0.1.0`。
4. 检查 Actions 的 CI 与 Release Job 均成功；在 GitHub Release 下载 wheel、sdist 和 SHA256SUMS 并核对校验值。
5. 在目标推理主机手动准备匹配的 Python/CUDA 环境、按模型安装依赖，并准备模型权重后启动服务、进行健康验收；通过网络策略限制 API 访问。

失败的 tag 发布不覆盖已有 Release。修复后应使用新版本号和新 tag 发布，避免移动已发布版本标记。

## 5. 4090 与后续 GitLab

4090 是组内 GPU 资源，不是当前 MicroInfer CI Runner。公开仓库 PR 不能触发它；模型 Test、Bench、Profile 继续按各模型的 `tests/<model_name>/README.md` 协调 GPU 后手动执行。共享 Runner 如用于组内其他私有可信任务，应由独立私有基础设施仓库管理，使用隔离执行器、资源配额和跨平台互斥；不能把 Runner 注册令牌或宿主机管理权限写入本仓库。

GitLab CI 尚未建立。后续迁移应按同一边界重写 `.gitlab-ci.yml`，不能复制旧 Go 流水线，也不能把 GitHub Runner token 复用为 GitLab Runner 凭证。

## 6. 责任边界与人工验收

| 对象 | 自动化责任 | 人工责任 |
|---|---|---|
| GitHub Actions | CPU 静态检查、配置预检、包构建和 Release 制品 | 审查 workflow 改动，保护 main/tag 规则 |
| Python 模型验证 | 不运行 GPU/权重测试 | 按预约执行组件/模型 Test、Bench、Profile 和 API smoke；记录硬件、revision、commit 与结果 |
| 推理环境 | 不自动 SSH、不重启服务、不修改权重 | 维护者按目标主机流程安装、启动、验收和回滚 |
| 4090/GitLab | 当前未接入 MicroInfer 流水线 | 由共享基础设施负责人单独评估和建设 |

## 7. 来源

- [GitHub Actions secure use reference](https://docs.github.com/en/actions/reference/security/secure-use)
- [GitHub Actions workflow syntax and token permissions](https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-syntax)
- [actions/checkout v7.0.1 release](https://github.com/actions/checkout/releases/tag/v7.0.1)
- [actions/setup-python v7.0.0 release](https://github.com/actions/setup-python/releases/tag/v7.0.0)

## 8. 修订记录

| 日期 | 更新 |
|---|---|
| 2026-11-03 | 按 Python FastAPI 当前架构替代旧 Go Gateway CI/CD 计划；定义 CPU CI、GitHub Release 持续交付和人工 GPU/部署边界。 |
