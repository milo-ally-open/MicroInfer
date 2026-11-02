# 【SOP】使用 GitHub Actions 验证并发布 Go Gateway

- **文档版本**：v1.0.0
- **适用对象**：第一次配置 GitHub Actions、希望从 Pull Request 一直走到 Gateway 镜像发布的研发人员
- **适用项目**：`milo-ally-open/MicroInfer`；其他仓库需替换仓库所有者、仓库名和镜像名
- **适用范围**：GitHub.com、Go Gateway、GitHub 托管的 Ubuntu Runner、GitHub Container Registry（GHCR）、Docker Compose
- **不包含**：Inference Engine、模型权重、GPU 测试、4090 自托管 Runner、生产服务器自动部署
- **对应工作流**：`.github/workflows/ci.yml`、`.github/workflows/release-gateway.yml`
- **Runner 安全规划**：[CI/CD Runner 架构与落地计划](CI_CD_Runner_Architecture_Plan.md)
- **整体流程**：配置仓库规则 → PR 自动检查 → 合并到 `main` → 推送版本 Tag → 发布 Gateway 镜像和 GitHub Release → 人工部署与验收

---

## 全文目录

1. [任务说明](#1-任务说明)
2. [前置准备](#2-前置准备)
3. [配置 GitHub 仓库](#3-配置-github-仓库)
4. [操作步骤](#4-操作步骤)
5. [结果验证](#5-结果验证)
6. [常见问题](#6-常见问题)
7. [相关资料](#7-相关资料)
8. [修订记录](#8-修订记录)

## 1. 任务说明

### 1.1 背景与目的

本 SOP 使用项目现有的 Go Gateway 作为实际示例，不另造一个与产品无关的 `2 + 3` 演示程序。读者会亲手验证代码、运行集成测试、构建容器镜像，并通过版本 Tag 将 Gateway 发布到 GHCR。

GitHub Actions 的 Runner 由 GitHub 托管，本流程不需要注册或维护 GitLab 风格的 Runner。CI 运行在一次性 GitHub 托管环境中；它不读取模型权重和仓库密钥。发布阶段只上传 Gateway 镜像，不登录部署机；部署仍由维护者人工执行。

### 1.2 完成目标

- Pull Request 中的 Gateway CI 通过格式检查、`go vet`、Go 测试和 Docker 镜像构建。
- 只有经过评审并合并到 `main` 的提交才创建版本 Tag。
- 推送符合 `vMAJOR.MINOR.PATCH` 格式的 Tag 后，发布工作流先复用 CI；全部通过后才向 GHCR 发布带版本号与提交 SHA 的镜像，并创建 GitHub Release。
- 使用指定 GHCR 镜像启动 Gateway，健康检查通过；需要回滚时能切回上一个已验收版本。

### 1.3 流程与概念

```text
Pull Request
    ↓
GitHub 托管 Runner：格式检查 → go vet → Go 测试 → Docker 构建
    ↓ 全绿并经人工审查
合并 main
    ↓ 维护者推送 vX.Y.Z Tag
复用相同 CI
    ↓ 通过
GHCR Gateway 镜像 + GitHub Release
    ↓ 人工选择版本并部署
Gateway 健康检查与 API 冒烟
```

| 概念 | 在本 SOP 中的作用 |
| --- | --- |
| Workflow | `.github/workflows/*.yml` 中定义的一条自动化流程。 |
| Job / Step | Job 是 Runner 执行的工作单元；Step 是其中按顺序执行的动作或命令。 |
| GitHub 托管 Runner | GitHub 为 Job 提供的临时 Ubuntu 环境；本流程不接触 BISG 4090。 |
| Pull Request 检查 | 合并前自动验证源代码、集成测试和容器构建。 |
| Tag / Release | Tag 指向一个明确提交；Release 为该版本生成发布记录。 |
| GHCR | GitHub Container Registry，保存本项目发布的 Gateway 镜像。 |
| `GITHUB_TOKEN` | GitHub 为工作流自动提供的短期凭证；本流程不需要个人 PAT。 |

## 2. 前置准备

### 2.1 账号与工具

| 项目 | 要求 |
| --- | --- |
| GitHub 权限 | 对 `milo-ally-open/MicroInfer` 有创建分支、Pull Request 和推送版本 Tag 的权限；仓库设置由管理员完成。 |
| 本地工具 | Git；本地复现 Go 检查时安装 Go 1.23 或兼容版本；本地构建容器时安装 Docker Desktop 并启动 Linux Engine。 |
| 部署环境 | 仅在执行手动部署时需要 Docker Compose v2、Inference 上游服务以及 Gateway 密钥。 |
| 代码状态 | 从最新 `main` 创建分支。开始前先检查并保留工作区已有改动；不要将 checkpoints、密钥、`.env` 或本地模型缓存加入提交。 |

本仓库公开可见。因此，Pull Request CI 只使用 GitHub 托管 Runner。不要将带有模型权重、凭证或共享 GPU 的持久化自托管 Runner 直接开放给本仓库的 Pull Request。

### 2.2 工作流实际做什么

`.github/workflows/ci.yml` 在推送和 Pull Request 时运行以下检查：

1. 检查 `src/gateway/` 和 `tests/integration/` 的 Go 格式。
2. 对 Gateway 模块执行 `go vet` 和 `go test -race`。
3. 对独立 Go 集成测试模块执行 `go vet` 和 `go test -race`。
4. 使用 `deploy/docker/Dockerfile` 构建 Gateway 镜像。

集成测试通过内存 HTTP Handler 检查 Gateway 路由和鉴权，不需要启动 Inference 或加载权重。它不代表图片、视频、多模态推理或 GPU 性能已经验收。

Python 模型组件和模型级 Test、Bench、Profile 由工程师按 [`tests/README.md`](../../tests/README.md) 手动执行，不属于此工作流，也不提交给 GitHub/GitLab Runner 调度；运行前需要预约 GPU 资源并检查显存负载。

`.github/workflows/release-gateway.yml` 只响应 `v*` Tag。它会再次运行同一 CI；测试通过后发布：

```text
ghcr.io/milo-ally-open/microinfer-gateway:<版本 Tag>
ghcr.io/milo-ally-open/microinfer-gateway:sha-<提交短 SHA>
```

## 3. 配置 GitHub 仓库

以下是**仓库管理员一次性设置**。普通开发者可跳到[步骤 1](#步骤-1从-main-创建工作分支)。本节只说明网页设置，不会由本 SOP 自动修改远端配置。

### 3.1 启用 Actions 并限制工作流权限

1. 打开 GitHub 仓库 `milo-ally-open/MicroInfer`。
2. 进入 **Settings → Actions → General**。
3. 确认允许仓库运行 GitHub Actions。
4. 在 **Workflow permissions** 中将默认 `GITHUB_TOKEN` 权限设为只读（`Read repository contents and packages permissions`）。
5. 不勾选“Allow GitHub Actions to create and approve pull requests”。

CI 工作流只需读取仓库。发布 Job 在 YAML 中单独申请 `contents: write` 和 `packages: write`，用于创建 Release 和推送 GHCR 镜像。

### 3.2 保护 `main`

1. 进入 **Settings → Rules → Rulesets**，创建针对分支的规则集；若账户界面仍显示旧入口，也可在 **Settings → Branches → Add branch protection rule** 配置。
2. 规则目标选择 `main`。
3. 要求 Pull Request 才能合并；禁止强制推送和删除受保护分支。
4. 添加必需状态检查：`Go Gateway CI / Go checks and image build`。
5. 保存规则，并确认自己仍有按组织规则合并代码的权限。

如果状态检查列表尚未出现，先按步骤 1 创建一个测试 Pull Request，等待 CI 至少运行一次，再返回本页添加对应检查名称。

### 3.3 保护版本 Tag

1. 在 **Settings → Rules → Rulesets** 创建针对 Tag 的规则集。
2. Tag 名称模式填写 `v*`。
3. 限制允许创建或更新此类 Tag 的人员为维护者或发布负责人。
4. 不允许覆盖已发布版本 Tag。需要修复发布时创建新的版本号，不移动旧 Tag。

这一步很重要：推送匹配的 Tag 会自动发布 GHCR 镜像和 GitHub Release。

### 3.4 首次发布后检查 GHCR 权限

GHCR 包会在第一次成功发布时创建。打开仓库 **Packages → microinfer-gateway → Package settings**：

- 公开发布：将包设为 Public，使部署机无需登录即可拉取。
- 私有发布：保留 Private；部署机需使用仅有 `read:packages` 权限的凭证登录 GHCR。
- 确认包关联到 `milo-ally-open/MicroInfer` 仓库。

CI 发布不需要手工创建 PAT，也不要把 PAT 放入仓库文件或工作流日志。

## 4. 操作步骤

### 步骤 1：从 `main` 创建工作分支

**操作位置**：仓库工作副本的 PowerShell、Git Bash 或 WSL 终端。

确保当前工作区改动已按团队流程妥善保存，然后从最新 `main` 创建分支。本例使用 `ci/gateway-smoke`；团队可按命名规范替换。

```bash
git switch main
git pull --ff-only origin main
git switch -c ci/gateway-smoke
```

**预期结果**：当前分支为 `ci/gateway-smoke`，并基于远端最新 `main`。如果 `git status` 显示不属于本次工作的本地改动，先停止并确认这些改动的归属，不要覆盖或顺手提交。

### 步骤 2：做一个最小变更并本地复现 Go 检查

**操作位置**：工作分支；可使用无运行行为影响的文档修改来验证流水线。不要为演示目的改动鉴权、限流、路由或生产配置。

提交前可以在本地复现 CI 的 Go 检查：

```bash
cd src/gateway
test -z "$(gofmt -l .)"
go vet ./...
go test -race -count=1 ./...

cd ../../tests/integration
test -z "$(gofmt -l .)"
go vet ./...
go test -race -count=1 ./...
```

若要在本机检查 Docker 构建，先确认 Docker Desktop 的 Linux Engine 已启动，然后回到仓库根目录执行：

```bash
cd ../..
docker build --pull --file deploy/docker/Dockerfile --tag microinfer-gateway:ci .
```

Windows PowerShell 没有 `test` 命令时，可在 WSL/Git Bash 执行以上代码，或按平台改写格式检查命令。CI 中运行环境为 Linux。

**预期结果**：命令退出码为 `0`。Go 模块没有单元测试文件时，`go test` 显示 `[no test files]` 代表该包成功编译；真正的 HTTP Handler 集成测试位于 `tests/integration/`。

### 步骤 3：提交并推送分支

先确认待提交文件，再只添加本次有意修改的文件：

```bash
git status --short
git add <本次修改的文件>
git diff --cached --check
git commit -m "chore(ci): verify gateway workflow"
git push -u origin ci/gateway-smoke
```

`<本次修改的文件>` 是占位文字，需替换为真实路径，不要原样复制。不要使用 `git add .` 把 checkpoints、密钥或工作区其他任务改动一并提交。

**预期结果**：GitHub 上出现 `ci/gateway-smoke` 分支，提交列表中仅包含本次演练所需改动。

### 步骤 4：创建 Pull Request 并查看 CI

**操作位置**：GitHub 仓库网页。

1. 打开仓库 **Pull requests → New pull request**。
2. `base` 选择 `main`，`compare` 选择 `ci/gateway-smoke`。
3. 检查文件差异与提交内容，确认没有权重、密钥、`.env` 或无关工作区改动。
4. 创建 Pull Request。
5. 打开 Pull Request 的 **Checks** 页签，进入 `Go Gateway CI` 检查 Job。
6. 按顺序确认格式检查、Gateway 模块检查、集成测试检查和容器构建均通过。

**预期结果**：PR 状态为通过；`main` 保护规则允许维护者合并。若失败，打开失败 Step 的日志定位原因，修复后继续推送同一工作分支，GitHub 会重新运行 CI。

### 步骤 5：评审并合并到 `main`

**操作位置**：Pull Request 页面。

完成代码审查并确认 CI 全绿后，按团队约定使用 **Squash and merge** 或普通合并。合并前检查 PR 的 base 确实是 `main`。

**预期结果**：改动进入 `main`，PR 页面记录通过的 CI 和评审结果。未通过检查或未经评审的提交不应进入发布流程。

### 步骤 6：用版本 Tag 发布 Gateway

**前置条件**：版本号已按团队发布策略确认；步骤 3.3 的 Tag 保护已生效；`main` 最新提交的 CI 通过。

从本地更新 `main`，再创建一个未使用的语义版本 Tag。以下 `v0.1.0` 仅作格式示例，实际版本需由发布负责人确定：

```bash
git switch main
git pull --ff-only origin main
git tag -a v0.1.0 -m "Release Gateway v0.1.0"
git push origin v0.1.0
```

**操作位置**：GitHub 仓库 **Actions → Release Go Gateway**。

打开由 Tag 触发的运行记录，确认 `Verify before release` 成功后才进入 `Publish Gateway image`。发布 Job 会构建镜像、推送两个 Tag，并创建带自动生成说明的 GitHub Release。

**预期结果**：Release 页面存在 `v0.1.0`；GHCR 包中存在：

```text
ghcr.io/milo-ally-open/microinfer-gateway:v0.1.0
ghcr.io/milo-ally-open/microinfer-gateway:sha-<12 位提交 SHA>
```

如果版本 Tag 格式不符合 `vMAJOR.MINOR.PATCH`，工作流会在发布前失败。不要删除或移动已发布 Tag；修正后使用新的版本号重新发布。

### 步骤 7：在目标环境手动部署并验收

本步骤属于部署操作，不由 GitHub Actions 执行。先按 [Gateway Docker 部署说明](../../deploy/docker/README.md)准备宿主机 Inference、API Key、HMAC 密钥和 `deploy/docker/` 配置。生产环境部署前先在测试环境验证同一镜像。

在部署机的 `deploy/docker/.env` 中填写已验收镜像：

```dotenv
GATEWAY_IMAGE=ghcr.io/milo-ally-open/microinfer-gateway:v0.1.0
```

若 GHCR 包为 Private，先交互式登录，避免把 PAT 写进命令历史：

```bash
docker login ghcr.io --username YOUR_GITHUB_USERNAME
```

然后拉取并启动固定版本镜像：

```bash
cd deploy/docker
docker compose pull gateway
docker compose up --no-build -d
docker compose ps
docker compose logs --tail=100 gateway
```

从部署机执行健康检查：

```bash
curl --fail http://127.0.0.1:8000/healthz
curl --fail http://127.0.0.1:8000/readyz
curl --fail http://127.0.0.1:8000/metrics
```

`/healthz` 检查 Gateway 进程；`/readyz` 还要求配置中的 Inference 上游可用。若要验证真实聊天请求，再按仓库根目录 `client.py` 设置 Gateway API Key 并执行调用。不要在 SOP、日志或截图中记录真实密钥。

### 步骤 8：回滚到上一个已验收版本

如果新镜像部署后验收失败，先恢复 `.env` 中上一个已验收的 `GATEWAY_IMAGE`，再重新拉取并重建容器：

```bash
cd deploy/docker
docker compose pull gateway
docker compose up --no-build -d
docker compose ps
curl --fail http://127.0.0.1:8000/healthz
```

回滚不需要移动 Git Tag，也不需要重新构建旧版本。记录故障版本、回滚版本、时间和检查结果，后续修复通过新 Tag 发布。

## 5. 结果验证

| 阶段 | 必须看到的结果 |
| --- | --- |
| 仓库规则 | `main` 要求 PR 与 CI；`v*` Tag 仅允许发布负责人创建或更新。 |
| PR CI | 格式、`go vet`、Go 测试和 Docker 构建全部通过；没有模型权重或密钥进入工作流。 |
| 版本发布 | `vX.Y.Z` Tag 运行发布工作流；CI 通过后有同版本 GHCR 镜像和 GitHub Release。 |
| 部署验收 | Gateway 容器运行；`/healthz` 返回成功；配置正确且 Inference 正常时 `/readyz` 成功。 |
| 回滚 | 将镜像切回此前验收版本后容器重新启动，健康检查通过。 |

完成一次真实发布后，记录仓库 Tag、GitHub Actions 运行链接、Release 链接、镜像 Tag、部署环境、验收时间和回滚版本。不要把未执行的部署步骤标记为已完成。

## 6. 常见问题

| 现象 | 检查方法 |
| --- | --- |
| PR 没有 CI 检查 | 确认 `.github/workflows/ci.yml` 已合并到该分支，Actions 已启用，PR 的 base 是 `main`。 |
| Go 检查失败 | 打开失败 Step 的完整日志；本地分别在 `src/gateway/` 和 `tests/integration/` 复现 `gofmt`、`go vet` 和 `go test`。 |
| Docker 构建找不到文件 | 检查 Docker build context 是否为仓库根目录，以及 `deploy/docker/.dockerignore` 是否保留 Gateway 源码与 `gateway.yml`。 |
| 发布工作流没有触发 | Tag 必须推送到远端并以 `v` 开头；发布步骤只接受 `v数字.数字.数字`。 |
| GHCR 推送报权限错误 | 检查发布 Job 的 `packages: write`、仓库 Actions 权限和包策略；CI Job 不应为了修复此问题而提升权限。 |
| 部署机拉取私有镜像失败 | 使用仅有 `read:packages` 的凭证登录，并确认该账号有包读取权限。 |
| `/healthz` 成功但 `/readyz` 失败 | Gateway 进程仍在运行；检查 Gateway 配置中的 Inference 地址、端口和上游健康状态。 |
| 本机 Docker 构建无法连接 Engine | 启动 Docker Desktop 的 Linux Engine；GitHub Hosted Runner 上的构建不依赖开发机 Docker 服务。 |

## 7. 相关资料

| 资料 | 用途 |
| --- | --- |
| [GitHub Actions 工作流语法](https://docs.github.com/en/actions/writing-workflows/workflow-syntax-for-github-actions) | 查询触发器、权限、Job 和 Step 配置。 |
| [GitHub Actions 安全使用参考](https://docs.github.com/en/actions/reference/security/secure-use) | 理解公开仓库与自托管 Runner 的安全风险。 |
| [GitHub Runner 访问控制](https://docs.github.com/en/actions/how-tos/manage-runners/self-hosted-runners/manage-access) | 配置 Runner Group 与仓库访问策略。 |
| [使用 GitHub Actions 发布容器包](https://docs.github.com/en/actions/tutorials/publish-packages/publish-docker-images) | 理解 Actions 向 GHCR 发布镜像的权限和步骤。 |
| [GitHub Container Registry 权限](https://docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-container-registry) | 配置包可见性、仓库关联和下载权限。 |
| [Go Gateway 部署说明](../../deploy/docker/README.md) | 查看 Compose、密钥、健康检查和宿主机上游要求。 |
| [Go Gateway 测试说明](../../tests/README.md) | 查看 Go 集成测试与手动 Gateway 冒烟的边界。 |

## 8. 修订记录

| 版本 | 日期 | 更新说明 |
| --- | --- | --- |
| v1.0.0 | 2026-11-02 | 为 MicroInfer Go Gateway 编写 GitHub Actions CI、GHCR 发布及人工部署的完整操作 SOP。 |
