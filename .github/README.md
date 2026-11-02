# GitHub Actions：Go Gateway

本目录的 GitHub Actions 只验证与交付 Go Gateway。Inference Engine、CUDA、模型权重和 4090 GPU Runner 不进入公开仓库的 Pull Request 工作流。

按仓库网页设置、创建 PR、发布版本和手动部署的逐步 SOP，见[《使用 GitHub Actions 验证并发布 Go Gateway》](../docs/Operations/GitHub_GoGateway_CICD_SOP.md)。

## 工作流

| 工作流 | 触发条件 | 作用 |
| --- | --- | --- |
| `Go Gateway CI` | 任意分支推送、Pull Request；也可由发布工作流复用 | 格式检查、Gateway 与集成测试、Docker 镜像构建检查 |
| `Release Go Gateway` | 推送 `vMAJOR.MINOR.PATCH` 格式的 Tag | 先复用 CI；通过后将 Gateway 镜像发布到 GHCR，并创建 GitHub Release |

CI 使用 GitHub 托管的 `ubuntu-latest` Runner；权限为只读，不访问仓库 secrets，不下载 checkpoints。Python 模型组件和模型级 Test、Bench、Profile 由工程师在 GPU 资源分配后手动运行，不进入任何 Runner。Release 只对有权限的 Tag 推送者开放，发布工作流用最小化的 `GITHUB_TOKEN` 权限上传镜像和创建 Release。它不会登录或修改部署服务器。

## 仓库管理员首次设置

1. 在 `Settings → Branches` 或 `Rulesets` 保护 `main`：要求 Pull Request 合并，并要求 `Go Gateway CI / Go checks and image build` 检查通过。
2. 增加 `v*` Tag 规则，限制谁可以创建或更新发布 Tag。发布前先确认 `main` 上的 CI 全绿。
3. 在 `Settings → Actions → General` 中将工作流的 `GITHUB_TOKEN` 默认权限设为只读。发布工作流只在发布 Job 单独申请 `contents: write` 和 `packages: write`。
4. 首次发布后，在仓库 `Packages` 页面核对 `microinfer-gateway` 的来源仓库、可见性和访问权限。公开下载需将包设为 Public；Private 包需在部署机配置仅有 `read:packages` 权限的凭证。

当前仓库是公开仓库。不要把持久化自托管 Runner（尤其是带模型权重和 4090 GPU 的主机）挂到此仓库执行 PR 工作流。GitHub 说明公开仓库的 PR 可能运行不可信代码并危及自托管主机。GPU 自动验收应在单独评估可信提交流和隔离方式后接入。

## 本地复现 CI

从仓库根目录执行：

```bash
cd src/gateway
test -z "$(gofmt -l .)"
go vet ./...
go test -race -count=1 ./...

cd ../../tests/integration
test -z "$(gofmt -l .)"
go vet ./...
go test -race -count=1 ./...

cd ../..
docker build --pull --file deploy/docker/Dockerfile --tag microinfer-gateway:ci .
```

集成测试构造 Gateway Handler 并经 HTTP 接口检查，不启动 Inference、不需要 API Key 或模型权重。

## 发布与部署

通过评审并合并到 `main` 后，维护者按仓库版本策略创建并推送版本 Tag，例如：

```bash
git tag -a v0.1.0 -m "Release Gateway v0.1.0"
git push origin v0.1.0
```

发布完成后，GHCR 镜像地址为：

```text
ghcr.io/milo-ally-open/microinfer-gateway:v0.1.0
ghcr.io/milo-ally-open/microinfer-gateway:sha-<提交短 SHA>
```

部署仍按 [`../deploy/docker/README.md`](../deploy/docker/README.md) 手动执行。将 `deploy/docker/.env` 中的 `GATEWAY_IMAGE` 指向版本镜像，先 `docker compose pull gateway`，再使用 `docker compose up --no-build -d`。发布不会自动变更线上服务；回滚时将镜像 Tag 改回已验收版本并重新启动 Gateway。
