# GitHub Actions

本目录维护 MicroInfer 的 GitHub CI 与版本制品交付工作流。

- `workflows/ci.yml`：PR/main 分支运行 CPU 静态检查、模型配置预检和 Python 包构建；`v*` tag 通过检查后创建 GitHub Release。
- 使用 GitHub 托管 Runner，不将公开仓库代码送到共享 4090。
- 不在 Runner 执行模型组件/模型级 Test、Bench、Profile、权重下载、GPU 推理或部署服务。
- CD 当前交付 Python wheel、sdist 和 SHA256 校验文件；不自动 SSH 到推理主机、不构建推理 Docker 镜像、不上传模型权重。
- Actions 固定到完整 commit SHA；全局 token 权限只读，Release job 单独获得 `contents: write`。

分支保护、tag 规则和交付边界见 [CI/CD 架构与操作说明](../docs/Operations/CI_CD_Runner_Architecture_Plan.md)。
