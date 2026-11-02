# 部署

本目录按服务划分部署方式。Go Gateway 可独立构建为 Docker 镜像；Inference Engine 在宿主机的 Python 环境中安装和运行，不构建为 Docker 镜像。

| 服务 | 部署位置 | 说明 |
| --- | --- | --- |
| Gateway | [`docker/`](docker/) | Go HTTP API、认证、策略、路由和上游转发；Docker Compose 只启动此服务 |
| Inference Engine | 宿主机 | 使用独立 Python/Conda 环境加载模型并使用 GPU，Gateway 通过宿主机网络访问 |
| MediaMTX | [`mediamtx/`](mediamtx/)（可选） | Mage-VL Live 视频需要的媒体 sidecar，单独安装与启动 |

完整的 Gateway 容器部署、密钥配置、Inference 网络准备和验收步骤见 [`docker/README.md`](docker/README.md)。
