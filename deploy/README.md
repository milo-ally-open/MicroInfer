# 部署

部署直接运行 Python FastAPI Inference 服务。Inference Engine、模型权重和 GPU 运行时安装在宿主机 Python 环境中，不构建为 Docker 镜像。控制面由独立项目负责。

| 服务 | 部署位置 | 说明 |
| --- | --- | --- |
| Inference API | 宿主机 | FastAPI 直接提供 `/v1/*` 模型 API、API Key 认证、租户限流和 GPU 推理 |
| MediaMTX | [`mediamtx/`](mediamtx/)（可选） | Mage-VL Live 视频需要的媒体 sidecar，单独安装与启动 |

Inference 安装、密钥配置和直接服务步骤见 [`../src/inference/README.md`](../src/inference/README.md)。
