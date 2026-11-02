# Gateway Docker 部署

此部署只将 Go Gateway 构建为 Docker 镜像。Inference Engine、模型权重和 GPU 运行时保留在宿主机；Compose 不构建或启动 Inference 容器。Gateway 使用 `host.docker.internal` 访问宿主机上的 Inference API。

## 目录内容

| 文件 | 用途 |
| --- | --- |
| `Dockerfile` | 多阶段编译 Go Gateway，最终镜像不包含 Go 工具链或模型代码 |
| `compose.yml` | 只定义 `gateway` 服务，并挂载只读配置与 API Key 文件 |
| `gateway.yml` | 容器专用配置；上游地址指向宿主机的 Mage-VL 和 Qwen3-VL 服务 |
| `.env.example` | Compose 变量模板；复制为本地 `.env` 后填入 HMAC 密钥 |
| `secrets/api-keys.example.json` | API Key 文件格式模板；不包含有效密钥 |

## 前置条件

- Linux 主机或 Docker Desktop，支持 Docker Compose v2。
- Go 不需要安装在部署主机上；Docker 镜像构建阶段会使用 Go 编译器。
- 宿主机已安装并启动需要提供的 Inference 模型。
- Gateway 对外使用的 API Key 与 session HMAC 密钥已经生成。

Gateway 镜像不需要 CUDA、NVIDIA Container Toolkit 或模型权重。GPU 只由宿主机上的 Inference Engine 使用。

## 1. 配置宿主机 Inference

Gateway 容器不能通过 `127.0.0.1` 访问宿主机服务。启动 Inference 时将 API listener 绑定到容器可访问的宿主机接口：

```bash
inference start --model mage-vl-4b --host 0.0.0.0
inference start --model qwen3-vl-2b --host 0.0.0.0
```

默认端口分别为 `8001` 和 `8012`。只启动实际部署的模型即可；如果 `gateway.yml` 保留两个模型路由，`/readyz` 会检查两个上游，因此两个模型都需要就绪。只部署部分模型时，修改 Gateway 配置，只保留需要的路由，并保持默认视频与 Live 模型指向具备相应能力的路由。

绑定 `0.0.0.0` 会让 Inference 监听所有宿主机接口。用主机防火墙限制 `8001`、`8012` 只允许本机 Docker 网络访问；不要直接将这些端口暴露到公网。Gateway 默认只发布到宿主机 `127.0.0.1:8000`。

确认宿主机服务健康：

```bash
curl http://127.0.0.1:8001/health
curl http://127.0.0.1:8012/health
```

Windows/Mac Docker Desktop 已提供 `host.docker.internal`。Linux Compose 通过 `extra_hosts: host-gateway` 添加此主机名映射。

## 2. 配置密钥

从仓库根目录进入部署目录并创建本地配置：

```bash
cd deploy/docker
cp .env.example .env
mkdir -p secrets
cp secrets/api-keys.example.json secrets/api-keys.json
```

编辑 `secrets/api-keys.json`，为每个客户端设置唯一 `id`、`tenant_id` 和至少 16 个字符的随机 `key`。生成 API Key 示例：

```bash
openssl rand -hex 32
```

生成 session HMAC 密钥并将输出填入 `.env` 的 `GATEWAY_SESSION_HMAC_KEYS`：

```bash
python -c 'import base64,secrets; print(base64.urlsafe_b64encode(secrets.token_bytes(32)).decode().rstrip("="))'
```

在 Linux 部署机上，Gateway 以 UID/GID `10001` 运行。将只读 API Key 文件授权给该 UID，并限制宿主机权限：

```bash
sudo chown 10001:10001 secrets/api-keys.json
chmod 0400 secrets/api-keys.json
chmod 0600 .env
```

如果容器 UID/GID 不同，需同步调整 `.env` 中的 `GATEWAY_UID`、`GATEWAY_GID` 和密钥文件属主。不要将 `.env` 或 `secrets/api-keys.json` 提交到 Git。

session HMAC 密钥解码后至少为 32 bytes。轮换时可在变量中按顺序配置逗号分隔的密钥：第一个密钥用于签发新 session，后续密钥用于验证旧 session；待旧 session 到期后再移除旧密钥。

## 3. 构建并启动 Gateway

仍在 `deploy/docker/` 目录下执行：

```bash
docker compose config --quiet
docker compose up --build -d
docker compose ps
docker compose logs -f gateway
```

默认使用本地源码构建。GitHub 发布流程通过 Tag 将 Gateway 镜像推送到 GHCR；需要部署已发布版本时，在 `.env` 中将 `GATEWAY_IMAGE` 改为已验收的镜像地址，例如 `ghcr.io/milo-ally-open/microinfer-gateway:v0.1.0`，然后执行：

```bash
docker compose pull gateway
docker compose up --no-build -d
```

如果 GHCR 包设为私有，部署机需先使用仅有 `read:packages` 权限的凭证登录 `ghcr.io`。Tag 发布只交付镜像，不会自动登录部署机或更新线上服务；回滚时将 `GATEWAY_IMAGE` 改为上一个已验收版本并重新启动。

Docker build context 使用仓库根目录；`.dockerignore` 限定传入构建器的内容为 Gateway 源码和容器配置，不会发送模型 checkpoints、Inference 包、分析产物或仓库其他文件。

停止 Gateway：

```bash
docker compose down
```

## 4. 验收

`/healthz` 检查 Gateway 进程存活；`/readyz` 检查所有已配置模型的 Inference 上游；`/metrics` 输出 Prometheus 指标：

```bash
curl http://127.0.0.1:8000/healthz
curl http://127.0.0.1:8000/readyz
curl http://127.0.0.1:8000/metrics
```

健康检查容器只探测 `/healthz`，不会因 Inference 临时重启而让 Docker 不断重启 Gateway。客户端请求仍需通过 `Authorization: Bearer <API_KEY>` 认证；仓库根目录 `client.py` 提供 OpenAI Python SDK 的 Gateway smoke 调用。

## 网络与生产配置

- `GATEWAY_BIND_ADDRESS` 默认 `127.0.0.1`。有受控反向代理时可让代理访问宿主机回环端口；确需远程直连时才绑定 `0.0.0.0`，并配置防火墙和 TLS。
- 将 `gateway.yml` 的 `public_base_url` 设置为客户端实际访问的 URL；TLS 由反向代理终止时应填写 HTTPS 地址。
- 容器访问 Inference 的上游地址在 `gateway.yml` 中配置。更改宿主机端口或主机访问方式时同步更新 `upstream.base_url` 和每个模型的 `base_url`。
- Compose 只部署 Gateway。MediaMTX 是 Mage-VL Live 视频的可选独立 sidecar，使用 [`../mediamtx/README.md`](../mediamtx/README.md) 单独配置。
- 当前限流状态保存在 Gateway 单进程内存中；多实例共享限流、生产级密钥管理和反向代理 TLS 配置需由部署环境另行提供。
