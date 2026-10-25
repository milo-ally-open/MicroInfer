# MediaMTX 部署

MediaMTX 是独立的实时媒体 sidecar，用于接收 RTSP/RTMP 推流。它不属于 Gateway、Inference Engine 或单一模型插件。图片和上传视频文件流程不依赖 MediaMTX。

## 前置条件

- 已安装 MediaMTX 二进制，并可通过 `mediamtx` 命令调用。
- 已启动支持 Live 视频的 Inference 服务。当前默认回调指向 Mage-VL Inference：`http://127.0.0.1:8001/v1/live/mediamtx/auth`。
- 如果推流客户端不在本机，需按部署网络配置防火墙、主机名和对外地址。

## 启动

从仓库根目录启动：

```bash
mediamtx deploy/mediamtx/mediamtx.yml
```

当前配置启用 RTSP/TCP（端口 `8554`）和 RTMP（端口 `1935`），并通过 HTTP 调用 Inference 的授权回调。HLS、WebRTC、SRT、MediaMTX API、metrics 和 pprof 均关闭。

MediaMTX 与 Inference 分别启动和停止。停止服务前，应先结束活动 Live session，再停止媒体 sidecar。

## 配置

服务自身的协议监听、授权回调和运行选项只在 [`mediamtx.yml`](mediamtx.yml) 配置。若 Inference 地址或端口变化，更新 `authHTTPAddress`；若改变 RTSP/RTMP 监听端口，同时更新使用 Live 的模型 `configs/network.yml`。

模型 `network.yml` 记录推理服务连接 MediaMTX 所需的地址和发布端公开主机名，不用于配置 MediaMTX 服务本身。模型 manifest 不登记该 sidecar 配置。

## 网络路径

Live 控制请求经 Gateway 转发到 Inference；MediaMTX 的 HTTP 授权回调直接访问 Inference。发布客户端依据 Live 响应中的地址直接向 MediaMTX 推流。Gateway 不转发 RTSP/RTMP 封包。
