# MediaMTX Sidecar 配置

MediaMTX 是独立的实时媒体 sidecar，不属于推理引擎，也不属于任何单一模型包。服务配置统一放在本目录，部署时与推理 API 分别启动和停止。

从项目根目录运行已安装的 MediaMTX 二进制：

```bash
mediamtx deploy/mediamtx/mediamtx.yml
```

默认配置只启用 RTSP/TCP 和 RTMP 输入，并通过 HTTP 调用推理 API 的 `/v1/live/mediamtx/auth` 授权回调。默认回调地址为 `http://127.0.0.1:8001`。如果推理 API 地址或端口改变，需要更新 `mediamtx.yml` 中的 `authHTTPAddress`；如果改变 RTSP/RTMP 监听端口，还要同步更新使用 Live 功能的模型 `configs/network.yml` 中的客户端连接端口。

模型 manifest 不登记 MediaMTX 配置路径。模型的 `network.yml` 只保存推理服务使用的 RTSP/RTMP 地址和返回给发布客户端的公开主机名；协议监听、授权回调及 MediaMTX 自身的运行选项只在本目录维护。
