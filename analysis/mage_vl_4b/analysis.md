# Mage-VL Gateway 验收

本目录保存 Mage-VL 的问题、解决方案和验收效果；表格中的编号对应 [analysis.xlsx](analysis.xlsx) 里的请求和测量数据。

## 验收环境

历史记录日期为 2026-10-22。模型 `mage-vl-4b`（`microsoft/Mage-VL`），revision `d88b153285f1633a61b2f693c59c8576693af185`；Transformers NF4/bfloat16，RTX 4060 Laptop 8,188 MiB，WSL Ubuntu 24.04.5，Python 3.12.15。Gateway `127.0.0.1:8000` 转发至 Inference `127.0.0.1:8001`；`/healthz`、`/readyz`、`/v1/models` 均返回 HTTP 200。

## 问题、解决方案、效果

| 问题 | 解决方案 | 效果 | XLSX 证据 |
|---|---|---|---|
| 需要确认图片聊天 SSE 能经 Gateway 到达 Mage-VL 并返回响应。 | 通过 Gateway 调用 `POST /v1/chat/completions`，使用单张图片输入并读取 SSE。 | HTTP 200，1/1 成功；Gateway 2.312 s，TTFT 2.145 s，响应 690 bytes。 | MV-01 |
| 需要确认上传视频理解能经 Gateway 到达 Mage-VL 并返回结果。 | 通过 Gateway 调用 `POST /v1/videos/understand`；本次临时采用 4 帧、max side 448。 | HTTP 200，1/1 成功；Gateway 1.129 s，请求正文发送完成 0.060 s，响应 760 bytes。 | MV-02 |

合计 2 个串行请求，2/2 成功、0 错误，总计 3.6399 s、0.5495 requests/s、平均 1.8195 s、p50 1.1513 s、p95/p99 2.4878 s。样本量不足以代表 SLA；视频正式默认为 16 帧、max side 896；未测 Live/MediaMTX，未修改正式 YAML。

素材记录：`magevl-soccer-frame.jpg`，146,169 bytes，SHA-256 `B295BC10691C857622128484089E6B1695F88B4B0182E228C6C9AD9EA238203A`；`test.mp4`，3,914,911 bytes，SHA-256 `9D840E86558FE5A59BB743092D161228C63C687AB53588EE476A1CD46064125A`。
