# 端到端性能驗證紀錄

## 2026-10-22｜Gateway → Inference 首次完整流程

**結果：通過功能性端到端驗證。** Gateway 健康檢查、就緒檢查、模型別名查詢、圖片聊天 SSE 與視訊檔案上傳均完成；性能數據只代表本次單次序列測量，不構成穩定基線或 SLA。

### 環境與設定

| 項目 | 本次值 |
| --- | --- |
| 測試時間 | 2026-10-22 00:47:03 UTC（2026-10-22 08:47:03 UTC+08:00） |
| 主機／GPU | Windows 主機、WSL Ubuntu 24.04.5；NVIDIA GeForce RTX 4060 Laptop GPU，8,188 MiB |
| 推論環境 | Conda `magevl-transformers`，Python 3.12.15 |
| Gateway／推論監聽 | `127.0.0.1:8000` → `127.0.0.1:8001` |
| Gateway 設定 | `src/gateway/configs/gateway.yml` |
| 推論設定 | `src/inference/models/mage_vl_4b/configs/inference.yml` 的臨時測試副本；正式 YAML 未修改 |
| 模型與後端 | 專案別名 `mage-vl-4b`；上游 `microsoft/Mage-VL`；Transformers；revision `d88b153285f1633a61b2f693c59c8576693af185` |
| 權重配置 | NF4，compute dtype `bfloat16`，`device_map: auto`；使用本機固定快照並啟用離線載入 |
| 視訊測試覆寫 | `video_max_frames: 4`、`video_max_frame_side: 448`，只用於本次臨時 YAML；正式設定仍為 16 幀、896 px |
| 測試素材 | `magevl-soccer-frame.jpg`：146,169 bytes；SHA-256 `B295BC10691C857622128484089E6B1695F88B4B0182E228C6C9AD9EA238203A`。`test.mp4`：3,914,911 bytes；SHA-256 `9D840E86558FE5A59BB743092D161228C63C687AB53588EE476A1CD46064125A` |

測試由 `tests/end2end/performance_baseline.py` 執行，模式 `mixed`、總請求數 2、並行數 1、逾時 900 秒。API Key 與 Live session HMAC 測試密鑰只存於工作樹外的臨時目錄，測試後移除。

### 通路結果

| 檢查／場景 | 路徑 | 結果 | 觀測值 |
| --- | --- | --- | --- |
| Gateway 存活 | `GET /healthz` | HTTP 200 | `status: ok` |
| Gateway 就緒 | `GET /readyz` | HTTP 200 | `status: ready`；模型載入期間曾回 503，載入完成後轉為 200 |
| 模型清單 | `GET /v1/models` | HTTP 200 | 對外別名 `mage-vl-4b` |
| 圖片聊天 SSE | `POST /v1/chat/completions` | HTTP 200，1/1 成功 | Gateway 記錄總時長 2,312 ms；客戶端 SSE 首個文字 token 2.145 s；回應 690 bytes |
| 視訊理解上傳 | `POST /v1/videos/understand` | HTTP 200，1/1 成功 | Gateway 記錄總時長 1,129 ms；客戶端本文傳送完成 0.060 s；回應 760 bytes |

### 性能結果

基線腳本彙總 2 個請求，總計 2 成功、0 失敗，錯誤率 0%。`p50`／`p95`／`p99` 使用 nearest-rank 算法。

| 指標 | 本次測量 | 樣本數 |
| --- | ---: | ---: |
| 總耗時 | 3.6399 s | 2 個序列請求 |
| 吞吐量 | 0.5495 requests/s | 2 個序列請求 |
| 客戶端端到端延遲平均 | 1.8195 s | 2 |
| 客戶端端到端延遲 p50 | 1.1513 s | 2 |
| 客戶端端到端延遲 p95 | 2.4878 s | 2 |
| 客戶端端到端延遲 p99 | 2.4878 s | 2 |
| SSE 首個文字 token 延遲 | 2.1450 s | 1 |
| 視訊本文傳送完成時間 | 0.0600 s | 1 |

### 結論與限制

- 已驗證客戶端經 Go Gateway 到 Mage-VL 推論 API 的圖片 SSE 與視訊上傳路徑；未測試即時推流／MediaMTX。
- 這是連線、路由、格式轉送與回應完成的功能性驗證，不是模型精度評測；未執行 Video-MME 或其他資料集測試。
- 每種請求只有 1 次；整體只有 2 次。此樣本量不足以形成穩定延遲分位數，尤其 `p95`／`p99` 在兩筆資料時等於最大值。未設定或宣稱任何 SLA。
- 本次為控制 8 GiB GPU 的峰值，視訊抽幀與解析度使用臨時覆寫值；測得結果不代表正式 YAML 預設 16 幀、896 px 的性能或品質。
- 推理服務與 Gateway 均已停止；停止後 `nvidia-smi` 顯示 GPU 已用 3,147 MiB、可用 4,811 MiB。測試期間不使用併發。
- 本次未修改 Gateway、Inference 程式或正式推理 YAML；未下載模型、未提交或推送變更。
