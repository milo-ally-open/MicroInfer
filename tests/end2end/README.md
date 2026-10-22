# 端到端測試

`test_example.py` 以 mock 驗證 `example.py` 使用 OpenAI Python SDK 發出的圖片聊天請求。這是範例程式測試，不連接 Gateway、推論服務或 GPU：

```powershell
python -m unittest discover -s tests/end2end -v
```

`test_gateway_client.py` 使用本機 HTTP stub 檢查性能基線客戶端的 SSE 與影片 multipart 請求格式，不連接真實服務。

`performance_baseline.py` 是連接已部署 Gateway 和推論服務的實測工具。執行前先確認服務已就緒並設定 `GATEWAY_URL` 與 `VIDEO_API_KEY`；本工具不下載模型、不啟動服務，也不設定硬性 SLA。`text-sse` 量測 Qwen3 純文字 SSE，不需媒體素材；`chat-sse` 與 `video`／`mixed` 量測 Mage-VL，後三者需要對應素材。

Qwen3 純文字 SSE 範例：

```powershell
python tests/end2end/performance_baseline.py --mode text-sse --model qwen3-0.6b --requests 3 --concurrency 1 --output tests/end2end/results/qwen3-0.6b-gateway-smoke-2026-10-22.json
```

```powershell
python tests/end2end/performance_baseline.py --mode mixed --requests 10 --concurrency 1 --output results/gateway-baseline.json
```

性能報告包含模型別名、總吞吐、錯誤率、p50／p95／p99 延遲、SSE 首文字 token 延遲及影片本文送完時間。

2026-10-22 已透過本機 Go Gateway 與真實 Qwen3-0.6B CUDA API 執行 3 個串行純文字 SSE 請求；在 WSL `magevl-transformers` 環境及低顯存臨時設定（128 個 KV blocks、單序列、最多 16 個新 token）下，3/3 成功，吞吐約 0.96 req/s，端到端延遲 p50 約 0.93 秒、TTFT p50 約 0.065 秒。另以 2 個並行文字請求確認兩者成功，並在首個 SSE 文字增量後斷開客戶端；服務健康資訊確認 Scheduler、Engine lease 與 Worker 實體 KV pages 均歸零。兩份報告分別位於 [`results/qwen3-0.6b-gateway-smoke-2026-10-22.json`](results/qwen3-0.6b-gateway-smoke-2026-10-22.json) 與 [`results/qwen3-0.6b-gateway-lifecycle-smoke-2026-10-22.json`](results/qwen3-0.6b-gateway-lifecycle-smoke-2026-10-22.json)。這些是低負載鏈路 smoke 數據，不代表正式容量或 SLA。

同日使用 Gateway 臨時驗收設定再完成 Qwen 非串流 JSON 與 SSE 各 1 次：JSON HTTP 200、生成 9 個 token 並正常停止；SSE 1/1 成功，TTFT 約 0.294 秒。報告位於 [`results/qwen3-0.6b-gateway-json-smoke-2026-10-22.json`](results/qwen3-0.6b-gateway-json-smoke-2026-10-22.json) 與 [`results/qwen3-0.6b-gateway-sse-smoke-2026-10-22.json`](results/qwen3-0.6b-gateway-sse-smoke-2026-10-22.json)。臨時設定只用於單一 Qwen 上游運行；未形成正式容量數據。

同日另使用 Mage-VL 固定 snapshot 與 4060 低顯存 int8 CPU offload 臨時設定，透過 Gateway 完成圖片聊天非串流 JSON 1 次：HTTP 200、回傳專案別名、生成 8 個 token 後依請求上限正常結束。測試素材與報告位於 [`magevl-soccer-frame.jpg`](../../magevl-soccer-frame.jpg) 與 [`results/mage-vl-4b-gateway-json-smoke-2026-10-22.json`](results/mage-vl-4b-gateway-json-smoke-2026-10-22.json)。這是功能檢查，不代表完整預設設定的性能或生成品質。

同日另以真實 Mage-VL 及 Go Gateway 完成圖片聊天 SSE 與視訊上傳各 1 次，2/2 成功；環境、臨時低顯存影片設定與量測值記錄在 [`checkpoints/val.md`](../../checkpoints/val.md) 及 [`checkpoints/val.xlsx`](../../checkpoints/val.xlsx)。Live／MediaMTX 已完成 1 次合成 RTSP 功能 smoke；所有路徑樣本數都很少，僅供確認端到端功能，不構成穩定性能基線或 SLA。高併發仍未測。

同日另在系統臨時目錄建立沒有模型專屬 Runner／Adapter／InputProcessor 工廠的 `standard` profile，使用固定 Qwen3 snapshot 啟動完整 API Server。`/health`、`/v1/models`、非串流聊天與文字 SSE 均通過；SSE 與非串流文本一致，公開 alias、usage 與 `[DONE]` 正確。這是直接 API 驗證，並未連 Go Gateway；正式 Registry 的 Qwen 項目仍保留 `custom`。報告見 [`results/qwen3-0.6b-standard-profile-api-smoke-2026-10-22.json`](results/qwen3-0.6b-standard-profile-api-smoke-2026-10-22.json)。

`live_smoke.py` 透過 Gateway 建立 Mage-VL 即時工作階段，使用 FFmpeg 產生合成 RTSP 影像，等待 SSE `analysis` 事件並停止工作階段。執行前須自行啟動 Gateway、Inference 與 MediaMTX，並設定 `VIDEO_API_KEY` 和可用的 FFmpeg；此工具不啟動服務、不下載模型，也不驗證生成內容的語意正確性：

```powershell
$env:VIDEO_API_KEY = '<Gateway API Key>'
python tests/end2end/live_smoke.py --timeout-seconds 240 --max-tokens 16 --output tests/end2end/results/mage-vl-4b-live-smoke.json
```

2026-10-22 使用 Mage-VL、Go Gateway、MediaMTX 與 FFmpeg 合成來源完成一次即時影片端到端 smoke。Gateway `/v1/models`、即時工作階段建立、SSE 連線與停止均回傳成功；MediaMTX 收到 RTSP 發布並提供影格讀取，SSE 收到 `analysis` 事件。測試使用 int8 CPU offload 臨時設定、320×240、2 FPS、`frames` backend、最多 8 個輸出 token；首個 analysis 約 36.4 秒，測試總耗時約 38.1 秒。報告中的發布程序在 session 停止後以 255 結束；session 清理成功，但未驗證未授權／過期媒體 URL 的拒絕邊界。這是單次低負載功能檢查，不代表穩定性能基線或 SLA。報告位於 [`results/mage-vl-4b-live-smoke-position-fix-2026-10-22.json`](results/mage-vl-4b-live-smoke-position-fix-2026-10-22.json)。

除錯過程也保留兩份失敗報告：[`results/mage-vl-4b-live-smoke-final2-2026-10-22.json`](results/mage-vl-4b-live-smoke-final2-2026-10-22.json) 與 [`results/mage-vl-4b-live-smoke-nf4-2026-10-22.json`](results/mage-vl-4b-live-smoke-nf4-2026-10-22.json)。根因是 Mage 多模態模型增量 decode 時，位置編碼仍依完整 `attention_mask` 產生整段 position，令單 token query 被擴成整段上下文，並造成 Attention 輸出投影維度錯誤。現已改為依完整 mask 計算位置後，只傳入目前 query 對應的尾端位置；`tests/inference/test_mage_vl_stepwise_batch.py` 的 CPU 單元測試與上述真實 Gateway smoke 均已驗證修正。此修正不代表已驗證高併發或語意準確性。
