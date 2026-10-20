# 端到端測試

`test_example.py` 以 mock 驗證 `example.py` 使用 OpenAI Python SDK 發出的圖片聊天請求。這是範例程式測試，不連接 Gateway、推論服務或 GPU：

```powershell
python -m unittest discover -s tests/end2end -v
```

`test_gateway_client.py` 使用本機 HTTP stub 檢查性能基線客戶端的 SSE 與影片 multipart 請求格式，不連接真實服務。

`performance_baseline.py` 是連接已部署 Gateway 和推論服務的實測工具。執行前先確認服務已就緒，設定 `GATEWAY_URL` 與 `VIDEO_API_KEY`，並提供圖片／影片素材；本工具不下載模型、不啟動服務，也不設定硬性 SLA。

```powershell
python tests/end2end/performance_baseline.py --mode mixed --requests 10 --concurrency 1 --output results/gateway-baseline.json
```

性能報告包含總吞吐、錯誤率、p50／p95／p99 延遲、SSE 首文字 token 延遲及影片本文送完時間。真實推論性能基線尚未執行。
