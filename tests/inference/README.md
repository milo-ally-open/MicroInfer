# 推論測試

本目錄一般測試不載入 CUDA、模型權重或 MediaMTX。來源契約測試檢查啟動選項、登錄表、API 路由、SSE 標記及 API／Engine／Worker／Model Runner 分層；Engine 測試使用假 Worker、Adapter 與 Runner 驗證就緒狀態、請求分派、容量拒絕、串流取消釋放及 Live 共用准入。序列化測試固定聊天與影片 API response 欄位。`test_batch_planner.py` 驗證線上 Scheduler 使用的 token 批次規劃器；KV cache 測試另驗證原子 batch lease 的容量拒絕與 block 回收。`test_qwen3_native_runtime.py` 是例外的 CPU 合成測試，需安裝 PyTorch 和 Safetensors；使用小型隨機權重驗證 Qwen 計算圖、變長批次結果、Safetensors 嚴格載入與實體 page 所有權，不下載真實模型或使用 GPU。Qwen3-0.6B 的真實權重 smoke test 由 `scripts/qwen3_0_6b/test.py` 執行。

```powershell
python -m unittest discover -s tests/inference -v
```

這些測試不證明 FastAPI runtime、Mage-VL 真實載入、GPU 推論或 MediaMTX 連線成功。模型 smoke test 需另行提供已快取權重與可用 GPU，並明確標記執行環境；不得由測試自動下載大型模型。

要取得實際 Mage-VL 模型結構及 cache 資訊，請使用模型專屬工具 [`scripts/mage_vl_4b/test.py`](../../scripts/mage_vl_4b/test.py)。執行時必須明確傳入 YAML、模型快照、圖片或影片、問題文字與輸出報告路徑；不會從腳本位置推導專案根目錄或預設讀取同目錄 YAML。相對路徑以啟動命令時的目前工作目錄解析。YAML 欄位沿用服務的模型／Processor／runtime／API／generation／optimization 結構，輸出 token 數預設取自明確指定的 YAML。它不是 unittest，不從 `src/` 匯入程式，且只允許本機快照。使用方式見 [`scripts/README.md`](../../scripts/README.md)。
