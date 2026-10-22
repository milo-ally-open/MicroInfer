# 推論測試

本目錄一般測試不載入 CUDA、模型權重或 MediaMTX。來源契約測試檢查啟動選項、登錄表、API 路由、SSE 標記及 API／Engine／Worker／Model Runner 分層；Engine 測試使用假 Worker、Adapter 與 Runner 驗證就緒狀態、請求分派、容量拒絕、串流取消釋放及 Live 共用准入。序列化測試固定聊天與影片 API response 欄位。`test_batch_planner.py` 驗證線上 Scheduler 使用的 token 批次規劃器；KV cache 測試驗證單請求 block lease 的容量拒絕、回收及重用。`test_mage_vl_stepwise_batch.py` 和 `test_qwen3_native_runtime.py` 是隔離的 CPU 合成測試，需安裝 PyTorch；Qwen 測試另需 Safetensors。它們使用假模型或小型隨機權重驗證 Mage-VL prefill/decode 批次 Cache 邏輯、Qwen 計算圖及實體 page 所有權，不下載真實模型或使用 GPU。Qwen3-0.6B 的真實權重 smoke test 由 `scripts/qwen3_0_6b/test.py` 執行。

`test_model_integration_contract.py` 透過 `model_contract.py` 對每個登錄模型套用同一份 Runner／Adapter／InputProcessor AST 契約；標準模型會檢查共用預設元件。Launcher 的 `validate` 預檢 Registry、capability、YAML 與 factory class methods，不匯入模型程式。`test_standard_text_runtime.py` 是 CPU synthetic test，驗證標準預設元件的 Chat Template、Cache、batch work items、finish/cancel 和 token 輸出。`test_standard_text_snapshot.py` 預設跳過；只有明確設 `RUN_STANDARD_TEXT_SNAPSHOT=1`、`STANDARD_TEXT_INFERENCE_CONFIG` 並提供本機 snapshot 時才載入真實模型。`STANDARD_TEXT_DEVICE` 預設為 `cpu`，亦可明確選擇 `cuda:0`；GPU 模式要求空閒顯存至少 3072 MiB（可用 `STANDARD_TEXT_MIN_FREE_GPU_MB` 調高），採 `float16`、單請求及最多 2 個新 token，不啟動 API 或常駐服務。Gateway 與 Inference 的跨設定 alias、上游識別碼、能力及 endpoint 比對位於 `tests/end2end/`，不將 Gateway 依賴搬入推論測試。

```powershell
python -m unittest discover -s tests/inference -v
```

這些測試不證明 FastAPI runtime、Mage-VL 真實載入、GPU 推論或 MediaMTX 連線成功。模型 smoke test 需另行提供已快取權重與可用 GPU，並明確標記執行環境；不得由測試自動下載大型模型。

要取得實際 Mage-VL 模型結構及 cache 資訊，請使用模型專屬工具 [`scripts/mage_vl_4b/test.py`](../../scripts/mage_vl_4b/test.py)。執行時必須明確傳入 YAML、模型快照、圖片或影片、問題文字與輸出報告路徑；不會從腳本位置推導專案根目錄或預設讀取同目錄 YAML。相對路徑以啟動命令時的目前工作目錄解析。YAML 欄位沿用服務的模型／Processor／runtime／API／generation／optimization 結構，輸出 token 數預設取自明確指定的 YAML。它不是 unittest，不從 `src/` 匯入程式，且只允許本機快照。使用方式見 [`scripts/README.md`](../../scripts/README.md)。
