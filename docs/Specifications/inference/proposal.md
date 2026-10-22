# 提案：推論服務五元件最小 Engine

> 規格：[spec.md](spec.md)｜設計與調研：[design.md](design.md)｜任務與驗收：[tasks.md](tasks.md)
> 狀態：五個 Engine 元件、Runner 相容輸入批次 forward、Worker Cache 生命週期及 API／Engine Core／GPU Worker 三程序邊界均已實作。T-41 以假 Worker 驗證 IPC、取消、事件順序和程序隔離；本次未啟動 FastAPI runtime、真模型、GPU 或 MediaMTX。

## 背景

目前唯一登錄模型是 `mage-vl-4b`，上游為 `microsoft/Mage-VL`，revision 為 `d88b153285f1633a61b2f693c59c8576693af185`。API、Engine Core 和 GPU Worker 現已分開；Engine 逐輪排程 prefill/decode，並透過 EngineClient 將 request、事件和取消訊息跨程序傳送。

本次參考 vLLM 的 API Server、Engine Core、Scheduler、KV cache 管理及 Worker／Model Runner 職責，將推論鏈路改成可逐 token 排程的最小 Engine。此為本專案自己的實作，不是 vLLM 後端，也不以複製 vLLM 專用資料結構為目標。

## 已確認事實與決策

- 外部端點及 API schema 維持 `/health`、`/v1/models`、聊天/SSE、影片理解及 `/v1/live/*`；文字 SSE 保留逐 token chunk、finish usage 和 `[DONE]`。
- API Server、Engine Core、GPU Worker 以 `spawn` 建立三個程序。API 端 `EngineClient` 負責 IPC；Engine Core 管理 Scheduler、KV lease 與批次生命週期；模型權重和 Cache object 只存在 GPU Worker。
- Scheduler 對 prefill 將完整 prompt 作為一輪預算，decode 每輪分配一個 token，對活動請求公平輪轉；相容同階段 request 可合併為一次模型 forward。目前不切塊 prefill、不混合 prefill/decode，也不宣稱 continuous batching。
- Engine KV Cache Manager 以固定大小邏輯 block 預留容量；Worker 的 `RequestCacheManager` 實際持有每 request Transformers Cache、追蹤長度並在終止時釋放。邏輯 block 尚未映射至注意力核心中的實體 GPU 頁面，不宣稱 PagedAttention、prefix cache 或實際顯存池化。
- Runner 以 `past_key_values` 暴露 prefill／decode step。模型固定 revision 的既有載入部分驗收與後續限制記於任務驗收紀錄；本輪程序隔離使用假 Worker，沒有啟動模型或 GPU。
- 首版 stepwise sampler 支援 greedy、temperature、top-k、top-p、min-p、repetition penalty 及 no-repeat n-gram；beam search、自訂 generation kwargs 與自訂 cache implementation 暫不支援。
- `runtime.uvicorn_workers` 必須是 `1`；Engine Core 的啟動／請求 timeout 及 GPU Worker timeout 分別由模型 `inference.yml` 設定。
- Mage-VL 模型載入器、逐步生成狀態與元件替換登記位於 `models/mage_vl_4b/components/`；共用 Engine 不依賴具體模型內部類別。
- Mage-VL 固定 revision 的模型設定、視覺塔、多模態組裝及生成頭已拆入模型包；語言骨幹仍透過 Transformers `AutoModel` 組裝，Processor 仍透過 `AutoProcessor` 載入同一固定 revision 的模型專屬程式。

## 範圍

- `api/` 保留 HTTP、schema、SSE 序列化及本文限制。
- `engine/` 實作 EngineClient、獨立 Engine Core、公平 Token Scheduler、邏輯 KV block 預留及 GPU Worker IPC。
- `models/mage_vl_4b/` 實作 Mage-VL 前處理、逐步 Model Runner 及結果轉換；模型僅於 Worker 子程序載入。
- 在模型包內拆出 `model_loader.py`、`stepwise_generation.py`、`component_registry.py` 及固定 revision 模型定義；`inference.yml` 的空替換表維持上游架構，未登記實作在載入權重前拒絕。
- 將已下載的固定 revision 模型程式依視覺元件、視覺塔、多模態組裝及生成頭拆入 Mage-VL 模型包；載入器用本地模型類別與設定，防止模型架構程式由權重快照動態注入。
- 更新 `inference.yml`、README 和本 Spec，記錄排程／cache 邊界、配置和不支援項目。
- 在 `tests/inference/` 驗證 Scheduler、相容 Runner 批次、Worker Cache 生命週期、三程序 IPC、取消清理及 SSE 契約；本輪使用假 Worker，不載入權重。
- 線上 Engine 已呼叫批次規劃並將工作批次傳至 Runner；Runner 對結構相容的 Mage-VL 輸入使用一次多序列 forward，不相容時逐 request fallback。
- 在 `scripts/mage_vl_4b/` 提供 Mage-VL 專屬離線探索工具；只讀模型 YAML，不從 `src/` 匯入程式，並記錄模型結構、輸入 tensor、cache 及 GPU 資訊。其他模型的探索工具放在對應別名子目錄。

## 非目標

- 不接入 vLLM/SGLang，不增加模型或模型精度測試。
- 不載入真實權重或啟動 GPU；本地模型程式與權重鍵相容性須後續整合驗收。
- 不實作 GPU 實體分頁 cache、prefix cache、KV eviction／offload、chunked prefill、多 GPU 或多 Engine Core。
- 不宣稱已支援 continuous batching；chunked prefill 與 prefill/decode 混合調度另列後續任務。
- 不改 Go Gateway，不改 API 路徑及既有外部 response schema。
- 重用已存在於 Git 忽略 `checkpoints/` 的固定 revision 快照，不重複下載或修改權重；不載入模型、不啟動 GPU／MediaMTX、不操作遠端伺服器、不提交或推送。

## 風險與回復

逐步 Runner 依賴模型的 `prepare_inputs_for_generation`、`past_key_values` 和 cache 長度行為；固定 revision 的 GPU 驗收仍是上線前必要條件。若此介面不相容，可回復 Mage-VL Runner 與 Engine 的逐步生成提交，並在完成相容設計前停用新 Worker 啟動；HTTP route 和 schema 不必變更。

邏輯 cache 預留是容量政策和追蹤資訊，不控制 HF 動態張量的實際配置。健康資訊將其 bytes 標示為估算值，避免誤認為實際 VRAM 使用量。

## 完成條件

- 五個元件皆在實際請求鏈路中有可驗證職責；API Server、Engine Core、GPU Worker 獨立，並完成 REQ／AC／T 追蹤。
- 對話、影片、Live 生成共用逐步 Worker 呼叫；每輪 token turn 之間可切換其他活動請求。
- Worker 子程序生命週期、Scheduler 取消、cache block 回收及既有 API 序列化有無模型測試證據。
- README、Spec 及 YAML 不把邏輯 cache reservation 描述成實體 PagedAttention，不把架構參考描述為 vLLM 整合。

## 變更記錄

| 日期 | 內容 |
|---|---|
| 2026-10-20 | 由整請求准入調整為五元件最小 Engine；加入子程序 Worker、逐步生成、Token Scheduler 與邏輯 KV block 管理。 |
| 2026-10-20 | 增加未接入線上鏈路的批次規劃／原子 KV 預留邏輯，以及不匯入 `src/`、只使用本機權重的模型能力探測腳本。 |
| 2026-10-20 | 按模型別名整理探索腳本；依固定 revision 靜態拆出 Mage-VL 載入、逐步生成和元件 registry，補齊 REQ-36 無權重驗收。 |
| 2026-10-22 | 將本機已下載的 Mage-VL 固定 revision 模型定義按元件拆入模型包，模型類別改由本地程式載入；權重／GPU 整合仍待驗收。 |
| 2026-10-22 | 完成相容 Mage-VL 請求的多序列 forward、Worker Cache object 生命週期及 EngineClient 三程序隔離；本輪透過假 Worker 驗證程序 PID、SSE 事件順序、取消、錯誤回傳及資源回收，未啟動 GPU。 |
