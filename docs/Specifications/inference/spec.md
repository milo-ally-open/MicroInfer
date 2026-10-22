# 推論引擎五層行為規格

> 提案：[proposal.md](proposal.md)｜技術設計：[design.md](design.md)｜實施與驗收：[tasks.md](tasks.md)
> 本規格描述 Qwen3-0.6B 實驗改造及應保持的共用服務行為。已完成、進行中和待驗收的狀態以 `tasks.md` 為準。

### REQ-01：L1 接口與門面維持 API 契約

- **AC-01.1** `GET /health`、`GET /v1/models`、`POST /v1/chat/completions` 的路徑、HTTP 方法、模型別名和 OpenAI response schema 維持現有行為；Qwen3 仍只登錄文字聊天能力。
- **AC-01.2** 共用 `api/` 不匯入模型包，不載入模型權重、GPU Runtime 或模型專屬計算圖。
- **AC-01.3** Qwen `InputProcessor` 在 API Server 程序讀取模型 YAML 指定的本機 Tokenizer／Chat Template，將請求轉為包含 alias、token IDs 和 sampling params 的 `PreparedChatInput`；該資料可經 IPC 傳遞，且 API 程序不執行模型 forward。
- **AC-01.4** Qwen Worker 回傳 token ID 後，API 增量 Detokenize 並發出文字 SSE；完成時輸出尚未送出的文字、finish reason、usage 和 `data: [DONE]`，串流合併文字須等於完整生成文字。
- **AC-01.5** Mage-VL Worker 既有 `text_delta` 路徑仍轉成相同 OpenAI SSE schema；不得要求 Mage-VL 改用 Qwen tokenizer 或 token ID。
- **AC-01.6** API 客戶端斷線時，原有取消 IPC 與 shielded cleanup 仍生效；取消後不得繼續向該 stream 發送 token。
- **AC-01.7** Qwen 非串流聊天回應由 API Server 將 Worker 回傳的 token ID 序列解碼為 OpenAI `content`；Worker 不產生對外聊天字串。

### REQ-02：L2 Engine Core 統一管理請求執行閉環

- **AC-02.1** 每個可排程生成依序經過輸入準備、准入與 Cache reservation、Scheduler 建批、Executor 執行 Worker、批次結果更新、逐 request 事件發布及完成收尾。
- **AC-02.2** 成功、拒絕、取消、超時、Worker 異常和關閉均回收 Scheduler 名額、Engine lease 及 Worker request state；釋放流程冪等。
- **AC-02.3** `EngineOutputProcessor` 將 Worker token ID 或文字增量轉成不依賴 FastAPI／SSE 的 Engine 輸出事件；同一 Worker 結果不得同時包含 token ID 與文字增量。
- **AC-02.4** Engine 依 request ID 分派事件；一個請求的錯誤、取消、完成或文字輸出不得串入另一個 request stream。

### REQ-03：Scheduler 與 KV Cache Manager 保留資源所有權

- **AC-03.1** Scheduler 依活動請求上限、等待佇列上限和每輪 token 預算建立 prefill／decode 批次；decode 每 request 每輪最多執行一個 token。
- **AC-03.2** 相容的同 phase Qwen request 由一次批次 Model Runner 呼叫處理；不同 phase 可由同一 Worker RPC 分組，但不得把不相容的 Cache 或 attention 狀態混入同一次 forward。
- **AC-03.3** 共用 `KVCacheManager` 管理 Engine 邏輯 block reservation；Qwen Worker pool 管理與 block ID 對應的實體 GPU page，Mage-VL 仍使用 Transformers Cache。
- **AC-03.4** 請求成功、取消、Worker 失敗或 Engine 關閉後，Engine lease 和 Qwen 實體 page 都可回收及供後續請求重用。
- **AC-03.5** Qwen Attention 對 KV page 的 gather／SDPA 實作須如實描述；除非程式直接透過 block table 讀寫實體 page，否則不得稱為 PagedAttention。

### REQ-04：L3 Executor、Worker 與 Platform 隔離執行細節

- **AC-04.1** Engine Core 只依賴 `Executor` 呼叫生命週期、RPC、批次執行、健康資訊和關閉；Engine 不自行建立 Pipe 或操作 multiprocessing process。
- **AC-04.2** `LocalProcessExecutor` 以 `spawn` 啟動本機 Model Worker 子程序，完成 ready handshake 後才回報可執行，並保留既有 timeout、錯誤和 graceful close 行為。
- **AC-04.3** Worker 在自己的程序內載入模型與 Runtime；API Server 不載入模型權重，Engine Core 不持有 GPU tensor。
- **AC-04.4** `Platform` 定義裝置解析、記憶體觀測與資源釋放契約；Qwen YAML 選用已實作的 `torch_cuda`。未知平台必須明確失敗；未實作的 Ray、CPU、NPU 和多 GPU 不得出現在支援清單。
- **AC-04.5** 進程 IPC／執行層的 fake Worker 測試驗證 API Server、Engine Core 與 Worker PID 隔離，並覆蓋 startup、RPC、stream、cancel、timeout、錯誤和關閉。

### REQ-05：L4 Model Runner 以 InputBatch 執行批次模型呼叫

- **AC-05.1** Qwen `InputBatch` 將每輪 token tensor、每 request token 長度、position offset、request ID 和 block table 對齊組裝；數量或長度不一致時在 forward 前拒絕。
- **AC-05.2** 變長 prefill 使用明確 padding 長度資料；每個 request 的 position、Cache block table、sampling state 和輸出依 request ID 對應。
- **AC-05.3** Model Runner 負責模型輸入 tensor、模型 forward、Sampling 和 Worker result；HTTP schema、SSE JSON 和 OpenAI response 不進入 Runner。
- **AC-05.4** 首版 `InputBatch` 每輪建立；不得宣稱持久 GPU buffer、CUDA Graph capture/replay 或 fused Attention 已實作。

### REQ-06：L5 模型包持有模型拓撲與生態相容

- **AC-06.1** Qwen3-0.6B 模型拓撲、Attention、RoPE、RMSNorm、MLP、Sampling、Paged KV storage 及 Safetensors key mapping 均由 `models/qwen3_0_6b/` 持有。
- **AC-06.2** Transformers 的 `AutoConfig`／`AutoTokenizer` 只用於固定 revision 的 config／Tokenizer／Chat Template；Qwen forward 不使用 `AutoModelForCausalLM` 或 `generate()`。
- **AC-06.3** 權重載入驗證固定 repo、revision、必要 key 和 tensor shape；未知或不相容權重在 Worker ready 前明確失敗。
- **AC-06.4** `mage-vl-4b` 與 `qwen3-0.6b` 的上游模型識別碼、專案 alias、執行後端及 API 能力各自登錄；本規格不選定第三個模型。

### REQ-07：設定、健康狀態和失敗回報反映實際能力

- **AC-07.1** 模型、Runner、Scheduler、KV pool、Worker timeout、API 網路和 MediaMTX 的可調參數保存在對應 YAML；密鑰不寫入 YAML 或 Git。
- **AC-07.2** Qwen `/health` 可回報 backend、platform、API／Engine／Worker PID、Scheduler、Engine lease、Worker page 使用和 CUDA memory；GPU 欄位只在 Worker ready 後提供。
- **AC-07.3** Worker 啟動、Executor timeout、Tokenizer 載入、平台不支援、Cache 容量不足和模型輸入不合法均回報明確錯誤；不得靜默切換到未排程的 `generate()`。
- **AC-07.4** Go Gateway、Inference API 和模型 Runner 仍保持程序／責任分離；不把 gateway API Key 或租戶政策移至推理層。

### REQ-08：測試與性能證據對應五層邊界

- **AC-08.1** `tests/inference/` 的無權重測試覆蓋 PreparedChatInput IPC 序列化、EngineOutputProcessor、增量 Detokenizer、InputBatch、Scheduler／Cache、Executor／Worker failure path 和 SSE domain event。
- **AC-08.2** `tests/gateway/`、`tests/inference/`、`tests/end2end/` 維持獨立；Qwen API 改造不得把 Gateway 測試合併至推理套件。
- **AC-08.3** 固定 Qwen3-0.6B snapshot 驗收記錄模型 revision、硬體／軟體版本、非串流回答、文字 SSE 完整內容、finish usage、`[DONE]`、併發隔離和取消後 page 回收；CPU mock 不替代實機證據。
- **AC-08.4** Qwen 性能報告記錄 request／token throughput、TTFT、inter-token latency、request latency 分位數、錯誤率和 peak allocated／reserved VRAM；此階段建立基線，不設硬 SLA。
- **AC-08.5** 本專案不因本規格增加模型精度資料集或 Video-MME 評測。
