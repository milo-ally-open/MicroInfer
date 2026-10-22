# 技術設計：五層推理引擎

> 提案：[proposal.md](proposal.md)｜行為規格：[spec.md](spec.md)｜任務與驗收：[tasks.md](tasks.md)

## 參考資料與本地證據

vLLM V1 官方架構將 API Server、Engine Core 和 GPU Worker 分成不同程序；API Server 執行輸入處理並串流結果，Engine Core 執行 Scheduler、管理 KV Cache 並協調 Worker，Worker 載入模型並執行 forward。Model Runner 負責模型輸入張量和模型執行細節。參考：[vLLM V1 Architecture Overview](https://docs.vllm.ai/en/v0.19.0/design/arch_overview/)。本設計採用這些責任邊界，不複製 vLLM 的內部類別、IPC、Executor 或 Kernel，也不聲稱 vLLM 相容。

[nano-vLLM README](https://github.com/GeeeekExplorer/nano-vllm) 將自身描述為精簡、可讀的 vLLM 實作，展示其以較小程式結構承載 Engine、批次和最佳化概念。本專案參考其「用可讀的核心流程學習／組織 Engine」方向；不把 README 中宣稱的功能或 benchmark 當成本專案能力，也不將 nano-vLLM 加入執行依賴。

本地現況以提交 `0cb3c56` 的 Qwen3-0.6B 原生計算圖為基線：固定 revision 的 config／tokenizer／Safetensors 載入、自有 Qwen3 PyTorch 模型、批次 prefill/decode、Worker 實體 KV pool 和文字 SSE 已存在。改造前的 RTX 4060 數值驗收保存在 `checkpoints/val.md` 與 `checkpoints/val.xlsx`，不能替代本次 token event／Executor／InputBatch 的回歸驗收。Mage-VL 仍使用 Transformers 模型路徑。此處不把 SGLang、vLLM backend、PagedAttention Kernel、Ray 或第二個原生模型寫成已完成。

## 五層責任與相依

| 層 | 責任 | 本地程式邊界 | 不擁有 |
|---|---|---|---|
| L1 接口與門面 | HTTP、OpenAI schema、模型輸入 Tokenize、增量 Detokenize、SSE 和 Engine IPC client | `api/`、`engine/client.py`、模型包的 `input_processor.py` | GPU tensor、Scheduler、KV pool、模型 forward |
| L2 編排與調度 | Engine Core 請求狀態、准入、Scheduler、邏輯 Cache reservation、Worker 結果事件化 | `engine/engine.py`、`scheduler.py`、`kv_cache.py`、`output_processor.py` | HTTP/SSE schema、實體 GPU tensor、模型拓撲 |
| L3 執行抽象與 Worker | Executor 管理本機程序和 RPC；Platform 解析裝置與資源觀測；Worker 程序載入 Runner | `engine/executor.py`、`platform.py`、`worker.py` | request policy、模型輸入格式、HTTP |
| L4 模型運行時 | Qwen Runner 組裝輸入批次、forward、Cache 存取和 Sampling | `models/qwen3_0_6b/model_runner.py`、`model/input_batch.py`、`sampler.py` | OpenAI request／response、模型別名路由 |
| L5 模型與算子 | Qwen 模型計算圖、層、Attention、RoPE、權重映射和上游相容 | `models/qwen3_0_6b/model/`、`models/qwen3_0_6b/utils/` | 共用 HTTP API、Engine 佇列和租戶政策 |

模型生態相容是跨層責任：API 端的 Chat Template／Tokenizer 在 L1 依模型建立輸入；模型 config、Safetensors 權重映射和計算圖在 L5。通用影片讀取與 Live session 仍由既有 `media/` 持有；Mage-VL 的影片採樣和 Processor 留在 Mage 模型包。五層以程式職責劃界，不代表五個新 package 或五個 OS process。

## 程序拓樸

```text
API Server process (L1)
  FastAPI routes / schemas / SSE serializer
  Qwen3InputProcessor + IncrementalDetokenizer
  EngineClient
       │ local duplex Pipe
       ▼
Engine Core process (L2)
  InferenceEngine → TokenScheduler → EngineOutputProcessor
       │
       └── LocalProcessExecutor (L3) ── local duplex Pipe
                                          │
                                          ▼
Model Worker process (L3)
  Qwen3ModelAdapter → Qwen3ModelRunner (L4) → Qwen3ForCausalLM (L5)
  TorchCudaPlatform + GPU KV page pool
```

仍為 API Server、Engine Core、單一 Model Worker 三個程序。`LocalProcessExecutor` 目前只管理一個 `spawn` 子程序；介面為未來更換執行拓樸留出位置，沒有多 GPU、Ray 或 Worker pool 實作。Engine Core 不匯入模型包，模型包 factory 只在對應程序建立該模型所需元件。

## 請求與文字串流

1. Qwen app 將 `Qwen3InputProcessor` 注入共用 `create_app()`。API lifespan 先從 YAML 指定的本機 snapshot 載入 Tokenizer，再啟動 EngineClient；此處只載入 CPU tokenizer，不載入權重或 GPU Runtime。
2. Chat route 驗證 model alias，呼叫模型 `InputProcessor` 形成 `PreparedChatInput(model_alias, input_ids, sampling_params)`，再透過 EngineClient 傳入 Engine Core。資料結構只包含可 pickle 的小型 Python 資料，不包含 tokenizer、GPU tensor 或模型 Cache。
3. Engine Core 將 PreparedChatInput 傳給 Worker Adapter。Adapter 建立 request state，Engine 保留 token 容量 reservation，Scheduler 排入 prefill/decode 批次。
4. Worker 的 Qwen Model Runner 以 `InputBatch` 組裝 token tensor、長度、position offset 和 block table，呼叫本地 Qwen forward 及 Sampler。生成步驟只回傳 token ID；Mage-VL 仍回傳文字增量。
5. Engine `EngineOutputProcessor` 依 request ID 將 token ID 或文字增量轉成 Engine stream event。EngineClient 將事件送回 API process。
6. Qwen API 的 `IncrementalDetokenizer` 累積 token IDs 並對完整前綴解碼；當前前綴若含不完整 Unicode replacement，先緩衝該位置之後的文字。可安全輸出的 delta 立即序列化成 OpenAI SSE chunk；finish 時 flush 剩餘內容，再送 finish usage 和 `[DONE]`。已送出的前綴若被 tokenizer 改寫，輸出處理器明確失敗，避免靜默重複／漏字。
7. HTTP SSE 的取消仍由 route 關閉 EngineClient stream，依 Engine cancellation path 停止 request 並釋放 Scheduler／Cache／Worker 狀態。一般非串流請求由 Worker 回傳 `InferenceResult`；Qwen token IDs 在 API L1 解碼成 `content` 後，再沿用現有 OpenAI response serializer。

InputProcessor／Tokenizer 位於 API process，因此 Qwen API 和 Worker 各保留一份小型 tokenizer instance；這是 Tokenize 與模型載入隔離的首版成本。固定 snapshot 驗收需確認其 Chat Template 結果與 Worker 最終 decode 使用同一 revision。

## L2 排程與 KV 所有權

Engine Core 保持接收、准入、排程、Executor 執行、更新狀態、發佈事件和收尾流程。`EngineOutputProcessor` 只轉換 Worker result；request terminal state、finish operation、取消和 Cache release 仍由 Engine Core 管理。

Token Scheduler 繼續依 `max_num_seqs`、等待上限、`max_num_scheduled_tokens`、prefill chunk 和 batch wait 設定建批；批次按 request ID 回傳結果。混合 prefill/decode 可同一次 Worker RPC 傳輸，Runner 仍按 phase 分批 forward。Sampler 保持在 L4 Worker，模型產生的 token 不交由 API 重新採樣。

`KVCacheManager` 是 L2 邏輯資源管理器。Mage-VL 對應 Transformers Cache request state；Qwen Engine block ID 映射到 Worker 內的實體 page pool。Qwen Attention 目前 gather paged K/V 後交給 PyTorch SDPA，因此本地 pool 具有物理分頁存儲，但不是直接 block-table kernel。Engine 的 reservation／容量統計與 Worker 實際頁面使用量分開回報。

## L3 Executor 與 Platform

`Executor` 是 Engine Core 使用的 process／execution contract。`LocalProcessExecutor` 擁有 spawn、Pipe RPC、ready handshake、request timeout、error transport 和 close。`worker.py` 保留子程序內執行模型 Adapter 的 `ModelWorker` contract 與 `InProcessModelWorker`；不得讓 Engine Core直接依賴 Worker process 細節。

`Platform` 是 Worker Model Runner 使用的 device／memory contract。Qwen YAML 的 `runtime.platform: torch_cuda` 由 `create_platform()` 建立 `TorchCudaPlatform`；裝置選擇、CUDA 可用性、VRAM／peak memory 觀測及 CUDA cache 清理經該介面執行。未知名稱在 Runner 載入前明確失敗。此設計只落實已用到的 CUDA 平台，不建立空的 Ray／NPU／CPU 實作。

## L4 InputBatch 與模型運行時

`InputBatch.build()` 依排程工作項目建立本輪的 padded token ID tensor，並檢查 request 數量、token length、position offset 和 block table 一一對應。Qwen forward 仍接收 per-request 長度／偏移／block tables，padding 位置不參與有效 token 計算。

InputBatch 暫時每輪動態分配 GPU tensor。後續若性能樣本確認 CPU tensor 組裝或分配是瓶頸，可將容量預配置、Pinned Memory、非同步 H2D 和 CUDA Graph 作為獨立最佳化，需先保持相同的 request mapping／SSE／Cache acceptance。首版不加入這些未測機制。

## L5 Qwen 模型與生態相容

Qwen 模型拓撲及 Attention 等算子留在模型包；模型專屬 `utils/` 提供上游 config／tokenizer／固定 revision 權重格式適配等輔助功能。Safetensors loader 在 Worker ready 前驗證 metadata、revision、必要 tensor 和 shape。模型 Registry 由 `model_registry.yaml` 登記 alias、設定路徑、對外 API ID、Runner／Adapter 工廠和選用 API 擴充；共用 route 不以模型 alias 分支。

## 新模型接入與共用裝配

`api/service_app.py` 讀取 Launcher 設定的模型 alias，建立共用 `EngineClient`、API request limits、輸入處理器和選用 Live manager。`engine/model_bootstrap.py` 從 Registry 載入模型 Runner／Adapter 工廠，建立 `InferenceEngine`、`LocalProcessExecutor` 和 `InProcessModelWorker`。Runner／Adapter 工廠使用 `module:attribute` 字串；工廠僅在相應程序中載入，因此 API process 不會匯入模型 Runner 或 GPU Runtime。

Registry 的 `api.model_id` 是推論 API 接受及回傳的識別碼；上游 model repo ID 留在 `inference.yml` 的 `model.id`。`api.capabilities` 明確列出聊天輸入模態、文字串流、`text_delta`／`token_ids` 輸出、影片上傳及 Live 能力；API app 依此掛載路由並拒絕不支援的輸入。`input_processor` 與 `live_session_manager` 是依能力需要登記的工廠。特殊前處理、Tokenizer、權重載入、模型拓撲和 cache 邏輯仍由模型包負責。

新模型最少實作 Runner／Adapter 並登記工廠和 YAML；不需寫模型專屬 `app.py` 或 `engine_core.py`。逐步生成 Adapter 維持 Engine 的 prepare／prefill／decode／finish／cancel 契約，Runner 持有模型 cache state 並依 request ID 回傳批次結果。共用 KV Manager 的租約不替模型宣稱支援相同的物理 cache 或 batch 語義。完整操作步驟及工廠簽章見[新模型接入操作指南](model-integration.md)。

Mage-VL 與 Qwen 均由共用入口裝配；模型目錄不保留未使用的 `app.py`／`engine_core.py` 包裝檔。

### 標準文字模型預設元件

`integration_profile: standard` 可省略 `components.runner`、`components.adapter` 和 `api.input_processor`。API Server 使用 `api/default_chat_processor.py`；Worker 使用 `engine/standard_text.py` 中的 `StandardTextModelRunner`／`StandardTextModelAdapter`。Registry 仍保留逐項覆寫入口，不要求標準模型先複製一組包裝類別。

預設 Runner 以 `AutoModelForCausalLM` 載入固定 revision，僅支援單一已登記 `torch_cuda` 裝置、decoder-only causal LM、模型提供可重用的 `past_key_values`、文字 `chat_template` 和 token ID 輸出。API 處理器只接收文字，不接受圖片、影片或 Live。Scheduler 提交的 `execute_batch()` 仍按 request 逐項呼叫 forward，各 request 的 Cache／生成狀態分開持有；這是可用的標準接入路徑，不代表連續批次 forward、實體 block cache 或 PagedAttention。

預檢以 YAML 與 AST 確認 capability／方法合同，不 import 工廠或權重。Worker 在模型載入前再次檢查實例方法；如架構使用 encoder-decoder、`device_map` 多裝置、不可重用 Cache，或需要自訂採樣／影音流程，應改登記 `custom` 並提供模型 Runner／Adapter。即使預檢通過，將模型標成標準相容前仍需以固定 revision snapshot 實際驗證 Cache 延續和 token 輸出。

## 模型接入 profile 與能力預檢

`integration_profile` 分成 `standard` 與 `custom`。`standard` 使用共用的受限文字 Runner，適用於符合本設計中 decoder-only、單裝置、文字 Chat Template 與可延續 `past_key_values` 契約的模型；它不代表任意 `AutoModel` 都相容。`custom` 代表模型包需擁有專屬計算圖、權重轉換、影音前處理、Tokenizer 或 Cache 行為。Profile 是接入方式標記，不代表套件或模型名稱保證相容。目前 Mage-VL 和 Qwen3-0.6B 都是 `custom`；專案已有標準文字預設 Runner，但尚無任意模型通用相容層。

`launcher.py validate` 為無權重預檢：載入登錄與 YAML，檢查 profile、API capability、Runner／Adapter／選用工廠模組、`token_ids` 解碼依賴及 Live 設定路徑；它不匯入工廠、不下載或載入 snapshot、不建立 CUDA context。`tests/inference/model_contract.py` 使用 AST 檢查方法存在性；這是結構合同，不取代固定 revision 的模型／裝置整合驗收。跨服務設定另由 `tests/end2end/test_config_compatibility.py` 比對 Gateway 與 Inference 的 alias、upstream model ID、能力及 network endpoint。

## 設定與營運

Qwen `configs/inference.yml` 保存 model repo／revision、dtype/device、`runtime.platform`、本機模型路徑、Context、generation、Scheduler、KV block、Worker 和 Engine timeout。網路端點仍來自 `configs/network.yml`。API／Gateway 邊界不變；MediaMTX 只供 Mage-VL Live flow。健康資訊如實區分 API、Engine Core、Worker、邏輯 reservation、實體 KV page 和 GPU memory。

錯誤按所在邊界傳遞：InputProcessor 的非法模型輸入回 4xx；Engine 准入／Cache 容量錯誤沿既有 `InferenceError` 映射；Executor／Worker 啟動和 RPC 超時標為服務錯誤並走清理。Worker 不得在初始化失敗後回退到 Transformers `generate()`。

## 驗收策略

- 單元／契約：`tests/inference/` 驗證 API input contract、detokenizer、EngineOutputProcessor、InputBatch、Executor 和各種 cleanup；不需下載模型。
- API 兼容：對 Qwen 呼叫一般 Chat、SSE、併發 SSE、非法輸入和 disconnect；Mage-VL 的舊文字增量路徑保留回歸檢查。
- 固定 snapshot：確認 Qwen input tokenization、native model output、逐 token event 合併文字、SSE finish usage／`[DONE]` 和取消後 Worker page 回收。
- 性能：在推理 API 驗收後記錄 request／token throughput、TTFT、inter-token latency、request latency p50/p95、錯誤率和 GPU peak memory；與端到端 Gateway performance test 分開，不設定硬 SLA。
- 不執行資料集精度測試；vLLM/nano-vLLM 公開結果不作為本地性能證據。
