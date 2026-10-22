# 推論服務設定與啟動

推論服務由 `launcher.py` 啟動。`model_registry.yaml` 將模型別名連結到推論、網路、媒體設定及模型元件工廠，並登記整合 profile 與 API 能力；API、Engine Core 和 Worker 由共用入口裝配。可先執行 `python src/inference/launcher.py validate` 做無權重／無 GPU 預檢，再啟動服務。一般啟動時，啟動器依登錄表載入路徑；進行實驗或切換設定時，也可以在命令列逐一指定設定檔。

## 設定檔分工

| 設定檔 | 用途 | 主要內容 |
| --- | --- | --- |
| `inference.yml` | 模型與推論行為 | 模型識別碼與 revision、權重載入、精度與量化、生成參數、請求限制、視訊處理及推論最佳化 |
| `network.yml` | API 與串流網路端點 | API 綁定位址與連接埠、MediaMTX 的 RTSP／RTMP 位址，以及回傳給推流端的公開主機名稱 |
| `mediamtx.yml` | MediaMTX sidecar | 推流協定、監聽連接埠及 MediaMTX 回呼 API 設定 |

Mage-VL 與 Qwen3 的服務設定分別位於 `src/inference/models/mage_vl_4b/configs/` 和 `src/inference/models/qwen3_0_6b/configs/`；各模型的 Python 依賴位於對應 `requirements/`。`model_registry.yaml` 保存整合 profile、能力、設定路徑、Runner／Adapter 工廠及可選 API 元件；明確指定命令列設定時，該次啟動會以命令列路徑覆蓋相應的登錄表路徑。

## 推論服務分層

請求主鏈路參考 vLLM V1 的五個核心職責，採用 API Server、Engine Core、GPU Worker 三個程序。API Server 透過共用 `EngineClient` 與 Engine Core 使用本機 Pipe 通訊；Engine Core 管理 Scheduler、KV lease 並透過既有 Worker IPC 呼叫 GPU Worker。這不是 vLLM 後端，也沒有使用 vLLM attention kernel。

| 元件 | 程式位置 | 職責 |
| --- | --- | --- |
| API Server | `api/app.py`、`api/service_app.py`、`api/routes/`、`api/schemas.py`、`api/serializers.py` | 共用 app 裝配、HTTP 路由、schema、請求本文上限及 OpenAI/SSE response 序列化 |
| Engine Core | `engine/engine.py`、`engine/client.py` | 獨立程序中的推論生命週期；API 端 `EngineClient` 提供 IPC proxy |
| Scheduler | `engine/scheduler.py` | 完整 prompt prefill 為一輪、decode 每輪一個 token；活動請求輪轉，佇列與序列數由 YAML 控制 |
| KV Cache Manager | `engine/kv_cache.py` | 預留 prompt+輸出 token 邏輯 block，計算估算值，完成／取消後回收 block ID |
| Executor／Worker | `engine/executor.py`、`engine/worker.py` | `LocalProcessExecutor` 透過 Pipe 管理本機 Worker 子程序；`InProcessModelWorker` 保留程序內測試與整合用途 |
| 模型 Adapter／Runner | `models/mage_vl_4b/adapter.py`、`model_runner.py` | Adapter 負責 Mage-VL 輸入及結果適配；Runner 在 Worker 子程序載入模型並逐步執行生成 |
| 模型輸入處理 | `models/mage_vl_4b/processor.py` | Mage-VL 專屬 prompt、影像／影片 Processor 與生成參數 |
| 模型元件 | `models/mage_vl_4b/components/` | 固定 revision 的模型設定、視覺塔、多模態組裝、生成頭、逐步生成狀態及替換登記 |
| 純文字模型 Adapter／Runner | `models/qwen3_0_6b/adapter.py`、`model_runner.py`、`model/` | Hugging Face 相容輸入、Qwen3 本地 PyTorch 計算圖、批次 prefill/decode、GPU 實體 KV page |
| 通用媒體／Live session | `media/video.py`、`media/live_sessions.py` | 上傳影片暫存、通用 metadata／影格讀取及即時 session 生命週期 |

## Qwen3-0.6B 五層推論架構

L1–L3 的 API、Engine 與 Worker 裝配由 Mage-VL 和 Qwen3 共用；Qwen3 此處的 L4／L5 展示自有 PyTorch 計算圖，Mage-VL 則保留 Transformers Runner 與模型元件。需求、設計與驗收證據見 [Inference Spec](../../docs/Specifications/inference/)。

| 層 | 職責 | 目前實作位置 |
| --- | --- | --- |
| L1 接口與門面 | API schema、Tokenizer、Detokenizer、SSE／JSON 序列化與 Engine IPC client | `api/`、`engine/client.py`、Qwen `input_processor.py` |
| L2 編排與調度 | Engine Core 請求生命週期、Token Scheduler、邏輯 KV reservation、輸出事件分派 | `engine/engine.py`、`scheduler.py`、`kv_cache.py`、`output_processor.py` |
| L3 執行抽象與 Worker | 本機程序啟動、Worker RPC、硬體與記憶體觀測 | `engine/executor.py`、`worker.py`、`platform.py` |
| L4 模型運行時 | Qwen 輸入批次、模型 forward、Sampling 與輸出 token ID | Qwen `model_runner.py`、`model/input_batch.py`、`sampler.py` |
| L5 模型與算子 | Qwen 模型拓撲、Attention、RoPE、權重映射及上游格式相容 | Qwen `model/`、`utils/` |

Qwen `AutoConfig`／`AutoTokenizer` 僅用於固定 revision 的設定、Tokenizer 與 Chat Template；模型 forward 由本地 PyTorch 計算圖執行。批次 prefill/decode 的 Engine block ID 對應 GPU Worker 中的 K/V page。API 保留 `/v1/chat/completions`、`qwen3-0.6b` alias、逐 token SSE、finish usage 和 `[DONE]`；Worker 回傳 token ID，API 負責串流及非串流 Detokenize。客戶端斷線時 response 監聽 ASGI disconnect，透過 `EngineClient` 取消生成，並在 shielded cleanup 中等待 Worker 釋放 page。Attention 目前將 paged KV gather 成連續張量，因此不稱為 PagedAttention kernel。

目前 Qwen YAML 設定 1024 個 16-token pages；以 BF16 計算，K/V pool 預配置約 1.75 GiB。`/health` 回報執行 backend、Scheduler、Engine lease、Worker 實體 page 使用量與 GPU 峰值記憶體。Qwen API 仍只提供文字 Chat，不包含影片或 Live 路由。

Scheduler 為活動 chat、影片及 Live 工作維護等待／執行佇列，依 `max_num_seqs` 和 `max_num_scheduled_tokens` 建立 prefill、decode 或混合批次；滿載時等待佇列耗盡後回 `429`。啟用 `prefill_chunk_size_tokens` 時，長 prompt 會按 offset 分輪追加至同一 Cache。首段會延伸至涵蓋全部視覺 token，確保 Mage-VL 的影像／影片特徵只在首次 forward 編碼；若該不可拆分前綴超過每輪 token 上限，請求會在排程前拒絕。`batch_wait_seconds` 提供短暫請求合併窗口。

Engine Core 背景迴圈取得 Scheduler 批次、一次呼叫 Worker，依每個 request ID 更新狀態並把結果送回對應串流。Mage-VL Runner 對相容的同階段請求使用一次多序列模型 forward，並將 logits 與 Cache 分別還原到各 request；混合批次在同一 Worker 呼叫內按階段執行 prefill／decode 子批次，不會把不同 Cache 狀態強塞進同一模型 forward。不相容的輸入會退回逐 request forward。分塊 prefill 只追加當輪 token，完成最後一塊後才取樣第一個 token。Mage-VL 維持逐 request 的 `text` 事件；Qwen3 回傳 `token_id`，由 L1 Detokenize 後形成相同 OpenAI 文字 SSE。

Engine 對每個生成請求依序執行輸入準備、KV 容量預留、Scheduler 准入、Worker prefill／decode、狀態更新及事件發布。成功、拒絕、取消、Worker 逾時或異常均經同一個 `finally` 收尾，回收 Scheduler 名額與邏輯 KV reservation，並取消可能部分建立的 Runner 狀態。`stream_chat()` 將 Worker 結果依 request ID 發成 Engine event；Qwen3 使用 token ID、Mage-VL 使用文字增量；API 路由將其序列化為 OpenAI SSE chunk，取消 HTTP 串流時會關閉生成 iterator 以觸發收尾。

Mage-VL Worker 的 `RequestCacheManager` 以 request ID 持有 Transformers Cache 物件，Engine block ID 仍是邏輯 reservation。Qwen3 則使用模型專屬 `PagedKVCache` 實體 tensor pool；共用 Engine `KVCacheManager` 分配 page ID，Worker 驗證租約並在 finish／cancel／close 時釋放 page 所有權。這是物理儲存分頁，不是直接以 block table 計算的 PagedAttention kernel。

`engine/batch_planner.py` 已由線上 Scheduler 呼叫，按序列上限及每輪 token 預算產生 chunked prefill、batched decode 和混合階段批次。Mage-VL 相容輸入依其 Runner 的 Cache 管理方式批次 forward；Qwen3 同階段 request 由本地 Runner 一次批次 forward。KV Cache Manager 目前逐請求建立／釋放租約；尚未實作 prefix cache、page eviction/offload 及自訂 attention kernel。

Mage-VL 的 Engine lease 不代表 GPU 實體分頁；其 KV tensor 仍由 Transformers 動態配置。Qwen3 的 Engine 容量與 Worker page pool 從同一份模型 YAML 建立，啟動時核對 block 數與 block size。Worker 的 page pool 預先配置於 GPU，`/health` 分別回報 lease 估算值、實體 pool 使用量及實際 CUDA peak allocated/reserved bytes。

Engine Core 與 GPU Worker 都以 Python `spawn` 建立；API import 不載入 Mage-VL 模型。`runtime.uvicorn_workers` 必須為 `1`，避免重複建立整組 Engine Core／GPU Worker。Engine Core 啟動與 RPC 逾時分別由 `runtime.engine_core_startup_timeout_seconds` 和 `runtime.engine_core_request_timeout_seconds` 設定；`/health` 回報 API、Engine Core 與 GPU Worker PID。Chat 的 Qwen token ID／Mage-VL 文字增量事件經 EngineClient 傳回 API；Qwen3 由 API Detokenize 後序列化成 OpenAI SSE chunk，包含 finish usage 與 `[DONE]`；Live session 與媒體擷取留在 API 程序，其模型分析請求經 EngineClient 提交。Runner 支援逐步 `past_key_values` 解碼、greedy／temperature／top-k／top-p／min-p 及重複懲罰；beam search、自訂 generation kwargs 和非預設 Cache implementation 暫不支援。若固定 revision 不提供必要介面，Worker 會明確報錯，不回退到不受 Token Scheduler 控制的完整 `generate()`。

## 新增模型的最小步驟

新模型不需要建立專屬 `app.py` 或 `engine_core.py`。共用 `api/service_app.py` 依登錄項目建立 API，`engine/model_bootstrap.py` 建立 Engine Core、Executor 和 Worker。符合受限標準文字契約的模型可省略 Runner／Adapter／InputProcessor 工廠，使用 `api/default_chat_processor.py` 和 `engine/standard_text.py`；其他模型提供自己的 Runner／Adapter 與必要模型擴充。模型程式可透過共用 `engine.model_config.selected_config("INFERENCE_CONFIG")` 取得啟動設定；若需整理模型特有衍生欄位，再建立 `config.py`。啟動器仍會將 `src/inference/` 加入 Python import path；不把 `src/`、`inference/` 或 `models/` 根目錄當成 Python package。

逐步生成模型 Adapter 要符合共用 prepare／prefill／decode／finish／cancel 操作；Runner 持有模型 request state 與 cache。標準文字路徑只接受單裝置 causal LM、文字 `chat_template` 和逐步可重用 `past_key_values`；預設 `execute_batch()` 逐項 forward，不代表跨 request batching。多模態、encoder-decoder、自訂 Cache 或模型拓撲使用 `custom`，並由模型包適配。Live 模型才提供 Live manager factory。模型原生 cache 和 batch 語義由模型 Adapter／Runner 適配，不預設每個模型都能使用 Qwen 實體 KV page。若模型不支援逐步執行，需先新增並驗收適配方案，不能把整段生成標記為 token-scheduled。

新模型目錄、工廠簽章、登錄欄位和啟動驗證步驟見[新模型接入操作指南](model-integration.md)。只有 API 行為需要改變時才修訂對外 API Spec；模型拓撲、權重格式、Tokenizer 和輸入前處理留在模型包。

模型內部實作與最佳化應放在該模型的 `components/`，由模型自己的 registry 登記明確 slot 和 implementation；不要讓共用 Engine 綁定特定模型類別。Mage-VL 的固定 revision 模型程式已按設定／輸出、視覺元件、視覺塔、多模態組裝、生成頭和 StreamMind Gate 拆在 `components/`；`model_loader.py` 直接建構本地 `MageVLForConditionalGeneration`，不再從模型快照動態載入模型類別。revision 不符合本地模型程式固定版本時，載入器會拒絕啟動。Processor 仍透過 Transformers `AutoProcessor` 從同一個本機快照載入，因其提示模板及圖片／影片處理器是模型專屬實作。上游程式來源 revision 與授權記錄在各元件檔案及 `MAGE_VL_UPSTREAM_LICENSE.txt`。`model_loader.py`、`stepwise_generation.py` 與 `component_registry.py` 也與 `model_runner.py` 分開；Runner 保留 orchestration facade。

Mage-VL Registry 只接受模型包程式碼已登記的替換，YAML 不做動態 import。`configs/inference.yml` 的 `model.components.replacements` 預設為空；本地已有固定 revision 的上游模型實作，但尚無最佳化替換模組。錯誤選項會在大型權重載入前拒絕。HF 原生 `attn_implementation`、量化及 `device_map` 是載入前設定；Registry 替換在載入後執行，不能降低載入權重時的峰值顯存。未來若要改善載入峰值，需把替換接入模型建構或權重載入流程。

新增模型時，將模型設定、權重載入器、模型特有的 vision／language 元件、輸入處理和生成邏輯放在 `models/<model_alias>/`，共用 `api/`、`engine/` 和 `media/` 維持模型無關。Mage-VL 元件替換先在 `components/` 寫明確用途的實作，再由 Registry 登記 slot、implementation、factory 及必要的權重轉換函式，最後才由該模型的 inference YAML 選取。替換 registry 目前在本地模型載入後執行；先以假 module tree 驗證路徑與 state dict，任何真實權重或 GPU 效能結論都要另行實測。

目前登錄 `mage-vl-4b` 與 `qwen3-0.6b`。Qwen3-0.6B 是純文字模型，僅註冊聊天 API，沒有影片理解或 Live session 路由；它使用獨立 API 連接埠 `8011`，Mage-VL 預設仍為 `8001`。Qwen Runner 對同 phase 的變長 request 執行批次 prefill/decode；不同 phase 在同一 Worker RPC 中分組處理。批次輸出仍按 request 分離並逐 token 經 SSE 回傳。

兩模型目前的 profile 均為 `custom`。`api.capabilities` 明確聲明聊天輸入、文字串流、`text_delta`／`token_ids` 輸出、影片上傳與 Live；共用 API 會依登記啟用路由，聊天請求的輸入模態則在入口驗證。`standard` profile 有共用預設 Runner，但只支援明確列出的純文字契約；不是任意 AutoModel 相容層。部署前需以固定 revision 真實 smoke 驗證 Cache 延續、finish/cancel 和串流輸出。

`inference.yml` 的 `api.max_chat_request_bytes`、`api.max_live_request_bytes`、`api.max_video_request_bytes` 與 `api.max_video_bytes` 分別控制聊天本文、live 建立／MediaMTX callback 本文、視訊 multipart 本文及單一視訊檔案上限。Gateway 的相應本文上限不得高於推論設定。Mage-VL 聊天需要至少一張 `image_url`；Qwen3-0.6B 僅接受文字內容，收到圖片會回 HTTP 400。Gateway 的模型能力與上游位址須同步登記；影片可用 `?model=<alias>` 選擇上游，Live 建立可用 JSON `model` 選擇。

## 完整啟動順序

啟動依賴順序如下：

**啟用推論環境 → 選擇並啟動已登錄模型 API → 確認 API 就緒 →（Mage-VL 即時推流時）啟動 MediaMTX → 客戶端呼叫 API**

### 啟用環境並確認 GPU

在 WSL 專案根目錄執行。以下流程假設 `magevl-transformers` Conda 環境與鎖定依賴已建置完成：

```bash
conda activate magevl-transformers
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CUDA unavailable')"
python src/inference/launcher.py list
python src/inference/launcher.py validate
```

確認 PyTorch 顯示 CUDA 可用，且登錄表中的模型與設定檔狀態正常。若是首次建置環境，請先依 `src/inference/models/mage_vl_4b/requirements/requirements-lock.txt` 和目標主機的 CUDA 環境完成套件安裝，再繼續以下步驟。

### 啟動 Mage-VL API

確認網路可連線至 Hugging Face 後，在終端機 A 從專案根目錄啟動服務。此命令明確指定模型及三個設定檔；API 與 Engine 入口由共用裝配器載入，模型差異取自 `model_registry.yaml`：

```bash
python src/inference/launcher.py start \
  --model mage-vl-4b \
  --inference-config src/inference/models/mage_vl_4b/configs/inference.yml \
  --network-config src/inference/models/mage_vl_4b/configs/network.yml \
  --media-config src/inference/models/mage_vl_4b/configs/mediamtx.yml
```

啟動器會先載入設定，再由 Hugging Face Hub 準備 `inference.yml` 指定的固定 revision。模型快照會存放在專案的 `checkpoints/`；快取已有完整快照時會直接重用，缺少快照時會自動下載。下載及模型載入期間 API 尚未就緒，因此首次啟動需要等待較久。此命令會持續佔用終端機 A，請保持程序運行。若只需使用登錄表的預設設定，可簡寫為 `python src/inference/launcher.py start --model mage-vl-4b`。

### 啟動 Qwen3-0.6B 純文字 API

先用 `scripts/download.py` 下載固定 revision 至 `checkpoints/Qwen3-0.6B-c1899de/`，再啟動 Qwen API。它使用獨立連接埠 `8011`，不會取代 Mage-VL 的 `8001`：

```bash
python src/inference/launcher.py list
python src/inference/launcher.py start --model qwen3-0.6b
```

Qwen 僅註冊 `/health`、`/v1/models` 和文字版 `/v1/chat/completions`，沒有影片或 Live 路由，不需要 MediaMTX。以純文字非串流請求確認服務：

```bash
curl --fail http://127.0.0.1:8011/health
curl --fail http://127.0.0.1:8011/v1/models
curl --fail http://127.0.0.1:8011/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-0.6b","messages":[{"role":"user","content":"用一句話說明推理引擎。"}],"max_tokens":64}'
```

文字 SSE 使用同一端點，並在 request 設定 `stream: true`。Qwen Worker 逐 token 生成，API 仍輸出 OpenAI SSE chunks。

### 確認 API 就緒

開啟終端機 B，等終端機 A 完成模型載入後檢查健康狀態：

```bash
curl --fail http://127.0.0.1:8001/health
curl --fail http://127.0.0.1:8001/v1/models
```

`/health` 回報 `status: ok` 後，再從客戶端呼叫圖片理解或視訊上傳 API。API 預設監聽 `127.0.0.1:8001`，綁定位址與連接埠由 `network.yml` 控制。

### 即時推流時再啟動 MediaMTX

只有使用 `/v1/live/*` 即時推流 API 才需要 MediaMTX；圖片理解及上傳視訊檔案不需要它。必須先讓 API 就緒，再於終端機 B 執行 MediaMTX，因為 MediaMTX 會呼叫 API 的授權回呼：

```bash
mediamtx src/inference/models/mage_vl_4b/configs/mediamtx.yml
```

MediaMTX 啟動後，客戶端才建立即時工作階段、依回應中的 RTSP／RTMP 位址推流，並透過 API 取得理解結果。若調整 `network.yml` 中的 API 或串流端點，請同步確認 `mediamtx.yml` 的授權回呼及監聽連接埠。

### 停止服務

若有使用即時推流，先在終端機 B 按 `Ctrl+C` 停止 MediaMTX，再在終端機 A 按 `Ctrl+C` 停止 API 並釋放 GPU 資源。若只使用圖片或視訊上傳 API，停止終端機 A 的 API 即可。

## 模型與 API

模型快取位於專案的 `checkpoints/`，Hugging Face 權重及快取檔案會由 `.gitignore` 排除；`checkpoints/val.md` 與 `checkpoints/val.xlsx` 例外保留為可追蹤的工作流文件。`runtime.local_model_path` 可明確指定本機快照目錄，服務會直接載入該路徑；未設定時才使用 Hugging Face Hub 快取，並由 `runtime.local_files_only` 控制是否允許下載。

目前登錄並實作的模型執行方式為 Transformers，由 `default_backend` 及 `inference.yml` 中的 `backend.name` 標示；vLLM 僅作為 Engine、Scheduler、Worker 與 cache 職責的架構參考，並未接入。API 路由如下：

- `GET /health`
- `GET /v1/models`
- `POST /v1/chat/completions`：文字／圖片請求，支援 SSE 串流文字輸出。
- `POST /v1/videos/understand`：上傳視訊檔案並取得理解結果。
- `/v1/live/*`：即時視訊工作階段 API；需另行啟動 MediaMTX。

`src/inference/models/mage_vl_4b/requirements/requirements.txt` 列出直接依賴，`requirements-lock.txt` 固定環境套件版本。端到端 Python 客戶端範例位於專案根目錄的 `example.py`，透過 Go Gateway 呼叫推論服務。

模型結構、輸入 tensor、KV cache 與 GPU 資訊的手動探索工具位於 [`scripts/mage_vl_4b/test.py`](../../scripts/mage_vl_4b/test.py)，執行時須明確傳入 `--config` 與 `--model-path`，以及輸入、問題和報告路徑；不會從腳本位置推導專案目錄或預設 YAML。相對路徑以啟動命令時的目前工作目錄解析。工具從 YAML 載入模型、輸入限制和生成預設值，不從 `src/` 匯入模組，並強制離線使用明確指定的本機權重；它尚未在真實權重／GPU 上執行。使用方式見 [`scripts/README.md`](../../scripts/README.md)。
