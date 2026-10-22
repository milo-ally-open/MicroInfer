# 技術設計：五元件最小 Inference Engine

> 提案：[proposal.md](proposal.md)｜行為規格：[spec.md](spec.md)｜任務／驗收：[tasks.md](tasks.md)
> 本機可驗收範圍：Python 契約測試、子程序 IPC、Scheduler、cache reservation、模型元件 registry mock 和語法；固定 revision 程式、設定及權重快照已在本機，模型定義已拆入模型包；未載入權重或驗證 CUDA 行為。

## 調研依據與已知邊界

vLLM 官方架構將 API Server、Engine Core、GPU Worker 分程序運行；Engine Core 執行 Scheduler、管理 KV cache 並驅動 Worker。其 V1 Scheduler 為每輪分配 token budget，並由 cache manager 管理 KV block。參考：[Architecture Overview](https://docs.vllm.ai/en/latest/design/arch_overview/)、[V1 Scheduler API](https://docs.vllm.ai/en/latest/api/vllm/v1/core/sched/scheduler/)。本專案採用職責分離與逐輪排程概念，不移植 vLLM 的 executor、IPC、attention kernel、block table 或宣稱相容。

Mage-VL 固定 revision 的官方配置將 `MageVLForConditionalGeneration`、`MageVLProcessor` 和 `MageVLVideoProcessor` 登記為自訂程式；視覺編碼器有 24 層，文字端是 36 層 Qwen3，使用 8 個 KV heads、head dimension 128。模型有獨立視覺 patch embedding／attention／MLP／merger，語言 attention／MLP／embedding 及輸出 head；程式提供 `prepare_inputs_for_generation` 與 `past_key_values`。依據：[固定 revision config](https://huggingface.co/microsoft/Mage-VL/blob/d88b153285f1633a61b2f693c59c8576693af185/config.json)、[官方模型程式](https://huggingface.co/microsoft/Mage-VL/blob/d88b153285f1633a61b2f693c59c8576693af185/modeling_mage_vl.py)、[權重索引](https://huggingface.co/microsoft/Mage-VL/blob/d88b153285f1633a61b2f693c59c8576693af185/model.safetensors.index.json)。本機固定 revision 的程式、配置及權重檔均存在，模型類別已按職責拆至 Mage-VL 模型包；本次未匯入或載入模型，也未把靜態結果當成 GPU 驗證。

既有線上使用方式為 Transformers `AutoProcessor`／`AutoModelForCausalLM` 與完整 `generate()`。本次將同一固定 revision 的模型程式拆進 Mage-VL 包，Worker 改為直接載入本地 `MageVLForConditionalGeneration`；Processor 仍從同一快照透過 `AutoProcessor` 載入。唯一登錄別名為 `mage-vl-4b`，上游識別碼為 `microsoft/Mage-VL`。既有 `service.py` 已拆到 API、Engine、模型 Adapter／Runner 和 Media 模組。

### Mage-VL 模型元件邊界

```text
models/mage_vl_4b/
├── model_runner.py                 # 模型載入與 generation facade
└── components/
    ├── configuration_mage_vl.py    # 固定 revision 模型設定類別
    ├── MAGE_VL_UPSTREAM_LICENSE.txt # 上游模型程式的 Apache-2.0 授權
    ├── model_outputs.py            # Transformers ModelOutput 資料結構
    ├── vision_encoder.py           # Vision RoPE、patch embedding／merger、attention、encoder
    ├── vision_tower.py             # 視覺塔及 MageVL pretrained model 基底
    ├── multimodal_model.py         # 視覺特徵與 Qwen3 語言模型的組裝
    ├── conditional_generation.py  # 對外模型類別、logits 與生成介面
    ├── streammind_gate.py          # 官方可選 Gate 元件；預設服務路徑不載入
    ├── model_loader.py             # 本地模型類別、AutoProcessor 和 HF 載入選項
    ├── stepwise_generation.py      # 每個 request 的 prefill、decode、sampling、Cache 狀態
    └── component_registry.py       # 固定 revision module slots 與本地替換 factory 登記
```

上述模型類別由固定 revision 上游程式拆分而來，保留原始 module attribute 與預期 state-dict 路徑；來源 revision 寫入每個模型檔案，載入器在載入權重前確認 YAML revision 相符。上游程式檔及設定為 Apache-2.0，授權文本隨模型包保存。模型類別及設定直接從模型包載入，只有 Processor 繼續使用 `AutoProcessor` 和快照中的模型專屬程式。Qwen3 語言骨幹仍由 Transformers `AutoModel.from_config()` 建立。此做法將模型架構程式放回模型包，方便在局部檔案中演進，不改共用 Engine。權重能否與拆分後類別完全相容仍須後續在 CPU／GPU 整合環境驗收。

Registry 以已檢視的 `named_modules()` 路徑登記 vision／language 元件群，並在載入後比對實際模組樹。`inference.yml` 僅選取已由模型包 Python 程式登記的 slot／implementation，不允許 YAML 動態匯入任意程式碼。空的 `model.components.replacements` 是預設值，表示不做替換；本地上游模型定義不等於已實作最佳化。

設定錯誤或未登記的替換在下載／載入大型權重前失敗。合法替換須由 factory 建立 `torch.nn.Module`，並提供完全相容的 state dict 複製或明確權重轉換；所有候選模組準備完成後才改動模型樹，套用失敗會回復已變更路徑。此機制只提供本地擴充點，不等於最佳化已實作。

優先採用 Transformers 載入前已有的設定，例如 `model.attn_implementation`、量化、`device_map` 和 offload；它們在 `from_pretrained()` 前傳入。Registry 的模組替換發生在權重載入後，需要先建立原模型，故不能降低模型載入峰值顯存，權重轉換也可能暫時增加顯存。若未來優化需降低載入峰值，應將替換 factory 前移至本地模型建構／權重載入流程，而非宣稱 post-load replacement 能達成。

## 程序與元件

```text
API Server process
  FastAPI routes / schema / SSE serializer
  Engine Core
    TokenScheduler ── 公平輪轉 prefill/decode turn
    KVCacheManager ── 邏輯 block 預留、統計與回收
    IPC client ────── Pipe RPC
                         │
                         ▼
                   Model Worker process
                     MageVLModelAdapter
                     MageVLModelRunner
                       model + processor + past_key_values
```

此為最小兩程序拓樸：Engine Core 與 API Server 同進程，只有一個獨立 Worker 子進程。`launcher.py` 保持現有命令；`runtime.uvicorn_workers` 固定為 `1`，避免 API 複製時重複佔用 GPU。Worker 透過 multiprocessing `spawn` 啟動，`models.mage_vl_4b.app:create_model_worker` 延後匯入 Runner，避免 API Server import 時載入 Torch 或權重。Worker 啟動握手回報模型健康資訊；錯誤、逾時、程序退出及關閉均經 Pipe 處理。

### API Server

`api/` 維持現有路由、Pydantic schema、request body 限制、JSON/OpenAI response 與 SSE 序列化。上傳影片由 Engine 分塊寫入本機暫存檔，再傳路徑給 Worker，不經 Pipe 傳送整段影片 bytes。Live 影格透過本機 IPC 傳送；媒體 URL 與 Gateway API Key 邏輯不變。

### Engine Core

`engine/engine.py` 擁有 Worker lifecycle、推論操作、Scheduler／KV 管理與 Live session 關閉順序。API route 不直接呼叫模型 Adapter／Runner。一般 chat、SSE、影片與 Live 生成走同一個逐步排程迴圈；Worker RPC 每次只執行 prefill 或一個 decode step。

### Token Scheduler

`engine/scheduler.py` 維護活動序列、等待數及輪轉佇列：

- `engine.max_num_seqs` 限制同時保留生成狀態的序列數；`max_waiting_requests` 限制等待佇列。
- Prefill 是完整 prompt 的單一 turn，Scheduler 將 `prompt_tokens` 作為該輪預算。由於尚未實作 chunked prefill，prompt 必須不超過 `max_num_scheduled_tokens`。
- Decode 每個 turn 僅分配一個 token。完成當輪後，未完成 request 放回佇列尾端，讓其他活動序列先執行。
- 每輪只呼叫單一 request；這是公平逐步排程，沒有將多個序列合併成一個 batched GPU forward。
- 等待佇列滿時回傳 `429`；取消、模型錯誤及串流中斷均從排程狀態移除 request。

`engine/batch_planner.py` 是供未來批次 Model Runner 接入的純規劃 helper：它以序列進度快照、序列上限、token 預算及可選 prefill chunk 大小建立單輪計畫，decode 工作每序列分配一個 token，prefill 記錄 offset／chunk 長度，並回傳輪轉 cursor。該 helper 尚未被 `InferenceEngine` 或線上 `TokenScheduler` 呼叫；目前仍是一輪一個 request、完整 prompt prefill。

### KV Cache Manager

`engine/kv_cache.py` 依 `ceil((prompt_tokens + max_new_tokens) / block_size_tokens)` 預留邏輯 block，產生 request-to-block table，並回收 block ID 供下一個 request 使用。啟動後由 Runner 回報模型層數、KV heads、head dimension 和 dtype，Manager 據此估算 bytes/token 及在用 reservation bytes。

Worker 內的模型保留 Transformers Cache tensor；Engine 的 block table用於容量准入與生命週期追蹤，尚未映射成 attention 層可直接索引的實體 GPU page。因此目前不支援物理 block 重排、prefix/prompt cache reuse、KV eviction 或 VRAM 池化。`/health` 的 `kv_cache.estimated_bytes` 是估算預留量，不等於 CUDA 實際配置。

`KVCacheManager.reserve_batch()` 為規劃中的批次排程提供全有或全無的邏輯 block 預留；容量不足時不會留下部分 lease。這是容量帳本操作，不會把多個 request 的 Transformers Cache tensor 合併，也不會改變線上 Engine 目前逐 request 的 `reserve()` 呼叫。

### Worker／Model Runner

`engine/worker.py` 定義 `ModelWorker`、保留供單元測試使用的 `InProcessModelWorker`，並由線上 app 使用 `MultiprocessModelWorker`。Worker 子程序在啟動時載入 Adapter／Runner；父程序不持有模型權重。

Runner 的生成狀態以 request ID 保存輸入 IDs、attention mask、`past_key_values`、抽樣設定和產生 token。`prefill_generation()` 首次送入多模態輸入並得到首 token；後續 `decode_generation_step()` 使用 cache，只餵新 token，避免重跑視覺 Encoder。每個 request 完成或取消時釋放生成狀態。

目前 sampler 僅接受單序列、`num_beams=1`、`use_cache=true`，及 greedy／temperature／top-k／top-p／min-p／repetition penalty／no-repeat n-gram。自訂 generation kwargs、不同 Cache implementation、beam search 不可用；模型不支援 cache step 接口時 Worker 啟動／請求會以明確錯誤失敗。

## Mage-VL 離線能力探測

`scripts/mage_vl_4b/test.py` 是開發者手動執行的 Mage-VL 專屬探索範例。每次呼叫都必須明確提供 `--config`、`--model-path`、本機圖片或影片、`--question` 和 `--output`；沒有腳本同目錄 YAML 預設，也不依 `__file__` 推導專案根目錄。所有相對路徑均以啟動命令時的目前工作目錄解析。每個模型的探索腳本及其 YAML 依別名放在 `scripts/{alias}/`。YAML 沿用正式推論設定的 model／processor／runtime／api／generation／optimization 分層，輸出 token 預設讀取明確指定 YAML 的 `generation.max_tokens_default`；工具不把 `src/` 加入 import path，也不匯入推論服務模組，模型與 Processor 由腳本直接透過已安裝的 `transformers` 載入。腳本在載入 Hugging Face 套件前設定 `HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`，並對 Processor 和 Model 強制 `local_files_only=True`。指定的本機快照路徑無效時立即失敗，不搜尋快取或下載模型。

探測支援本機圖片及影片，按目前 YAML 的尺寸／影格上限準備輸入，進行 greedy prefill/decode，輸出模型與 attention 模組概況、tensor shape／dtype／bytes、每層 KV cache shape／bytes、GPU 記憶體及載入／生成計時。JSON 不輸出媒體 bytes 或生成文字，只保留生成字元數；錯誤報告會包含 traceback。這些計時是單機診斷資料，不是正式性能基線。腳本尚未在真實權重／GPU 執行；後續接入或更換 Runner 時可比較其實測介面和 cache 行為。

## 配置

`src/inference/models/mage_vl_4b/configs/inference.yml`：

| 路徑 | 職責 |
|---|---|
| `engine.max_num_seqs` | 活動 request 序列上限 |
| `engine.max_waiting_requests` | Scheduler 等待佇列上限 |
| `engine.max_num_scheduled_tokens` | 線上完整 prompt prefill turn 上限；純批次規劃 helper 的 token 預算由呼叫方傳入 |
| `engine.kv_cache.block_size_tokens` | 邏輯 cache block 的 token 數 |
| `engine.kv_cache.capacity_tokens` | 邏輯 cache 總容量 |
| `runtime.worker_startup_timeout_seconds` | Worker 模型載入握手逾時 |
| `runtime.worker_request_timeout_seconds` | Worker RPC 回應逾時 |

`network.yml` 仍只管理 API／MediaMTX 網路端點；模型識別碼、revision、精度、Processor、影片與生成選項仍放模型 `inference.yml`。API host/port、CLI、model registry 路徑保持不變。

## 錯誤與資源回收

- Scheduler 滿載或 KV block 容量不足：HTTP／SSE 以既有推論錯誤管道回報 `429`。
- Worker 啟動失敗：Engine 不進入 ready；`/health` 不回報 ready。
- Runner step 失敗、請求取消或 SSE 關閉：取消 Worker generation state，移除 scheduler request，回收 KV lease。
- Worker RPC 逾時或程序退出：回報執行錯誤並允許 Engine 關閉／重啟；本版不自動重載模型、不做請求重播。
- Engine 關閉：停止 Live sessions、等待 Scheduler 無活動 request、關閉 Worker 子程序。

## nano-vLLM 參考與本專案候選設計

> 本節是基於目前程式的架構評估，尚未納入已確認需求，也不表示下述批次執行或實體 KV page 已實作。

### 可借鑑的設計

nano-vLLM 的 Engine 每輪依序執行 `schedule`、Model Runner `run`、Scheduler `postprocess`，讓排程決策與模型執行形成單一明確循環；Scheduler 管理 waiting/running sequences，按 token 預算產生 prefill 或 decode 工作。其 BlockManager 追蹤 block、引用計數及可重用 prefix；Model Runner 建立實體 KV tensor、slot mapping 和 block table。參考：[LLMEngine](https://github.com/GeeeekExplorer/nano-vllm/blob/main/nanovllm/engine/llm_engine.py)、[Scheduler](https://github.com/GeeeekExplorer/nano-vllm/blob/main/nanovllm/engine/scheduler.py)、[BlockManager](https://github.com/GeeeekExplorer/nano-vllm/blob/main/nanovllm/engine/block_manager.py)、[ModelRunner](https://github.com/GeeeekExplorer/nano-vllm/blob/main/nanovllm/engine/model_runner.py)。

nano-vLLM 的公開範例以文字 prompt 呼叫 Qwen3，並以自訂 attention/cache 執行路徑支援上述實體 block 管理；因此可借鑑 Engine、Scheduler、Cache、Worker 的協作方式，但不能把它的 cache tensor、attention 實作或文字批次假設直接套到 Mage-VL。參考：[nano-vLLM README](https://github.com/GeeeekExplorer/nano-vllm)。

| 元件 | nano-vLLM 中的責任 | 本專案的適用方向 |
|---|---|---|
| API Server | 接收請求、整理輸入、輸出結果 | 保留目前 FastAPI 路由、schema、OpenAI 相容回應和 SSE 契約 |
| Engine Core | 驅動每輪排程、執行及結果回收 | 統一管理推論請求狀態，協調 Scheduler、Worker 和每請求事件輸出 |
| Scheduler | 從等待／執行序列選取本輪工作及 token 數 | 先以現有公平策略產生多筆工作計畫；是否採 prefill-first、chunked prefill 需依 SSE 延遲與混合媒體負載決定 |
| KV Cache Manager | 管理可由模型注意力實際索引的 KV blocks | 先保留容量帳本；只有 Runner／attention 能讀寫實體 page 後，才稱為實體 KV block manager |
| Worker／Model Runner | 批次準備張量、執行模型並管理 GPU cache | Worker 子程序獨佔模型、GPU tensor 和 Transformers Cache；以批次執行介面逐步取代目前單 request RPC |

### 對照目前版本

目前 request 狀態分散在 `InferenceEngine` 協程、`TokenScheduler` 的 `_RequestState`、Worker 內的 `MageVLModelRunner` 生成狀態及 Adapter 任務中。Scheduler 每輪只授權一個 request，Worker RPC 也只處理一個 request；`batch_planner.py` 目前是未接入線上循環的純規劃 helper。`KVCacheManager` 只記錄邏輯容量和 lease，實際 `past_key_values`／Transformers Cache tensor 留在 Worker，尚無供 attention 使用的實體 page table。這與 nano-vLLM 的批次執行和實體 KV block 管理有明確差距。

### 建議的最小目標形狀

```text
FastAPI routes / 現有 JSON、SSE 序列化
  └─ InferenceEngine：接收請求、維護序列狀態、輸出每請求事件
       ├─ TokenScheduler.schedule_batch() / postprocess()
       ├─ KVCacheManager：准入容量帳本（初期仍是邏輯容量）
       └─ MultiprocessModelWorker.execute_batch(work_items)
            └─ MageVLModelRunner：模型專屬輸入處理、批次 prefill/decode、GPU cache
```

以 `engine/sequence.py`（或同等單一模組）集中定義 `SequenceState`：request ID、來源類型、執行階段、prompt 進度、輸出上限及完成狀態。不要把模型 tensor 或 Transformers Cache 放入 Engine 狀態；它們由 Worker 按 request ID 保管。Live session 的生命週期仍由 `LiveSessionManager` 管理；每次送入模型的影格／片段生成才是可排程工作，不要把長連線 session 本身當作一條持續佔用 Scheduler 的 decode sequence。

Engine 每輪取得 `ScheduledWork[]`，至少描述 request ID、prefill/decode 階段、prompt offset、token budget 和完成／取消資訊；Worker 回傳對應的 per-request step result。Engine 更新序列狀態並將文字增量或完成事件放入該 request 的事件佇列，現有 route 從佇列輸出既有 JSON／SSE 格式。第一階段可令 `schedule_batch()` 回傳多筆工作但仍逐筆執行，以穩定狀態與事件流程；之後再新增 Worker 批次 RPC 和 Model Runner 真正的 batched forward。能力檢查需明確標示模型是否支援批次 prefill、decode 及 chunked prefill；不支援時保留單 request fallback。

Worker 已是獨立程序，因此 GPU tensor、cache tensor 及實體 KV page 的所有權必須留在 Worker。Engine 的 KV 管理器可做請求准入與預算，但 reservation 必須與 Worker 實際可用容量協調；不能只在 Engine 配 block ID，就宣稱已取得 GPU page。現有圖片、影片、Live frame 的 IPC 輸入契約可逐步演進，但不可為了批次排程把完整影片或 GPU tensor 搬進 Engine 狀態。

### KV Cache 與排程的遷移順序

1. 擴充 `scripts/mage_vl_4b/test.py`，在本機權重可用時記錄兩筆以上輸入的批次相容性、各輸入 cache 行為及 GPU 峰值；目前工具只探測單一輸入，不能據此宣稱 Mage-VL 可批次執行。
2. 集中化 Engine 序列狀態，讓 Scheduler 提供 `schedule_batch()` 和結果回收流程；先維持 batch size 1，使用 Worker mock 驗證併發、取消、錯誤和 SSE 事件順序。
3. 在模型能力探測及實測證明可行後，新增 Worker 批次 RPC 與 Mage-VL Model Runner 批次執行；不支援批次的路徑保留單筆執行。
4. 只有在確認 Mage-VL attention/cache 能消費 slot mapping／block table，並能在 Worker 內配置及回收實體 GPU pages 後，才實作實體 KV block pool 或 attention 替換。現有 `past_key_values` 步進能力本身不足以證明這些介面存在。
5. 以 mixed chat、影片及 Live 負載量測吞吐、SSE 首 token 延遲、後續 token 間隔、取消回收和顯存峰值，再決定 prefill/decode 優先級、chunk 大小及是否需要 prefix cache、CUDA Graph 或多 Worker。nano-vLLM 的 prefill 優先策略不應直接成為本專案預設，因為新請求的首 token 延遲、既有串流的 token 間隔及 Live 片段等待時間需要共同權衡。

Prefix cache 應延後評估。多模態快取鍵至少須涵蓋模型與 Processor revision、文字 token、影像／影格內容及預處理設定；只用文字 prompt hash 可能錯誤重用視覺特徵或 KV 狀態。這些是後續驗證項目，不是目前功能或已排定實作承諾。

## 追蹤

需求／驗收見 [spec.md](spec.md)；實作任務、證據及 GPU 驗收缺口見 [tasks.md](tasks.md)。
