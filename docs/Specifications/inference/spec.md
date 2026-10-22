# 行為規格：五元件最小 Inference Engine

> 提案：[proposal.md](proposal.md)｜技術設計：[design.md](design.md)｜任務／驗收：[tasks.md](tasks.md)
> 項目狀態：五元件已實作；執行證據與尚未執行的硬體驗收見 [tasks.md](tasks.md)。

## 名詞與現況基線

- **API Server**：FastAPI HTTP routes、schema、錯誤及 SSE/JSON response。
- **Engine Core**：獨立程序，協調 API 推論工作、Scheduler、KV Cache Manager 及 GPU Worker lifecycle；API Server 透過 `EngineClient` IPC 存取。
- **Token Scheduler**：以 prompt／decode token turn 排程活動 request。
- **KV Cache Manager**：按 token 容量建立邏輯 block table、估算 reservation、回收 block ID；底層 Cache tensor 仍由 Model Runner 管理。
- **Model Worker**：獨立子程序，載入模型並處理 Engine IPC；單一 Worker 對應一份模型副本。
- **Model Runner**：Mage-VL 模型載入、逐步 forward、`past_key_values` 及 token 選取。
- `mage-vl-4b` 是專案別名，對應 Hugging Face `microsoft/Mage-VL`；固定 revision 為 `d88b153285f1633a61b2f693c59c8576693af185`。

## 需求與驗收條件

### REQ-17：保留既有 CLI 與外部 API

- **AC-17.1** `launcher.py list`、`start --model mage-vl-4b`、設定覆寫及 host/port/GPU 旗標可解析。
- **AC-17.2** 既有健康、模型列表、chat、video、Live 及 MediaMTX auth 的路徑、方法、欄位與 response schema 不變。
- **AC-17.3** Chat SSE 保留 OpenAI chunk、錯誤事件和 `[DONE]`；Live SSE 保留 event id／keep-alive。
- **AC-17.4** 對外仍以 `mage-vl-4b` 回報模型名；上游仍登錄為 `microsoft/Mage-VL` 及固定 revision。

### REQ-18：API Server 與模型執行分離

- **AC-18.1** API route 只透過 Engine 操作推論，不直接呼叫 Adapter、Runner、Processor 或 CUDA API。
- **AC-18.2** `engine/` 與 `models/` 不匯入 FastAPI Request/Response／UploadFile 類型。
- **AC-18.3** 三組 route 位於 chat、video、live 模組；HTTP/OpenAI schema 序列化留在 `api/`。

### REQ-19：Engine Core 管理 Worker 與執行生命週期

- **AC-19.1** Engine 啟動 Worker 後才進入 ready；停止時關閉 Live sessions、等待排程工作結束並關閉 Worker。
- **AC-19.2** `engine.max_num_seqs` 控制同時活動序列，`max_waiting_requests` 控制佇列；排程器／cache 滿載時回 HTTP／SSE `429`。
- **AC-19.3** 假 Worker 測試驗證 start、ready、close、分派及失敗清理，不載入 CUDA 或權重。

### REQ-20：模型 Adapter 與 Model Runner 分離

- **AC-20.1** Mage-VL Adapter 處理提示詞、圖片／影片 Processor、生成選項及領域結果，不建立 HTTP response。
- **AC-20.2** Runner 負責模型／Processor 載入、stepwise forward、cache state、token decode 及資源回收。
- **AC-20.3** Runner 未驗證的模型內部替換點不寫成已具備能力；執行時缺少必要 cache API 時明確失敗。
- **AC-20.4** 更換相同領域契約的 Adapter／Runner 不需修改 API route 和 HTTP schema。

### REQ-21：模型專屬多模態處理與通用 Media 分離

- **AC-21.1** 上傳影片暫存、通用 metadata／frame 抽取與 Live session lifecycle 留在 `media/`。
- **AC-21.2** Mage-VL chat template、影像和影片輸入、codec 設定由模型 Adapter／Processor 負責。
- **AC-21.3** 圖片／影片不依賴 MediaMTX；Live RTSP／RTMP 仍直連 MediaMTX。

### REQ-22：五個 Engine 元件均有實作職責

- **AC-22.1** `api/` 接收及序列化請求；`engine/engine.py` 管理工作生命週期並把推論送進 Worker。
- **AC-22.2** `engine/scheduler.py` 執行 prefill/decode turn 排程；`engine/kv_cache.py` 建立 request-to-block table 並回收 block ID。
- **AC-22.3** `engine/worker.py` 提供獨立子程序 IPC Worker；模型 Adapter／Runner 在子程序載入並執行。
- **AC-22.4** 文件及 `/health` 把邏輯 block 預留與 Transformers 動態 Cache tensor 分別描述；不宣稱 PagedAttention、prefix cache 或 continuous batching。

### REQ-23：模型包擁有程式、設定、依賴與入口

- **AC-23.1** 登錄表內模型設定路徑及 app entrypoint 均指向存在的模型包檔案。
- **AC-23.2** inference/network/MediaMTX YAML 與依賴清單放在 `models/{model_name}/`。
- **AC-23.3** 現有 `launcher.py start --model mage-vl-4b` 及 CLI 設定覆寫仍可使用。

### REQ-24：分層測試且不做模型精度測試

- **AC-24.1** `tests/inference/` 以標準函式庫、Fake Worker 與合成輸入驗證 Engine、Scheduler、KV Manager、IPC 及契約。
- **AC-24.2** `tests/gateway/`、`tests/inference/`、`tests/end2end/` 保持分層。
- **AC-24.3** Inference 單元／契約測試不下載或載入權重、不啟動 GPU／模型／MediaMTX，也不以 mock 結果宣稱真模型驗收；經明確授權的獨立檔案下載與靜態閱讀不算模型驗收。

### REQ-25：新增模型沿用共用 Engine

- **AC-25.1** 模型包入口提供 Worker factory，並以既有共用 API、Engine、Media 元件組裝服務。
- **AC-25.2** `src/`、`inference/`、`models/` 根目錄仍非 Python package；相同 API 契約的新模型只需新增模型包及一筆登錄。
- **AC-25.3** 文件清楚標示目前只有 Mage-VL 一個模型；不預設第二個模型。

### REQ-26：所有 HTTP 推論 route 透過 Engine

- **AC-26.1** Chat、影片與 Live 推論均呼叫共用 Engine；API 不直接呼叫 Mage-VL Adapter 或 Runner。
- **AC-26.2** Engine 不依賴 FastAPI、UploadFile、Pydantic HTTP schema 或 OpenAI response 欄位。
- **AC-26.3** Fake Worker 測試驗證 route-to-engine 分派、錯誤映射及串流取消資源釋放。

### REQ-27：Adapter 提供模型接入邊界

- **AC-27.1** Mage-VL Adapter 將 chat、影片及 Live 輸入準備為 Runner 可執行的模型輸入，結果回傳 Engine domain object。
- **AC-27.2** Model Runner 不處理 HTTP、會話 URL 或 Gateway 認證。
- **AC-27.3** 模型包不使用冗長 `service.py`；模型實作維持在 `models/{model_name}/`。

### REQ-28：Live session 經 Engine 與共用 Scheduler

- **AC-28.1** Live create、events、stop 和 MediaMTX auth 透過 Engine session API 工作。
- **AC-28.2** Live frame／segment 生成使用共用 Token Scheduler、KV block reservation 及 Worker；關閉時先 stop sessions，再等待活動 request 回收資源。

### REQ-29：五個元件維持可替換的單向邊界

- **AC-29.1** Engine 依 `ModelWorker` contract 啟動、派發、查健康和關閉 Worker。
- **AC-29.2** Scheduler 與 KV Manager 不呼叫 API/Adapter；Worker 執行模型但不組裝 HTTP response。
- **AC-29.3** 子程序協定明確定義命令、結果／事件、錯誤、逾時、取消及 shutdown。
- **AC-29.4** 本專案僅參考 vLLM 職責與排程概念，不接入 vLLM，也不複製其 executor 或 kernel。

### REQ-30：Model Worker 隔離程序與 GPU Runtime

- **AC-30.1** 模型包的 Engine Core factory 使用 `MultiprocessModelWorker`，以 `spawn` 建立一個 GPU Worker 子程序；API import 不建立 Engine Core、Worker 或模型物件。
- **AC-30.2** Worker IPC 支援啟動 handshake、invoke、stream event、健康資訊、GPU 狀態、錯誤、逾時與 graceful close。
- **AC-30.3** `runtime.uvicorn_workers` 不是 `1` 時 launcher 拒絕啟動，避免重複載入模型副本。
- **AC-30.4** IPC 假 Worker 測試證明 Worker PID 不等於 API PID，並驗證呼叫、串流、健康和關閉。

### REQ-31：Token Scheduler 分配逐輪工作預算

- **AC-31.1** Scheduler 維護活動／等待 request，達到 `max_num_seqs` 及等待佇列上限時回 HTTP／SSE `429`。
- **AC-31.2** Scheduler 以完整 prompt token 數維持請求狀態；啟用 REQ-39 時可按 offset 分輪執行，未啟用時 prompt 超過 `max_num_scheduled_tokens` 必須明確失敗。
- **AC-31.3** Decode 每輪分配一個 token，完成後未結束 request 加入輪轉佇列尾端；可由 Fake Worker 測試觀察交錯順序。
- **AC-31.4** 錯誤、客戶端取消、Live stop 和 Engine shutdown 都移除排程狀態，Scheduler idle 後才允許關閉 Worker。

### REQ-32：KV Cache Manager 管理邏輯 token blocks

- **AC-32.1** 對每個生成工作依 prompt+最大輸出 token 數向上取整分配 block ID；容量不足時拒絕排程。
- **AC-32.2** 完成、取消及 Runner 錯誤都釋放 block table；後續 request 可重用已釋放的 block ID。
- **AC-32.3** Manager 回報容量、保留 token 數、block 數與估算 bytes；估算依 Worker 模型 cache profile，無 profile 時標記為 `0`/未知。
- **AC-32.4** Manager 不聲稱控制底層動態 Cache 的實體 GPU 配置，不支援 prefix cache／PagedAttention；此限制記錄在 README 與健康欄位說明。

### REQ-33：Model Runner 支援逐步生成與取消

- **AC-33.1** Runner 對每個 request 保存輸入、抽樣設定與 `past_key_values`；prefill 後每個 decode step 只處理新 token，並只在首次 forward 傳入視覺 tensor。
- **AC-33.2** greedy、temperature、top-k、top-p、min-p、repetition penalty 及 no-repeat n-gram 由 Runner 明確處理；EOS／max token 得到正確 finish reason。
- **AC-33.3** beam search、`use_cache=false`、自訂 GenerationMixin kwargs 和非預設 Cache implementation 以明確錯誤拒絕；不回退至不受排程的整段 `generate()`。
- **AC-33.4** 假模型／合成 cache 測試驗證 prefill、decode、EOS、max token、request 隔離及取消釋放；真模型相容性仍需固定 revision 的 GPU 驗收。

### REQ-34：Scheduler 依 token 預算建立線上批次

- **AC-34.1** 線上 `TokenScheduler.schedule_batch()` 呼叫 `engine/batch_planner.py`，批次尊重活動序列數與 scheduled-token 上限；同一階段相容的多個 decode／prefill 可共用 Worker 批次。
- **AC-34.2** 每個批次中的 request ID 唯一；規劃輸入進度或限制無效時明確拋出 `ValueError`。線上 chunked prefill 和 mixed-phase 行為另由 REQ-39 規定。
- **AC-34.3** `KVCacheManager.reserve_batch()` 以全有或全無方式建立邏輯 lease；容量不足不改變已用 block 狀態，成功 lease 可個別釋放並重用 block ID。此 helper 目前不接入線上 Engine 准入。
- **AC-34.4** Engine 將 `ScheduledBatch` 傳至 `Worker.execute_batch()`，並驗證逐 request 結果；Mage-VL 對相容輸入可用單次多序列 forward，不相容時逐 request fallback。此能力不代表 continuous batching 或實體 GPU cache page。

### REQ-35：提供不依賴推論服務 import 的 Mage-VL 能力探測範例

- **AC-35.1** Mage-VL 探索工具 `scripts/mage_vl_4b/test.py` 每次執行均要求明確提供 `--config`、`--model-path`、`--image` 或 `--video`、`--question` 與 `--output`；不得依 `__file__` 推導專案根目錄或預設 YAML。相對路徑以啟動命令時的目前工作目錄解析。欄位沿用正式推論設定分層，輸出 token 數預設讀取明確指定 YAML 的 `generation.max_tokens_default`，且不匯入 `src/` 下模組、不建立 HTTP API client；其他模型工具及 YAML 依別名放在 `scripts/{alias}/`。
- **AC-35.2** 探測在載入 Hugging Face API 前強制離線環境；只從必填 `--model-path` 載入本機 snapshot、Processor 與 Model，使用 `local_files_only=True`；路徑無效時失敗，不搜尋 HF 快取或下載模型。
- **AC-35.3** 程式生成 JSON 報告，欄位涵蓋模型類別／結構、attention 模組、處理後輸入 tensor、prefill 及逐 token cache 層 shape／bytes、GPU 記憶體和載入／推理計時；不寫入輸入媒體內容或模型生成文字。
- **AC-35.4** 腳本提供 `--help`，可用 compile／靜態檢查驗證不匯入專案推論模組；真實模型報告必須另在已準備好權重與 GPU 的環境執行，不以靜態檢查代替實機證據。

### REQ-36：在模型包內管理固定 revision 元件與本地替換

- **AC-36.1** Mage-VL 模型包以本地固定 revision 模型類別及 Transformers `AutoProcessor` 載入；專案 alias、上游 `repo_id`、revision、既有 CLI 和 HTTP API 契約維持不變。
- **AC-36.2** `components/` 分別承載模型載入、逐步 prefill/decode 狀態及該模型的元件 slot／替換 registry；共用 `engine/` 不匯入 Mage-VL 內部模組。
- **AC-36.3** Registry 依已確認的 Mage-VL `named_modules()` 路徑列出 vision 與 language 元件；載入後若固定 revision 結構不符，啟動明確失敗。
- **AC-36.4** 預設空 `model.components.replacements` 保持上游架構；未登記 slot／implementation 或空名稱須在模型權重載入前拒絕，YAML 不得指定任意 Python import。
- **AC-36.5** 替換 factory 只接受已登記本地實作，回傳 `torch.nn.Module`；預設以 strict state dict 複製，形狀或表示改變時須提供明確 transfer function。全部候選準備成功後才套用模型樹，失敗時回復已替換路徑。
- **AC-36.6** Transformers 原生載入前設定（如 `attn_implementation`、量化及 `device_map`）在 `from_pretrained()` 前套用；載入後元件替換不得被描述為可降低模型載入峰值顯存。
- **AC-36.7** 無權重測試以假 module tree 驗證 inventory、預設 no-op、設定拒絕、權重搬移與替換回復；本次不載入模型、不啟動 CUDA/GPU。

### REQ-37：將 Mage-VL 固定 revision 模型定義拆入模型包

- **AC-37.1** `models/mage_vl_4b/components/` 分別保存固定 revision 的模型設定／輸出、視覺元件、視覺塔、多模態組裝、生成頭及可選 StreamMind Gate；共用 `engine/` 不依賴這些模型類別。
- **AC-37.2** Worker 直接從模型包載入 `MageVLForConditionalGeneration`，不使用 `AutoModelForCausalLM` 動態執行快照內的模型類別；Qwen3 骨幹仍由 Transformers `AutoModel.from_config()` 組成。
- **AC-37.3** 模型程式固定 revision 與 `inference.yml` 選定 revision 不一致時，在權重載入前明確失敗；Processor 仍由 `AutoProcessor` 從相同本地快照載入。
- **AC-37.4** 拆分保留原始模型 module attribute 名稱與生成方法簽章，使固定 checkpoint 的 state-dict 路徑及現有 Adapter／Runner 呼叫契約不變；真實權重相容性須另外執行模型整合驗收。
- **AC-37.5** 此次不載入權重、不啟動 CUDA/GPU、不宣稱模型輸出等價或效能改善；文件標明靜態檢查與未執行的實機驗收。

### REQ-38：Model Runner 以單次 forward 執行相容的多 request 批次

- **AC-38.1** `execute_batch(requests)` 為每個 request 回傳獨立結果及生成狀態；相容輸入可在同一次模型 forward 共同 prefill 或 decode，Cache 與抽樣狀態不得跨 request 混用。
- **AC-38.2** 不相容的影像／影片 tensor 形狀可按明確能力策略拆批或回退單 request；不可靜默丟失視覺輸入、位置資訊或結果歸屬。
- **AC-38.3** 假模型測試驗證 batch 輸入對齊、結果映射、cache 隔離及錯誤時整批收尾；固定 revision 多模態整合需在專用模型環境另行驗收。

### REQ-39：支援 token-budgeted chunked prefill 與階段混合排程

- **AC-39.1** Scheduler 維護等待／執行佇列，依 `max_num_seqs` 及每輪 token budget 產生 chunked prefill、batched decode 或混合批次；prefill offset 和完成狀態可逐輪追蹤。
- **AC-39.2** Worker 只向模型 Cache 追加本輪 prompt chunk；輸入影像／影片視覺特徵不因文字 chunking 重複計算或錯位。
- **AC-39.3** 同一 request 每輪最多出現一次；取消、逾時、Worker 錯誤及 SSE 斷線均走共用清理，其他 request 狀態不受影響。

### REQ-40：KV Cache Manager 對應每 request 的實際 Cache 生命週期

- **AC-40.1** Manager 可識別 Worker 為 request 使用的實際 Cache 物件／不透明 handle、已處理長度及追加結果；執行狀態與 Cache 長度不一致時明確失敗。
- **AC-40.2** 成功、取消、逾時、錯誤及 Worker 關閉時均釋放對應 Cache 參照；管理器不把 Transformers Cache 稱為實體 paged block。
- **AC-40.3** 假 Cache 測試驗證長度更新、request 隔離及各終止路徑釋放；GPU 記憶體與多模態 cache 行為另行實測。

### REQ-41：以 EngineClient 分離 API Server 與 Engine Core 程序

- **AC-41.1** API Server 透過 `EngineClient` 提交請求、讀取逐 request 事件及取消；API route 不直接持有 Engine Core 或 Model Worker 物件。
- **AC-41.2** Engine Core 在獨立程序中協調 Scheduler、KV 管理和 GPU Worker；程序訊息只傳送可序列化請求資料、事件及錯誤，不傳 GPU tensor 或模型 Cache。
- **AC-41.3** IPC 支援啟動／就緒、請求錯誤、逾時、客戶端取消與優雅關閉；文字 SSE chunk、順序、finish usage 和 `[DONE]` 與既有 API 契約一致。
- **AC-41.4** 多程序 fake Worker 測試驗證 API／Engine／Worker PID 隔離、事件串流、取消與資源收尾；真模型驗收不得由 mock 結果替代。

## 五個元件對照

| 元件 | 程式位置 | 已實作行為／限制 |
|---|---|---|
| API Server | `api/` | HTTP、schema、錯誤、SSE/JSON response |
| Engine Core | `engine/engine.py`、`models/mage_vl_4b/engine_core.py` | 獨立程序內管理 Worker lifecycle、操作分派、Scheduler/KV 與批次執行 |
| Scheduler | `engine/scheduler.py`、`engine/batch_planner.py` | 線上規劃 token-budgeted chunked prefill、batched decode 及混合階段批次並傳至 Worker |
| KV Cache Manager | `engine/kv_cache.py` | 線上邏輯 block 預留、估算、釋放與 ID 重用；另有原子 batch reservation helper；皆非實體 PagedAttention |
| Worker／Model Runner | `engine/worker.py`、`models/mage_vl_4b/model_runner.py` | 第三個程序載入模型；相容同階段 request 可合併至單次 forward，混合階段在單次 Worker RPC 內分組執行，逐 request 保留 Cache 與文字事件 |

## 追蹤矩陣

| 需求 | 模組 | 任務 |
|---|---|---|
| REQ-17、REQ-23、REQ-25 | `launcher.py`、model registry、模型包 | T-14、T-17、T-18、T-26 |
| REQ-18、REQ-26 | `api/`、`engine/engine.py` | T-13、T-21、T-22、T-26 |
| REQ-19、REQ-28、REQ-29 | `engine/engine.py`、`media/live_sessions.py` | T-19、T-21、T-23、T-27 |
| REQ-20、REQ-21、REQ-27 | Mage-VL Adapter／Processor／Runner、`media/` | T-11、T-12、T-20、T-24 |
| REQ-22、REQ-30 | `engine/worker.py`、模型 app | T-23、T-27 |
| REQ-31 | `engine/scheduler.py`、Engine、Runner | T-24、T-27 |
| REQ-32 | `engine/kv_cache.py`、Runner cache profile | T-25、T-27 |
| REQ-33 | `models/mage_vl_4b/model_runner.py`、Adapter | T-24、T-27 |
| REQ-34 | `engine/batch_planner.py`、Scheduler、Worker、`tests/inference/` | T-28、T-37 |
| REQ-35 | `scripts/mage_vl_4b/test.py`、`scripts/mage_vl_4b/test.yaml`、`scripts/README.md` | T-29、T-33 |
| REQ-36 | `models/mage_vl_4b/components/`、Mage-VL YAML、`tests/inference/` | T-30、T-31、T-32、T-33 |
| REQ-37 | `models/mage_vl_4b/components/`、Mage-VL `model_loader.py`、Inference README | T-34 |
| REQ-38 | Model Runner、Mage-VL 多模態 batch、`tests/inference/` | T-38 |
| REQ-39 | Scheduler、batch planner、Model Runner、`tests/inference/` | T-39 |
| REQ-40 | KV Cache Manager、Worker cache adapter、`tests/inference/` | T-40 |
| REQ-41 | API Server、`EngineClient`、Engine Core IPC、`tests/inference/` | T-41 |
| REQ-24 | `tests/inference/`、`tests/end2end/` | T-27 |

## 變更記錄

| 日期 | 內容 |
|---|---|
| 2026-10-20 | 重新定義 vLLM 職責參考下的五元件最小 Engine，補上 Token Scheduler、KV block manager、子程序 Worker 與逐步 Runner 的可驗收行為。 |
| 2026-10-20 | 新增尚未接入線上路徑的批次規劃／原子 KV lease 驗收規格，以及 scripts 下的離線能力探測範例。 |
| 2026-10-22 | 將 REQ-34 更新為線上同階段批次規劃，新增 fused Runner batch、chunked/mixed scheduling、實際 Cache lifecycle 與 EngineClient 程序邊界的後續驗收條件。 |
| 2026-10-22 | 完成 REQ-38：Mage-VL 相容請求共用多序列 prefill/decode forward，Cache 分離回每個 request；文字 SSE 仍逐 token 發送。假模型 CPU 驗證範例已加入，未在此工作區執行。 |
| 2026-10-22 | 完成 REQ-39：線上 Scheduler 已接入 chunked prefill 與 prefill/decode 混合 Worker batch；首個 chunk 覆蓋視覺 token 區段，後續 chunk 追加至同一 request Cache。已通過無模型 Scheduler、SSE source contract、compileall 及端到端 mock 驗證；未載入固定 revision、未執行 GPU Cache 驗證。 |
| 2026-10-22 | 完成 REQ-40：Worker `RequestCacheManager` 以 request ID 持有實際 Cache object、核對序列長度並釋放參照；Engine block IDs 仍為邏輯 lease，不宣稱實體 GPU page。 |
