# 實施與驗證任務：推論服務分層與最小遷移

> 規格：[spec.md](spec.md)｜設計：[design.md](design.md)｜提案：[proposal.md](proposal.md)
> 五元件最新實作由 T-23 至 T-27 追蹤；T-19 至 T-22 是前一輪 Engine 主鏈路任務，保留作為歷史驗收紀錄。T-35 已部分完成：本機權重載入及 Gateway→Inference 圖片聊天已驗收，影片輸入仍待驗證。

前一版 `REQ-17` 至 `REQ-27` 中的通用 Protocol、component registry、capability validator 與多層空目錄設計已由本版收斂取代。為保留追蹤連續性，本版重新定義必要 REQ／AC，任務沿用 `T-09` 起的編號，不宣稱前一版已完成的工作可以直接驗收本版條件。

## 任務清單

- [x] **T-09 固定並驗證既有 CLI／API 契約** — 範圍：`tests/inference/`；交付：入口選項、路由、回應及 SSE 靜態／mock 契約檢查；覆蓋：REQ-17 / AC-17.1、AC-17.2、AC-17.3、AC-17.4；依賴：無；狀態：靜態契約測試通過。
- [x] **T-10 實作 Engine Runtime 與 Model Runner 邊界** — 範圍：`engine/runtime.py`、`models/mage_vl_4b/model_runner.py`；交付：載入／ready／close、semaphore 與 fake runner 測試；覆蓋：REQ-19 / AC-19.1、AC-19.2、AC-19.3；REQ-20 / AC-20.1、AC-20.4；依賴：T-09；狀態：Runtime fake runner 及容量閘門測試通過。
- [x] **T-11 拆分 Mage-VL 輸入處理與推論服務** — 範圍：`models/mage_vl_4b/processor.py`、`models/mage_vl_4b/service.py`；交付：抽離提示詞、圖片／影片 Processor、生成參數及服務協調；覆蓋：REQ-20 / AC-20.2、AC-20.3；REQ-21 / AC-21.2、AC-21.3；依賴：T-10；狀態：模組分層及語法檢查通過，未載入真實權重。
- [x] **T-12 拆分通用影片與 Live session** — 範圍：`media/video.py`、`media/live_sessions.py`；交付：通用影片暫存／metadata／影格讀取及可注入分析 callback 的 session manager；覆蓋：REQ-21 / AC-21.1、AC-21.3；依賴：T-11；狀態：模組分層及語法檢查通過，未啟動 MediaMTX。
- [x] **T-13 建立共用 API Server 與路由模組** — 範圍：`api/`；交付：schemas、app 組裝及聊天／影片／Live routes，保留 HTTP／SSE 行為；覆蓋：REQ-17 / AC-17.2、AC-17.3；REQ-18 / AC-18.1、AC-18.2、AC-18.3；REQ-24 / AC-24.1、AC-24.3；依賴：T-10、T-11、T-12；狀態：路由契約、依賴邊界及語法檢查通過。
- [x] **T-14 保留啟動選項與模型身份** — 範圍：`launcher.py`、`model_registry.yaml`；交付：保留啟動命令、覆寫旗標、專案別名及上游模型識別碼；覆蓋：REQ-17 / AC-17.1、AC-17.4；REQ-23 / AC-23.3；依賴：T-10、T-11、T-12、T-13；狀態：CLI 介面及模型身份靜態檢查通過。
- [x] **T-15 重寫架構文件與推論 README** — 範圍：Inference Spec 四文件、README、根目錄架構索引；交付：同步五元件映射、目標結構、目前限制與啟動方式；覆蓋：REQ-23 / AC-23.1；REQ-24 / AC-24.2、AC-24.3；依賴：T-10、T-11、T-12、T-13、T-14；狀態：Spec 追蹤校驗通過。
- [x] **T-16 執行非 GPU 驗收並更新紀錄** — 範圍：`tests/inference/`、本文件；交付：執行推論測試、Spec 結構校驗、Python compile 與 diff 檢查，逐項記錄結果；覆蓋：REQ-18 / AC-18.1、AC-18.2、AC-18.3；REQ-19 / AC-19.3；REQ-24 / AC-24.1、AC-24.2、AC-24.3；依賴：T-09、T-10、T-11、T-12、T-13、T-14、T-15；狀態：15 項單元／契約測試、Spec 校驗、compileall 及 diff check 均通過；未做 GPU／真模型驗收。
- [x] **T-17 將設定與依賴歸入模型包並移除舊目錄** — 範圍：`models/mage_vl_4b/configs/`、`models/mage_vl_4b/requirements/`、登錄表、相容測試；交付：搬移三個 YAML 和依賴清單，登錄直接載入模型包 `app.py`，移除 `src/inference/mage_vl_4b/`；覆蓋：REQ-23 / AC-23.1、AC-23.2、AC-23.3；依賴：T-14；狀態：路徑契約測試與 Gateway／Inference YAML 上限相容測試通過。
- [x] **T-18 固定新增模型的最小接入契約** — 範圍：模型 app 組裝、launcher、推論 README 與測試；交付：新模型只新增模型包及一筆登錄，不修改共用 API／Runtime；共用 import 不依賴 `src.inference` package；覆蓋：REQ-25 / AC-25.1、AC-25.2、AC-25.3；依賴：T-17；狀態：共用 app/service 邊界、import 結構及 onboarding 說明靜態測試通過；未新增第二個模型。
- [x] **T-19 建立共用 Inference Engine、Scheduler 與 Worker 契約** — 範圍：`engine/engine.py`、`engine/scheduler.py`、`engine/worker.py`、模型啟動組裝；交付：Engine 持有 Worker 生命週期與工作分派，Scheduler 套用整請求准入，現行 Worker 使用程序內實作；覆蓋：REQ-19 / AC-19.1、AC-19.2、AC-19.3；REQ-22 / AC-22.1、AC-22.2；REQ-26 / AC-26.2、AC-26.3；REQ-29 / AC-29.1、AC-29.2；依賴：T-18；狀態：fake Worker／Runner 驗證生命週期、准入、分派及串流名額釋放通過。
- [x] **T-20 將 Mage-VL 請求轉換收斂為 Model Adapter** — 範圍：`models/mage_vl_4b/adapter.py`、`model_runner.py`、`processor.py`；交付：Adapter 回傳領域結果／串流事件，Runner 保留底層載入與生成，移除 `service.py` 和全域 `state.py`；覆蓋：REQ-20 / AC-20.1、AC-20.2、AC-20.3、AC-20.4；REQ-27 / AC-27.1、AC-27.2、AC-27.3；依賴：T-19；狀態：來源契約檢查 Adapter／Runner 分層及舊 workflow 檔案已移除，未載入真模型。
- [x] **T-21 讓一般及 Live routes 經由 Engine 執行** — 範圍：`api/routes/`、`api/app.py`、`api/serializers.py`、`engine/engine.py`、`media/live_sessions.py`；交付：routes 不直接依賴模型 Adapter 或 session manager；HTTP/OpenAI response 在 API 序列化；Live callback 經 Engine 共用 Scheduler；覆蓋：REQ-17 / AC-17.2、AC-17.3；REQ-18 / AC-18.1、AC-18.2、AC-18.3；REQ-26 / AC-26.1；REQ-28 / AC-28.1、AC-28.2；依賴：T-19、T-20；狀態：路由分層、HTTP response／SSE serializer 契約及 Live 共用准入檢查通過。
- [x] **T-22 驗收並記錄 Engine 主鏈路及擴展邊界** — 範圍：`tests/inference/`、Inference Spec、README；交付：fake Worker／Adapter／Runner 測試、response serializer 和 HTTP／SSE／Live route 靜態契約測試、Spec 追蹤更新；覆蓋：REQ-18 / AC-18.2；REQ-20 / AC-20.1、AC-20.2、AC-20.3、AC-20.4；REQ-22 / AC-22.3、AC-22.4；REQ-24 / AC-24.1、AC-24.3；REQ-25 / AC-25.1、AC-25.2、AC-25.3；REQ-26 / AC-26.3；REQ-29 / AC-29.3、AC-29.4；依賴：T-19、T-20、T-21；狀態：21 項 Inference 測試、Spec 校驗、compileall、設定相容測試及 diff check 通過；未啟動 GPU／真模型。
- [x] **T-23 實作子程序 Model Worker 與 IPC** — `MultiprocessModelWorker` 使用 `spawn`、Pipe handshake/RPC/error/timeout/close；Mage-VL app 延後載入 Runner；launcher 限定單一 Uvicorn process。覆蓋：REQ-30 / AC-30.1、AC-30.2、AC-30.3、AC-30.4；REQ-22 / AC-22.3；依賴：T-22；狀態：純 Python child Worker IPC 測試通過，未啟動模型。
- [x] **T-24 實作逐步 Mage-VL Runner 與 Token Scheduler** — Engine 每輪執行完整 prompt prefill 或一個 decode token，活動請求輪轉；Chat、影片、Live 共用生成生命週期。覆蓋：REQ-31 / AC-31.1、AC-31.2、AC-31.3、AC-31.4；REQ-33 / AC-33.1、AC-33.2、AC-33.3、AC-33.4；REQ-28 / AC-28.2；依賴：T-23；狀態：fake Worker 測試驗證 turn 順序、取消及結果；未載入固定 revision。
- [x] **T-25 實作 KV Cache Manager 邏輯 block reservation** — 按 prompt+輸出預算分配 block ID，執行完成／取消／錯誤時回收並允許重用；回報 cache reservation 與估算 bytes。覆蓋：REQ-32 / AC-32.1、AC-32.2、AC-32.3、AC-32.4；REQ-22 / AC-22.2、AC-22.4；依賴：T-24；狀態：容量、拒絕、回收及 ID 重用單元測試通過。
- [x] **T-26 同步 Spec、設定與 Inference README** — 更新五元件職責、程序拓樸、Token turn、cache 限制、啟動及不支援選項；明確區分邏輯 cache 與實體分頁。覆蓋：REQ-22 / AC-22.1、AC-22.2、AC-22.3、AC-22.4；REQ-29 / AC-29.1、AC-29.2、AC-29.3、AC-29.4；REQ-30 / AC-30.3；依賴：T-23、T-24、T-25；狀態：文件已同步，並通過 T-27 Spec 結構與追蹤校驗。
- [x] **T-27 執行無權重驗收並記錄證據** — 執行 Inference 單元／契約測試、Spec validator、compileall、設定相容性與 diff check；不啟動 API／GPU／MediaMTX。覆蓋：REQ-22 / AC-22.1、AC-22.2、AC-22.3、AC-22.4；REQ-24 / AC-24.1、AC-24.3；REQ-30 / AC-30.4；REQ-31 / AC-31.1、AC-31.2、AC-31.3、AC-31.4；REQ-32 / AC-32.1、AC-32.2、AC-32.3、AC-32.4；REQ-33 / AC-33.1、AC-33.2、AC-33.3、AC-33.4；依賴：T-23、T-24、T-25、T-26；狀態：24 項 Inference 測試、7 項端到端 mock／契約測試（含 1 項 YAML 相容測試）、17 項需求／58 項驗收條件／19 項任務 Spec 校驗、compileall 和 diff check 均通過；未執行真模型驗收。
- [x] **T-28 建立可單測的批次排程計畫與原子 KV 預留** — 範圍：`engine/batch_planner.py`、`engine/kv_cache.py`、既有 `tests/inference/`；交付：prefill/decode token budget、chunk offset、cursor 輪轉和全有或全無 batch lease；覆蓋：REQ-34 / AC-34.1、AC-34.2、AC-34.3、AC-34.4；依賴：T-24、T-25；狀態：由純 Python 單元測試驗證，不接入線上 Engine，不執行模型。
- [x] **T-29 新增獨立 Mage-VL 離線能力探測腳本** — 範圍：`scripts/mage_vl_4b/test.py`、`scripts/mage_vl_4b/test.yaml`、`scripts/README.md`；交付：要求明確傳入 YAML、模型快照、媒體、問題和報告路徑；讀取模型 YAML、強制本機載入，支援圖片／影片並輸出模型結構、張量、KV cache 與 GPU 診斷 JSON；覆蓋：REQ-35 / AC-35.1、AC-35.2、AC-35.3、AC-35.4；依賴：T-20、T-26；狀態：實作與文件已更新；本次未重新執行 CLI 契約測試，未載入真實權重或執行 GPU 探測。
- [x] **T-30 靜態檢視 Mage-VL 固定 revision** — 範圍：官方 revision 配置與模型程式（存於 Git 忽略的 `checkpoints/`）；交付：核對 Auto classes、視覺／語言結構及可觀察元件路徑；覆蓋：REQ-36 / AC-36.1、AC-36.3；依賴：T-29；狀態：官方設定、Python 原始碼及權重快照已在本機；上游程式已靜態檢視，未載入或執行模型。
- [x] **T-31 拆分 Mage-VL 載入與逐步生成元件** — 範圍：`models/mage_vl_4b/model_runner.py`、`components/model_loader.py`、`components/stepwise_generation.py`；交付：Runner 保留 orchestration facade，AutoModel／Processor 載入與每 request cache state 各自獨立；覆蓋：REQ-36 / AC-36.1、AC-36.2；依賴：T-30；狀態：來源契約檢查確認 Runner 仍使用 Transformers 原生生成及既有逐步 API。
- [x] **T-32 建立模型專屬元件 inventory 與替換 registry** — 範圍：`components/component_registry.py`、Mage-VL `inference.yml`、`model_runner.py`、registry mock 測試；交付：固定 module slots、YAML implementation allowlist、權重複製／明確 transfer hook、原子替換及載入前設定驗證；覆蓋：REQ-36 / AC-36.3、AC-36.4、AC-36.5、AC-36.6、AC-36.7；依賴：T-30、T-31；狀態：空設定沿用官方模組；目前無本地最佳化實作。
- [x] **T-33 同步元件架構文件並執行無 GPU 驗收** — 範圍：Inference Spec、`src/inference/README.md`、`tests/inference/`；交付：記錄固定 revision 調研、元件邊界、顯存載入時序限制與驗收結果；覆蓋：REQ-36 / AC-36.1、AC-36.2、AC-36.3、AC-36.4、AC-36.5、AC-36.6、AC-36.7；依賴：T-30、T-31、T-32；狀態（當時）：40 項 Inference 測試、7 項 end-to-end 契約測試、Spec 校驗、compileall、test.py `--help`、`test.yaml` YAML 解析與 `git diff --check` 通過；未載入模型或執行 GPU 驗收。
- [x] **T-34 將固定 revision 模型定義拆入 Mage-VL 模型包** — 範圍：`models/mage_vl_4b/components/`、`tests/inference/test_source_contract.py`、Inference Spec／README；交付：本地設定、輸出、視覺元件／視覺塔、多模態模型、生成頭、Gate 及上游授權檔案；Worker 改載入模型包中的生成類別，並在載入權重前核對 revision；覆蓋：REQ-37 / AC-37.1、AC-37.2、AC-37.3、AC-37.4、AC-37.5；依賴：T-30、T-31；狀態：程式已拆分，既有來源契約已依本地載入方式更新；語法編譯通過；未載入權重，未驗證 state-dict 整合、服務啟動或 GPU 行為。
- [ ] **T-35 驗收本地 Mage-VL 模型類別與固定權重整合** — 範圍：`models/mage_vl_4b/components/`、Worker 啟動及既有 API；交付：在具備合適記憶體／GPU 的模型環境載入固定 revision，驗證 state-dict 權重、`prepare_inputs_for_generation`、圖片／影片輸入及既有 response contract；覆蓋：REQ-37 / AC-37.4、AC-37.5；依賴：T-34；狀態：部分完成（2026-10-22，本機 RTX 4060 Laptop GPU 以 NF4 載入固定權重，圖片 chat 經 Gateway→Inference 返回 200；影片輸入尚未驗證）；不做精度基準。
- [x] **T-36 建立 Runner 批次呼叫入口並保留文字串流** — 範圍：`engine/engine.py`、`engine/worker.py`、`models/mage_vl_4b/adapter.py`、`model_runner.py`；交付：Engine 每輪透過 `execute_batch(requests)` 到 Runner，結果依 request ID 回傳；逐 token `text` 事件仍序列化為 OpenAI SSE chunk；覆蓋：REQ-31 / AC-31.3、AC-31.4；REQ-33 / AC-33.1；依賴：T-23、T-24、T-34；狀態：初始 batch size 1 介面及文字事件／錯誤清理測試通過；後續線上批次接入見 T-37。
- [x] **T-37 將 batch planner 接入線上 Engine／Worker** — 範圍：`engine/engine.py`、`engine/scheduler.py`、`engine/worker.py`、Mage-VL Runner、Inference YAML；交付：Engine Core 背景迴圈執行同階段多 request prefill/decode 批次，受序列數及 token budget 約束；Worker 回傳逐 request 結果；覆蓋：REQ-31 / AC-31.1、AC-31.3、AC-31.4；REQ-34 / AC-34.1、AC-34.2、AC-34.4；依賴：T-36；狀態：Scheduler／Engine fake 測試驗證多請求 prefill/decode、輪轉、SSE 增量文字及收尾；Mage-VL 單次多序列 forward 由 T-38 實作。
- [x] **T-38 實作 Mage-VL 相容輸入的單次多 request forward** — 範圍：`models/mage_vl_4b/model_runner.py`、`components/stepwise_generation.py`、來源契約及 `tests/inference/stepwise_batch_example.py`；交付：相容多模態輸入共用一次 prefill/decode forward，逐 request 分離 logits、Cache 與抽樣狀態；不相容時逐 request fallback；覆蓋：REQ-38 / AC-38.1、AC-38.2、AC-38.3；依賴：T-37；狀態：程式、固定 revision 模型原始碼及 Transformers Cache API 已靜態檢視；語法與來源契約驗證通過；假模型 CPU 範例已提供，因目前 Python 環境未安裝 PyTorch而未執行，亦未執行真實多模態/GPU forward。
- [ ] **T-39 實作 chunked prefill 與 prefill/decode 混合排程** — 範圍：`engine/batch_planner.py`、`engine/scheduler.py`、Mage-VL Runner、`tests/inference/`；交付：offset、token budget、request 狀態及視覺輸入只處理一次；覆蓋：REQ-39 / AC-39.1、AC-39.2、AC-39.3；依賴：T-38；狀態：未開始，需先驗證多模態 Cache 追加行為。
- [x] **T-40 將 KV Cache Manager 連結至 Worker 實際 Cache 生命週期** — 範圍：`models/mage_vl_4b/components/cache_manager.py`、Stepwise Runner、Adapter health、`tests/inference/test_cache_manager.py`；交付：Worker 端每 request 持有實際 Cache object、追蹤實際長度與 Engine lease block IDs，長度不符時拒絕替換，finish／cancel／Worker close 釋放參照；Engine `KVCacheManager` 仍只管理邏輯容量；覆蓋：REQ-40 / AC-40.1、AC-40.2、AC-40.3；依賴：T-38；狀態：假 Cache 單元測試驗證長度、request 隔離與釋放，source contract 驗證 Runner 收尾掛接；未測 CUDA 配置、PagedAttention 或實際多模態 cache。
- [x] **T-41 透過 EngineClient 分離 API Server 與 Engine Core 程序** — 範圍：`api/`、`engine/client.py`、模型 `engine_core.py`、Inference YAML、`tests/inference/`；交付：API Server、Engine Core、GPU Worker 三程序；支援 request ID 多工 IPC、逐 token 事件、取消、錯誤、影片同機路徑轉交及優雅關閉；Live session／媒體擷取留在 API 程序，模型分析透過 EngineClient；覆蓋：REQ-18 / AC-18.1、AC-18.2；REQ-30 / AC-30.1、AC-30.2；REQ-41 / AC-41.1、AC-41.2、AC-41.3、AC-41.4；依賴：T-37、T-40；狀態：完成。三程序 fake Worker 測試驗證 PID 隔離、逐 token 事件順序、錯誤回傳、關閉 stream 後資源回收及優雅退出；Engine 測試另驗證 context 建立前取消會在排程前停止。來源契約確認既有 OpenAI SSE chunk、finish usage 和 `[DONE]` 路由未改。未啟動 FastAPI runtime、真模型、GPU 或 MediaMTX。

## 驗收紀錄

| 檢查 | 結果 | 證據／限制 |
|---|---|---|
| 推論單元及契約測試 | 通過 | `python -m unittest discover -s tests/inference -q`；24 項通過 |
| 端到端 mock／契約測試 | 通過 | `python -m unittest discover -s tests/end2end -q`；7 項通過（含下列設定檔相容測試） |
| 設定檔相容測試 | 通過 | `tests/end2end/test_config_compatibility.py`；確認 Gateway 本文上限不高於模型包推論 YAML |
| Spec 結構與追蹤校驗 | 通過 | `validate_spec_group.py docs/Specifications/inference`；17 項需求、58 項驗收條件、19 項任務 |
| Python 語法／差異檢查 | 通過 | 對 `src/inference/`、`tests/inference/` 執行 `compileall`；`git diff --check` 無差異錯誤 |
| FastAPI 啟動／HTTP runtime smoke | 部分通過 | `GET /health`、Gateway `GET /readyz`、授權 `GET /v1/models` 均返回 200；固定 revision Worker 已成功載入 |
| Mage-VL／GPU／MediaMTX | 部分驗收 | RTX 4060 Laptop GPU、NF4；圖片聊天完成 Gateway→Inference→Worker 往返並返回 200。未測影片輸入或 MediaMTX |
| Go Gateway／端到端效能 | 功能往返通過；基線未建立 | 已完成一次圖片聊天請求（Gateway 上游約 1.93 秒）；未執行性能基線或多請求測試 |

### Engine Core 程序隔離驗收（T-41）

| 檢查 | 結果 | 證據／限制 |
|---|---|---|
| EngineClient／Engine Core／GPU Worker IPC | 通過 | `python -m unittest test_engine_client -v`；3 項純 Python 多程序測試驗證三個不同 PID、逐 token 事件順序、遠端錯誤後可繼續服務、SSE iterator 關閉後 scheduler／KV lease 歸零及優雅退出。 |
| API／SSE 契約 | 通過 | `test_source_contract` 確認 API 只組裝 `EngineClient`、Engine Core factory 建立 `InferenceEngine`／`MultiprocessModelWorker`，chat route 保留逐 token text event 與 `[DONE]`；未啟動 FastAPI runtime。 |
| 推論與端到端契約 | 部分通過 | 31 項目標 Inference 測試及 7 項 `tests/end2end/` 測試通過。完整 Inference discovery 共 52 項，51 項通過；工作樹既有未追蹤 `tests/inference/test_capability_probe.py` 的字面 source assertion 失敗，本次未修改該檔。 |
| Spec／語法／差異 | 通過 | Spec validator：25 項需求、91 項驗收條件、33 項任務；`compileall` 與針對本次檔案的 `git diff --check` 通過；未載入權重、未執行 GPU、真實 API runtime 或 MediaMTX。 |

### 本次補充驗收（T-28、T-29）

| 檢查 | 結果 | 證據／限制 |
|---|---|---|
| Inference 測試套件 | 通過 | `python -m unittest discover -s tests/inference -q`；34 項通過，含批次規劃、KV batch lease 及探測工具契約測試 |
| 探測工具入口與隔離 | 通過 | `python scripts/mage_vl_4b/test.py --help` 成功；AST 契約測試確認不匯入 `src/` 模組，且檢查離線旗標與 `local_files_only=True` |
| Spec 結構與追蹤校驗 | 通過 | 19 項需求、66 項驗收條件、21 項任務；REQ／AC／T 唯一且追蹤完整 |
| 語法與差異檢查 | 通過 | `compileall` 涵蓋 `src/inference/`、`tests/inference/` 及探測腳本；`git diff --check` 無 whitespace error（Git 顯示既有 LF/CRLF 提示） |
| Mage-VL 能力報告 | 未執行 | 未載入權重、不啟動 CUDA；需在已備妥本機快取及 GPU 的推理環境手動執行工具 |

### 本次元件拆分與模型腳本驗收（T-30 至 T-33）

| 檢查 | 結果 | 證據／限制 |
|---|---|---|
| 固定 revision 官方程式靜態檢視 | 通過 | 已檢視本機快照的 `config.json`、`modeling_mage_vl.py` 及權重索引；未匯入或執行遠端模型程式 |
| Inference 無權重測試 | 通過 | `python -m unittest discover -s tests/inference -q`；40 項通過，含 module inventory、allowlist、權重相容替換、失敗回復及本地 YAML 契約 |
| End-to-end 契約測試 | 通過 | `python -m unittest discover -s tests/end2end -q`；7 項通過 |
| Spec／語法／腳本入口／差異 | 通過 | 20 項需求、73 項驗收條件、25 項任務；compileall、模型探索腳本 `--help` 及 `git diff --check` 均通過 |
| Mage-VL 探索設定 | 通過 | `test.yaml` 使用 PyYAML 6.0.3 解析；模型 repo／revision 與 `local_files_only: true` 符合本次固定配置 |
| 權重及 GPU 整合（T-33 當時） | 未完成 | 當時下載中斷且只有部分快照；未載入模型、啟動 CUDA/GPU 或執行 MediaMTX。後續快照狀態與模型包拆分見 T-34。 |

### 本次本地模型定義拆分（T-34）

| 檢查 | 結果 | 證據／限制 |
|---|---|---|
| 固定 revision 檔案 | 存在 | 本地兩個模型 shard、索引及官方模型程式均存在；未將權重加入 Git。 |
| Python 語法 | 通過 | 使用 workspace Python 對 9 個模型包檔案及修改後的來源契約檔執行 `py_compile`；未匯入模型類別或權重。 |
| Spec／差異格式 | 通過 | Spec validator：21 項需求、78 項驗收條件、27 項任務；`git diff --check` 無差異錯誤，僅有既有 LF/CRLF 提示。 |
| 真實權重載入／API／GPU | 部分驗證 | 固定 revision 以 NF4 載入成功；修正逐步生成重複傳入 `use_cache` 後，Gateway→Inference 圖片聊天返回 200。影片輸入及模型輸出品質未驗證。 |

## 變更記錄

| 日期 | 內容 |
|---|---|
| 2026-10-20 | 重寫任務分解，以五個 vLLM 參考元件收斂為 API、Runtime、Model Runner、Media 的可驗收遷移。 |
| 2026-10-20 | 依 vLLM Engine/Core 官方架構再校正目標：Engine 必須承接推論請求主流程；新增 T-19 至 T-22，待實作。 |
| 2026-10-20 | 落實 Engine、整請求 Scheduler、Worker 契約及 Mage-VL Adapter；同步路由、SSE／JSON 序列化、Live 准入、文件與 fake／契約驗證。 |
| 2026-10-20 | 完成五元件最小實作：API Server、Engine Core、輪轉 Token Scheduler、邏輯 KV block 管理及子程序 Model Worker／逐步 Runner；24 項 Inference、7 項端到端 mock／契約測試與 Spec 追蹤校驗通過。未做權重、GPU、API runtime 或 MediaMTX 驗收。 |
| 2026-10-20 | 加入未接入線上路徑的批次規劃／原子 KV 預留 helper，並提供不匯入 `src/` 的 Mage-VL 離線能力探測腳本；真模型探測待具備本機權重及 GPU 時執行。 |
| 2026-10-22 | 將固定 revision Mage-VL 設定、視覺、多模態組裝及生成類別拆入模型包；本次只做語法與規格檢查，真實權重載入待後續驗收。 |
| 2026-10-22 | 在 RTX 4060 Laptop GPU 上以 YAML 指定的 NF4 與本機快照載入固定 revision；修正逐步生成重複傳入 `use_cache`，完成 Gateway API Key、模型別名、Inference Worker 與圖片聊天的端到端往返。影片輸入及端到端性能基線仍待驗收。 |
