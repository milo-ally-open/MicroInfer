# 實施與驗收任務：Inference Engine 五層架構

> 提案：[proposal.md](proposal.md)｜規格：[spec.md](spec.md)｜設計：[design.md](design.md)
> 任務狀態涵蓋五層引擎與模型接入工作流；改造前已驗收的 Qwen3 基線保留於本地回復提交 `0cb3c56` 及 `checkpoints/val.md`、`checkpoints/val.xlsx`。

## 任務清單

- [x] **T-01 建立五層唯一規格與責任追蹤** — 範圍：本規格組四文件；交付：L1–L5 所有權、需求／驗收／任務 ID 映射、明確區分事實／決策／限制；覆蓋：REQ-01、REQ-02、REQ-03、REQ-04、REQ-05、REQ-06、REQ-07、REQ-08；依賴：無；狀態：已重寫並通過結構與追蹤校驗。
- [x] **T-02 將 Qwen 輸入處理與增量 Detokenize 放到 API 邊界** — 範圍：API input/output processor、聊天 route、Qwen `input_processor.py`、Adapter、Engine IPC contract；交付：`PreparedChatInput` IPC、token ID event、非串流與增量文字 SSE；覆蓋：REQ-01 / AC-01.1、AC-01.2、AC-01.3、AC-01.4、AC-01.5、AC-01.6、AC-01.7；依賴：T-01；狀態：CPU mock／source 契約測試及固定 Qwen3 snapshot 的真實 JSON／SSE Gateway smoke 通過。
- [x] **T-03 抽出 Engine Output Processor 並驗證生命週期** — 範圍：`engine/output_processor.py`、`engine/engine.py`、Engine／SSE 測試；交付：Worker token／text 轉 Engine event、終止路徑清理和 stream isolation；覆蓋：REQ-02 / AC-02.1、AC-02.2、AC-02.3、AC-02.4；REQ-03 / AC-03.4；依賴：T-01、T-02；狀態：Engine lifecycle、取消、Worker failure 與事件契約測試通過。
- [x] **T-04 拆出 Executor 與 Platform 契約** — 範圍：`engine/executor.py`、`worker.py`、`platform.py`、Mage／Qwen `engine_core.py`、Qwen YAML 和 Worker process tests；交付：Engine 依賴 Executor、LocalProcessExecutor 管理單 Worker spawn/Pipe、Qwen 使用 TorchCudaPlatform；覆蓋：REQ-04 / AC-04.1、AC-04.2、AC-04.3、AC-04.4、AC-04.5；REQ-07 / AC-07.2、AC-07.3；依賴：T-01；狀態：程序／RPC／startup cleanup 與未登記 Platform 測試通過；未執行 CUDA 裝置驗收。
- [x] **T-05 抽出 Qwen InputBatch 並保持批次語義** — 範圍：Qwen `model/input_batch.py`、`model_runner.py`、native runtime tests；交付：批次 tensor、長度、offset、block table 對齊及不相容輸入拒絕；覆蓋：REQ-05 / AC-05.1、AC-05.2、AC-05.3、AC-05.4；REQ-03 / AC-03.1、AC-03.2、AC-03.3、AC-03.4、AC-03.5；依賴：T-01；狀態：InputBatch、合成 Safetensors 權重載入、變長 prefill/decode 數值對照及實體 page 所有權測試均在 CPU 環境通過；固定 snapshot／CUDA 驗收仍屬 T-08。
- [x] **T-06 驗證模型包生態相容與登錄邊界** — 範圍：Qwen `utils/`、模型 package、`model_registry.yaml`、Mage/Qwen source contract；交付：固定 revision、權重 key／shape 和 alias／backend 職責不變；覆蓋：REQ-06 / AC-06.1、AC-06.2、AC-06.3、AC-06.4；REQ-07 / AC-07.1、AC-07.4；依賴：T-01、T-04；狀態：固定 revision、模型 alias、API／Engine 分離及 Mage/Qwen 路徑契約測試通過。
- [x] **T-07 完成分層契約與 failure-path 自動測試** — 範圍：`tests/inference/`；確認 `tests/gateway/`、`tests/end2end/` 維持分開；交付：無權重 unit／source／IPC 測試和 API SSE／取消契約證據；覆蓋：REQ-08 / AC-08.1、AC-08.2；依賴：T-02、T-03、T-04、T-06；狀態：Inference 套件 73 項測試全部通過；合成 Runner／InputBatch 測試使用 CPU PyTorch，不載入實際模型權重。
- [ ] **T-08 使用固定 Qwen3-0.6B snapshot 驗收串流與性能** — 範圍：Qwen snapshot 與 `checkpoints/val.*` 工作流；交付：非串流／SSE／併發／取消回收和性能基線；覆蓋：REQ-08 / AC-08.3、AC-08.4、AC-08.5；依賴：T-02 至 T-07；狀態：部分驗收：2026-10-22 以 WSL `magevl-transformers`、Qwen3-0.6B CUDA API 和 Go Gateway 完成非串流、逐 token SSE、3 個串行性能樣本、2 個並發請求及首個文字增量後的真實斷流回收；健康資訊確認 Scheduler、邏輯 lease 與 Worker 實體 pages 均歸零。報告位於 `tests/end2end/results/`；`checkpoints/val.md` 和 `checkpoints/val.xlsx` 保留未改。
- [x] **T-09 同步開發文件並通過 Spec 結構檢查** — 範圍：根目錄 `README.md`、`src/inference/README.md`、四份 Inference Spec；交付：實際目錄／指令／限制一致，Spec validator 及人工追蹤檢查；覆蓋：REQ-01、REQ-02、REQ-03、REQ-04、REQ-05、REQ-06、REQ-07、REQ-08；依賴：T-01、T-02、T-03、T-04；狀態：README 已同步，Spec validator 與任務映射校驗通過。
- [x] **T-10 共用模型裝配並簡化新模型接入** — 範圍：Launcher、API service factory、Engine／Worker bootstrap、Registry、模型入口清理及操作文件；交付：模型 Runner／Adapter 登錄、可選 API 元件、共用程序裝配，移除模型專屬 `app.py`／`engine_core.py` 包裝檔；覆蓋：REQ-09 / AC-09.1、AC-09.2、AC-09.3、AC-09.4、AC-09.5；依賴：T-01、T-04、T-06；狀態：程式與文件完成；未啟動 API 或模型，服務行為仍待實機驗收。
- [x] **T-11 登記模型 profile／API 能力並在服務邊界執行** — 範圍：`model_registry.yaml`、共用 API app／Chat route、Gateway model routes；交付：Mage-VL／Qwen capability、輸入模態與路由選擇；覆蓋：REQ-10 / AC-10.1、AC-10.3、AC-10.6；依賴：T-10；狀態：兩個 registry 能力已登記；Inference／Gateway／跨設定測試通過；真實 Gateway smoke 覆蓋 Qwen 文字 JSON／SSE、Mage 圖片 JSON／SSE／視訊上傳及 Live／MediaMTX 合成 RTSP。
- [x] **T-12 建立無權重預檢與重用模型契約測試** — 範圍：`launcher.py validate`、`tests/inference/model_contract.py`、接入指南及跨設定契約；交付：不匯入模型／不啟動 GPU 的登錄預檢和 Runner／Adapter／InputProcessor AST 合同；覆蓋：REQ-10 / AC-10.2、AC-10.4、AC-10.5、AC-10.6；依賴：T-11；狀態：Launcher 預檢、Inference 74 項測試、End2End 8 項設定與客戶端測試及兩組 Spec validator 均通過。
- [x] **T-13 提供標準文字模型預設元件** — 範圍：`api/default_chat_processor.py`、`engine/standard_text.py`、`model_bootstrap.py`、`service_app.py`；交付：可省略 Runner／Adapter／InputProcessor 工廠的標準路徑、逐 request Cache 與 token SSE；覆蓋：REQ-11 / AC-11.1、AC-11.2、AC-11.3、AC-11.4；REQ-10 / AC-10.2；依賴：T-12；狀態：CPU 合成 Cache／batch／finish／cancel、輸入與預檢測試通過；實際固定 snapshot smoke 另由 T-14 驗收。
- [x] **T-14 使用固定 Qwen snapshot 驗證標準預設 Runner** — 範圍：本機 `Qwen3-0.6B` snapshot；交付：以標準預設 InputProcessor／Runner 實際執行短生成，核對 prefill、Cache 延續、逐 token 輸出與取消清理；不切換現有 Qwen Registry profile；覆蓋：REQ-11 / AC-11.5；依賴：T-13；狀態：WSL `magevl-transformers`、PyTorch 2.14.1+cu126、Transformers 5.19.0、RTX 4060 Laptop GPU；`test_standard_text_snapshot.py` 2 項通過。另以獨立臨時 Registry 移除 Qwen 專屬工廠並登記 `standard` profile，啟動完整 API Server，驗證 health、models、非串流／SSE、別名／usage／`[DONE]` 和輸出一致；詳見 `tests/end2end/results/qwen3-0.6b-standard-profile-api-smoke-2026-10-22.json`。此驗證不改現有 Qwen Registry 項目。
- [x] **T-15 完成模型接入指南與 Gateway capability matrix** — 範圍：Mage custom 模型包文件、`tests/gateway/`、Inference 接入指南及 Gateway Spec；交付：列明 Mage／Qwen 各自宣告能力、正向路由與拒絕案例；覆蓋：REQ-09 / AC-09.5；REQ-10 / AC-10.6；依賴：T-12、T-13；狀態：指南、Gateway 雙模型 mock cases 和跨設定契約完成；真實 Gateway smoke 覆蓋 Qwen 文字 JSON／SSE、Mage 圖片 JSON／SSE／視訊上傳及 Mage Live／MediaMTX 合成 RTSP。

## 本次驗收紀錄

| 檢查 | 狀態 | 證據／限制 |
|---|---|---|
| 改造前本地回復提交 | 完成 | `0cb3c56 feat: add native Qwen3-0.6B inference engine`；本地提交，未推送。 |
| 五層程式改造 | 契約與標準 Runner snapshot smoke 通過 | L1 Tokenize／Detokenize、L2 Engine output/event、L3 Executor／Platform 與 L4 InputBatch、Qwen 合成 Runner 有 CPU 測試；標準預設 Runner 已按 T-14 通過本機 Qwen snapshot CUDA smoke。原生 Qwen profile 的 API／SSE／併發／性能驗收仍待 T-08。 |
| Inference Spec 結構與追蹤 | 通過 | `validate_spec_group.py docs/Specifications/inference`：11 REQ、54 AC、15 T；需求／驗收／任務追蹤通過。 |
| Python 單元／契約測試 | 通過 | `python -m unittest discover -s tests/inference -v` 共 85 項通過、2 項選擇性跳過；使用 WSL `magevl-transformers`，合成測試不初始化 GPU。 |
| Python 語法編譯 | 通過 | 使用可用 Python 執行 `compileall`，涵蓋 `src/inference/`；本次重新檢查亦涵蓋 `tests/inference/`。 |
| Spec 結構與追蹤校驗 | 通過 | `project-spec-workflow/scripts/validate_spec_group.py docs/Specifications/inference`：11 REQ、54 AC、15 T；Inference 與 Gateway 規格組均通過結構校驗。 |
| 無權重接入預檢與跨層能力契約 | 通過 | `launcher.py validate` 確認 Mage `custom`（image/text-delta/video/live）與 Qwen `custom`（text/token-id/stream）；標準預設元件 AST 預檢通過；End2End 10 項測試通過。 |
| 標準預設元件 CPU 合成契約 | 通過 | `test_standard_text_runtime.py` 驗證 chat template、文字能力拒絕、分塊 prefill、獨立 Cache、finish/cancel、Adapter token IDs 及無 import 預檢；尚不代表真實模型相容。 |
| 標準預設 Runner 固定 snapshot smoke | 通過 | WSL CUDA 真實模型 smoke 驗證 chat template、prefill、Cache 延續、逐 token 輸出、增量／最終文本一致和取消釋放；單請求、最多 2 個新 token，沒有啟動 API。 |
| 標準 profile 完整 API 實機 smoke | 通過 | 臨時 Registry 將 Qwen 設為 `standard` 並移除模型工廠；完整 API Server 真實載入 snapshot，`/health`、`/v1/models`、非串流／SSE、文本一致、finish usage 及 `[DONE]` 通過。直接 API 測試；不代表 Gateway 或 Mage 多模態整合。 |
| 真實模型經 Gateway 端到端 | 功能 smoke 通過；容量基線未建立 | Qwen CUDA API 已驗證 `/v1/models`、非串流／SSE 聊天、公開 alias／`[DONE]`、3 個串行請求、2 個並行請求及斷流資源回收，另有本次 JSON／SSE 報告見 `tests/end2end/results/`。Mage-VL 圖片 JSON／SSE、視訊上傳各有成功記錄，並完成 Mage Live／MediaMTX 合成 RTSP 單次 smoke。Mage 圖片／影片設定見 [`checkpoints/val.md`](../../../checkpoints/val.md)、[`checkpoints/val.xlsx`](../../../checkpoints/val.xlsx)。各路徑樣本不足以建立容量基線或 SLA。 |
| Go Gateway 雙模型 capability 測試 | mock 與已執行真實路由 smoke 通過 | `go test -count=1 ./...` 分別於 `src/gateway/` 與 `tests/gateway/` 通過；mock 覆蓋 Mage 正向圖片／影片／Live、Qwen 正向文字／SSE，以及 Qwen 不支援的圖片／影片／Live 拒絕。真實 Gateway 路徑另覆蓋 Qwen JSON／SSE 與 Mage 圖片 JSON／SSE、影片及 Live；Live 未授權或過期句柄的媒體存取仍待驗收。 |
