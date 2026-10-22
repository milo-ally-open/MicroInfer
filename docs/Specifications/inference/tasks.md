# 實施與驗收任務：Inference Engine 五層架構

> 提案：[proposal.md](proposal.md)｜規格：[spec.md](spec.md)｜設計：[design.md](design.md)
> 任務狀態只反映本次五層改造證據。改造前已驗收的 Qwen3 基線保留於本地回復提交 `0cb3c56` 及 `checkpoints/val.md`、`checkpoints/val.xlsx`。

## 任務清單

- [x] **T-01 建立五層唯一規格與責任追蹤** — 範圍：本規格組四文件；交付：L1–L5 所有權、需求／驗收／任務 ID 映射、明確區分事實／決策／限制；覆蓋：REQ-01、REQ-02、REQ-03、REQ-04、REQ-05、REQ-06、REQ-07、REQ-08；依賴：無；狀態：已重寫並通過結構與追蹤校驗。
- [x] **T-02 將 Qwen 輸入處理與增量 Detokenize 放到 API 邊界** — 範圍：API input/output processor、聊天 route、Qwen `input_processor.py`、Adapter、Engine IPC contract；交付：`PreparedChatInput` IPC、token ID event、非串流與增量文字 SSE；覆蓋：REQ-01 / AC-01.1、AC-01.2、AC-01.3、AC-01.4、AC-01.5、AC-01.6、AC-01.7；依賴：T-01；狀態：程式及 CPU mock／source 契約測試通過；尚未執行模型服務驗收。
- [x] **T-03 抽出 Engine Output Processor 並驗證生命週期** — 範圍：`engine/output_processor.py`、`engine/engine.py`、Engine／SSE 測試；交付：Worker token／text 轉 Engine event、終止路徑清理和 stream isolation；覆蓋：REQ-02 / AC-02.1、AC-02.2、AC-02.3、AC-02.4；REQ-03 / AC-03.4；依賴：T-01、T-02；狀態：Engine lifecycle、取消、Worker failure 與事件契約測試通過。
- [x] **T-04 拆出 Executor 與 Platform 契約** — 範圍：`engine/executor.py`、`worker.py`、`platform.py`、Mage／Qwen `engine_core.py`、Qwen YAML 和 Worker process tests；交付：Engine 依賴 Executor、LocalProcessExecutor 管理單 Worker spawn/Pipe、Qwen 使用 TorchCudaPlatform；覆蓋：REQ-04 / AC-04.1、AC-04.2、AC-04.3、AC-04.4、AC-04.5；REQ-07 / AC-07.2、AC-07.3；依賴：T-01；狀態：程序／RPC／startup cleanup 與未登記 Platform 測試通過；未執行 CUDA 裝置驗收。
- [x] **T-05 抽出 Qwen InputBatch 並保持批次語義** — 範圍：Qwen `model/input_batch.py`、`model_runner.py`、native runtime tests；交付：批次 tensor、長度、offset、block table 對齊及不相容輸入拒絕；覆蓋：REQ-05 / AC-05.1、AC-05.2、AC-05.3、AC-05.4；REQ-03 / AC-03.1、AC-03.2、AC-03.3、AC-03.4、AC-03.5；依賴：T-01；狀態：InputBatch、合成 Safetensors 權重載入、變長 prefill/decode 數值對照及實體 page 所有權測試均在 CPU 環境通過；固定 snapshot／CUDA 驗收仍屬 T-08。
- [x] **T-06 驗證模型包生態相容與登錄邊界** — 範圍：Qwen `compatibility/`、模型 package、`model_registry.yaml`、Mage/Qwen source contract；交付：固定 revision、權重 key／shape 和 alias／backend 職責不變；覆蓋：REQ-06 / AC-06.1、AC-06.2、AC-06.3、AC-06.4；REQ-07 / AC-07.1、AC-07.4；依賴：T-01、T-04；狀態：固定 revision、模型 alias、API／Engine 分離及 Mage/Qwen 路徑契約測試通過。
- [x] **T-07 完成分層契約與 failure-path 自動測試** — 範圍：`tests/inference/`；確認 `tests/gateway/`、`tests/end2end/` 維持分開；交付：無權重 unit／source／IPC 測試和 API SSE／取消契約證據；覆蓋：REQ-08 / AC-08.1、AC-08.2；依賴：T-02、T-03、T-04、T-06；狀態：Inference 套件 73 項測試全部通過；合成 Runner／InputBatch 測試使用 CPU PyTorch，不載入實際模型權重。
- [ ] **T-08 使用固定 Qwen3-0.6B snapshot 驗收串流與性能** — 範圍：Qwen snapshot 與 `checkpoints/val.*` 工作流；交付：非串流／SSE／併發／取消回收和性能基線；覆蓋：REQ-08 / AC-08.3、AC-08.4、AC-08.5；依賴：T-02 至 T-07；狀態：待實機執行。性能記錄需與前一版本分開，不覆蓋既有工作流文件歷史。
- [x] **T-09 同步開發文件並通過 Spec 結構檢查** — 範圍：根目錄 `README.md`、`src/inference/README.md`、四份 Inference Spec；交付：實際目錄／指令／限制一致，Spec validator 及人工追蹤檢查；覆蓋：REQ-01、REQ-02、REQ-03、REQ-04、REQ-05、REQ-06、REQ-07、REQ-08；依賴：T-01、T-02、T-03、T-04；狀態：README 已同步，Spec validator 與任務映射校驗通過。

## 本次驗收紀錄

| 檢查 | 狀態 | 證據／限制 |
|---|---|---|
| 改造前本地回復提交 | 完成 | `0cb3c56 feat: add native Qwen3-0.6B inference engine`；本地提交，未推送。 |
| 五層程式改造 | CPU 契約驗收完成 | L1 Tokenize／Detokenize、L2 Engine output/event、L3 Executor／Platform 與 L4 InputBatch、Qwen 合成 Runner 均有 CPU 測試；固定 snapshot 與 CUDA 執行仍待 T-08。 |
| Inference Spec 重寫 | 已驗收 | 五層責任、8 項 REQ、38 項 AC 與 9 項任務完成結構驗證；逐項映射由 validator 通過。 |
| Python 單元／契約測試 | 通過 | `python -m unittest discover -s tests/inference -q` 共 73 項全數通過；CPU 驗證環境使用 PyTorch 2.14.1+cpu、Safetensors 0.8.0、NumPy 1.26.4、Pillow 12.3.0，未啟動 GPU 推理服務。 |
| Python 語法編譯 | 通過 | Windows workspace Python 3.12.14 執行 `compileall`，涵蓋 `src/inference/` 與 `tests/inference/`。 |
| Spec 結構與追蹤校驗 | 通過 | 執行 `project-spec-workflow/scripts/validate_spec_group.py docs/Specifications/inference`，8 REQ、38 AC、9 T 均可追蹤。 |
| 固定 snapshot API／SSE／GPU 驗收 | 尚未執行 | 不以改造前記錄推定新路徑通過；需使用本機 Qwen3-0.6B snapshot 重新驗收。 |
| Go Gateway／端到端性能 | 尚未執行 | 需先完成推理 API 驗收；性能資料另記於 `tests/end2end/` 工作流，不設 SLA。 |
