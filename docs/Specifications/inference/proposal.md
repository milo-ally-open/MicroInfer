# 提案：以 Qwen3-0.6B 驗證五層推理引擎

> 行為規格：[spec.md](spec.md)｜技術設計：[design.md](design.md)｜實施與驗收：[tasks.md](tasks.md)
> 狀態：五層架構正在落實。Qwen3-0.6B 是本次實驗模型；尚未完成的驗收以 `tasks.md` 標示。

## 背景與已驗證基線

推理服務目前包含共用 FastAPI API、Engine Core、Token Scheduler、KV Cache Manager 和子程序 Model Worker。Mage-VL-4B 維持 Transformers 推理；Qwen3-0.6B 已有本地 PyTorch 計算圖、批次 prefill/decode、Safetensors 載入及 Worker 實體 KV page pool。兩者共用對外 API 與 Engine 主鏈路，但模型輸入、計算圖、Cache 實作不同。

改造前，Qwen3 Worker 會在生成步驟將 token ID 解碼成文字；Worker 程序管理和 Engine Core RPC 集中在 `engine/worker.py`；Qwen Runner 在 forward 前組裝批次張量。這使 API 輸入處理、Engine 輸出轉換、執行器、硬體平台和 Runner 批次組裝的責任邊界仍不夠清晰。

既有 RTX 4060 驗收及性能記錄保存在 `checkpoints/val.md`、`checkpoints/val.xlsx`。這些是先前版本的基線證據；本次結構改造後的 SSE、Executor 與 InputBatch 行為仍需按 `tasks.md` 重新驗收。

## 目標與決策

以 vLLM V1 的職責拆分為參考，將推理引擎明確劃分為五層：

1. **L1 接口與門面層**：HTTP／OpenAI schema、模型輸入處理、增量 Detokenize、SSE 序列化及 EngineClient。
2. **L2 編排與調度層**：Engine Core 請求生命週期、Token Scheduler、KV reservation 與 Worker 結果事件化。
3. **L3 執行抽象與 Worker 層**：Executor 管理程序及 IPC；Platform 封裝裝置和記憶體觀測；Worker 在子程序載入模型。
4. **L4 模型運行時層**：Model Runner 組裝 `InputBatch`、執行批次 forward、管理生成狀態和呼叫 Sampler。
5. **L5 模型與算子層**：模型拓撲、模型元件、權重映射及模型生態相容程式。

模型相容能力會橫跨 L1 輸入適配與 L5 權重／模型適配；效能能力會橫跨 L2 至 L5。五層用來界定程式責任，不要求每層獨立成程序或目錄。API Server、Engine Core、Model Worker 仍維持現有程序拓樸。

本次決策：

- 保持 Mage-VL 與 Qwen3 的模型包、載入方式及能力差異；不把任何一個模型的前處理或計算圖移入 Go Gateway 或共用 Engine。
- Qwen3 API 程序使用模型專屬 `InputProcessor` 載入本機 Tokenizer，產生可序列化的 `PreparedChatInput`；不在 API 程序載入模型權重或建立 GPU Runtime。
- Qwen3 Worker 回傳 token ID；API 輸出處理器按 tokenizer 規則逐步形成文字 chunk。Mage-VL 現有文字增量事件維持原路徑。
- L2 `EngineOutputProcessor` 只把 Worker token／文字結果轉成 Engine 事件，不負責 HTTP、JSON 或 SSE。
- L3 先提供 `Executor`、`Platform` 契約及目前實際使用的 `LocalProcessExecutor`、`TorchCudaPlatform`。Ray、多 GPU、其他硬體平台不宣稱已實作。
- L4 `InputBatch` 明確承載本輪 token 張量、長度、位置偏移和每 request block table；目前每輪建立，不宣稱已採用固定緩衝區或 CUDA Graph。
- L5 Qwen3 自有 PyTorch 模型計算圖；Transformers 僅提供 config／tokenizer 等生態相容能力，Safetensors loader 負責權重映射。

## 範圍

- 重寫本目錄 `proposal.md`、`spec.md`、`design.md`、`tasks.md`，以五層責任和驗收追蹤取代舊的三層規格文字。
- 將 Qwen3 tokenization／增量 detokenization 放到 API 輸入輸出邊界，保持 OpenAI Chat、逐步文字 SSE、finish usage、`[DONE]` 和取消收尾。
- 將 Engine Core 的 Worker RPC 改由 Executor 介面承接，並以 Platform 介面處理 Qwen CUDA 裝置與記憶體觀測。
- 將 Qwen 每輪批次張量組裝抽成 `InputBatch`，保留目前 prefill/decode 批次和每 request cache 隔離。
- 透過合成／契約測試及固定 Qwen3-0.6B 本機 snapshot 驗證；不把舊版本 GPU 結果當作新架構的驗收結果。

## 非目標

- 不改 Go Gateway、Mage-VL 模型元件或既有模型服務 API；不把 API Server 與 Engine Core 改成遠端服務。
- 不引入 vLLM 或 nano-vLLM runtime 依賴，不複製其專用程式碼，也不宣稱 vLLM 相容或達到其效能。
- 不在本次實作 Ray／多 GPU／Tensor Parallel、自訂 CUDA/Triton Attention、PagedAttention kernel、prefix cache、KV offload、量化、CUDA Graph 或靜態 GPU InputBatch。
- 不選定第二個模型；Qwen3-0.6B 是目前實驗模型，Mage-VL 只作為既有共用 API／Engine 相容路徑。
- 不做模型資料集精度評測，不下載大型模型，不啟動 GPU 或 MediaMTX 作為文件編輯流程的一部分。

## 風險、回復與提交基線

主要風險是 API／Worker token 契約變更造成 SSE 內容、分塊時機或取消清理退化。所有 token event 必須保留 request ID；模型增量輸出需與最終文字一致。若 tokenizer 在已送出的文字前綴上重新解碼出不同內容，輸出處理器必須報錯，不得靜默重複或丟字。

KV block 所有權仍分成 Engine 的邏輯 reservation 和 Qwen Worker 的 GPU 實體 page；本次不把這種映射描述成 Attention kernel 的分頁直接讀取。Executor 啟動失敗、逾時或正常關閉須維持 Worker 收尾行為。

改造前的工作樹已建立本地回復提交 `0cb3c56`（`feat: add native Qwen3-0.6B inference engine`）。它只在本地 Git 歷史中，沒有推送。此次五層改造也只保存在本地 Git 歷史，不推送遠端；保留 `checkpoints/val.md`、`checkpoints/val.xlsx` 原內容。

## 完成條件

- 五層責任、實際模組位置、依賴方向及限制在本規格、Inference README、程式和測試一致。
- Qwen API 的 Tokenize、Engine domain event、Worker token ID、增量 Detokenize、OpenAI SSE 順序形成可測鏈路；Mage-VL 文字事件路徑仍通過原有契約。
- Engine 依賴 `Executor` 契約；線上實作由 `LocalProcessExecutor` 執行 Worker 子程序。Qwen Runner 透過 YAML 選用 `torch_cuda` 平台並使用 `InputBatch`。
- 所有新增／更新任務均以唯一 `REQ`、`AC`、`T` ID 互相追蹤；只將實際執行的檢查記為通過。
- 固定 snapshot 的 API／SSE／取消及性能驗收完成前，不宣稱本次改造已通過真實模型整合。
