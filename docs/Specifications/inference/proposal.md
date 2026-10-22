# 提案：以 Qwen3-0.6B 驅動三層推理引擎

> 規格：[spec.md](spec.md)｜設計與調研：[design.md](design.md)｜實施任務：[tasks.md](tasks.md)
> 狀態：Qwen3 三層架構及其本地模型計算圖、實體 KV pool、批次 Runner 已實作；端到端性能基線及完整 failure-path 驗收依 `tasks.md` 紀錄。Mage-VL 路徑維持現況。

## 背景與已驗證基線

專案已登錄 `mage-vl-4b` 與 `qwen3-0.6b`。Qwen3-0.6B 固定使用 `Qwen/Qwen3-0.6B` revision `c1899de289a04d12100db370d81485cdf75e47ca`，服務預設監聽 `127.0.0.1:8011`。重構前 Runner 使用 `AutoModelForCausalLM`、逐 request forward 及 Transformers 動態 Cache；現行 Qwen 使用本地 PyTorch Qwen3 計算圖、批次 Runner 和 GPU 實體 KV page pool。模型權重映射、五 token greedy 相容、變長雙序列 prefill/decode、普通 Chat、並發 SSE、finish usage 與 `[DONE]` 已在 RTX 4060 Laptop GPU 驗證。端到端性能及客戶端斷流後 page 回收證據仍見 `tasks.md`。

## 目標與決策

Inference Engine 以三層形成明確且可替換的所有權邊界：

1. **生態兼容層**：讀取 Hugging Face config、tokenizer/chat template 與 Safetensors 權重格式；驗證固定模型 revision 和權重映射。此層不執行模型 forward 或 `generate()`。
2. **性能優化層**：由專案持有 Qwen3-0.6B 模型結構、批次 Model Runner、Token Scheduler、GPU Worker 及實體 paged KV pool。首版使用 PyTorch 與 SDPA，保留後續替換 Attention／算子的明確模組邊界；不把未實作的自訂 CUDA kernel 或 PagedAttention 寫成現況。
3. **對外接口層**：沿用共用 FastAPI 路由、EngineClient、OpenAI Chat Completions schema 與逐 token SSE 序列化，不讓模型或 cache 實作滲入 HTTP contract。

生態兼容層使用 Transformers 的 `AutoConfig`、`AutoTokenizer` 及 chat template，並讀取 Safetensors checkpoint；Qwen 模型計算圖與 cache 寫入改由本地模型程式負責。Qwen3 的模型別名、revision、`/v1/chat/completions`、`stream: true` 的逐 token內容、finish usage、錯誤事件及 `[DONE]` 保持不變。Mage-VL 不在本次遷移範圍。

## 範圍

- 更新 `docs/Specifications/inference/`，為三層邊界、Qwen3 模型執行、批次 prefill/decode、實體 KV block 與 SSE 相容性建立 REQ／AC／T 追蹤。
- 在 `models/qwen3_0_6b/` 建立 Hugging Face 生態載入、固定 revision Safetensors 權重映射，以及自有 Qwen3 模型／Attention／Runner／KV pool 模組。
- 將 YAML 的固定物理 block 數、block token 數、最大序列數、每輪 token 預算及 prefill chunk 大小同時提供給 Engine KV allocator 與 GPU Worker；請求 block ID 必須直接對應 Worker 中的物理 KV slot。
- 對相同 phase 的相容請求執行批次模型 forward；結果與 token 事件按 request ID 分流。不同長度 prefill 使用明確 mask；decode 每 request 每輪最多產生一個 token。
- 新增不需權重的模型映射、cache isolation、批次及取消測試；使用既有本機 Qwen snapshot 驗證 logits／token 相容性、服務啟動與 SSE。
- 保留既有 `tests/inference/`、`tests/gateway/`、`tests/end2end/` 分層；不改 Go Gateway、Mage-VL 程式或 API route/schema。

## 非目標

- 不引入 vLLM 或 nano-vLLM 為執行依賴，也不複製其授權程式碼或宣稱已對齊其效能。
- 不在首輪編寫自訂 CUDA/Triton attention kernel，不宣稱已實作 PagedAttention；首版以自有物理 block pool 配合 PyTorch SDPA，Attention 讀取方式及其複製成本須如實記錄。
- 不實作 prefix cache、KV eviction/offload、量化、Tensor Parallel、多 GPU 或 Qwen-VL；不預設下一個模型。
- 不做資料集精度評測；與 Transformers 參考實作的數值比對只驗證本地模型程式及權重映射，不代表模型品質評估。
- 不下載模型、不改模型快照、不操作遠端主機、不提交或推送。只使用已存在的本機權重及已授權的本機 GPU 驗收。

## 風險與回復

主要風險是自有 Qwen3 模組與固定 revision 的權重命名、RoPE／QK RMSNorm、GQA、padding mask 或 chat template 行為不相容。Worker 必須在 ready 前驗證模型設定、必要權重鍵與張量形狀；模型數值對照或服務啟動失敗時，不得靜默退回不受 Scheduler 控制的 `generate()`。目前未保留舊 Runner fallback；需要回復時應透過版本控制回復 Qwen 專屬改動。API 路由和 SSE serializer 不需回退或更改。實體 KV pool 容量以 YAML 固定並在 Worker 啟動時實際分配；分配失敗則 Worker 不 ready。

## 完成條件

- 三層責任及依賴方向在程式、文件和測試中一致；模型別名與既有 API 契約保持不變。
- Qwen3 模型 forward 不使用 `AutoModelForCausalLM` 或 `generate()`；配置、tokenizer/chat template 及 Safetensors 仍可透過生態兼容層載入。
- Scheduler 分派的 block ID 對應 Worker 實際 GPU KV page；完成、拒絕、取消、異常和 Worker 關閉後均可回收，且跨請求不洩漏內容。
- 相容的同階段請求由 Runner 批次執行；逐 request 的輸出 token 仍即時轉為原有 SSE chunk。
- 無權重測試、固定 snapshot 數值相容、API／SSE runtime 與 RTX 4060 記憶體／性能觀測分別記錄；不得以 mock 替代實機證據。

## 變更記錄

| 日期 | 內容 |
|---|---|
| 2026-10-22 | 原五元件 Engine、Mage-VL 元件及三程序 IPC 的歷史驗收見 `tasks.md`。 |
| 2026-10-22 | 提出以 Qwen3-0.6B 驗證生態兼容、性能優化及對外接口三層的深度重構；不改 Mage-VL 與外部文字 SSE 契約。 |
