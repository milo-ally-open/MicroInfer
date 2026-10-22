# 實施任務：Go API 閘道

> 關聯規格：[spec.md](spec.md)；關聯設計：[design.md](design.md)；提案：[proposal.md](proposal.md)  
> 狀態：T-01 至 T-12 記錄 Gateway v1 基線；T-13 多模型能力路由已通過本機 mock 與跨設定測試。2026-10-22 另以真實服務完成 Qwen3-0.6B 文字 JSON／SSE、Mage-VL 圖片 JSON／SSE／視訊上傳及 Mage Live／MediaMTX 合成 RTSP 功能 smoke。Live 僅有單次功能驗證；授權失敗、部署回復及正式容量仍未驗證，性能樣本不足以形成穩定基線。
> Go Gateway v1 已實作，不代表已部署或已連接真實推論服務。

## 任務清單

### 1. 設定與服務骨架

- [x] **T-01 建立 Gateway 設定載入與服務骨架** — 交付：`src/gateway/go.mod`、`internal/config/`、`cmd/gateway/main.go`、`configs/gateway.yml`；嚴格載入 YAML、拒絕無效設定、啟動／停止 HTTP server；覆蓋：REQ-09 / AC-09.1；REQ-11 / AC-11.1、AC-11.2；狀態：完成。

### 2. 認證、路由與轉送

- [x] **T-02 實作 API Key、租戶識別與秘密載入** — 交付：`internal/auth/` 的 Bearer API Key、tenant mapping、常數時間雜湊比對及輪替 HMAC 句柄；API Key 由掛載 JSON 檔載入，簽章密鑰由環境變數注入；覆蓋：REQ-02 / AC-02.1、AC-02.2、AC-02.3；REQ-11 / AC-11.3、AC-11.4；狀態：完成。
- [x] **T-03 實作模型別名、模型清單與一般聊天代理（初始 Mage 路由）** — 交付：建立 `mage-vl-4b` → `microsoft/Mage-VL` 的起始路由；雙模型 capability 路由由 T-13 擴充；覆蓋：REQ-01 / AC-01.1、AC-01.2、AC-01.3；REQ-03 / AC-03.1、AC-03.2；狀態：完成初始基線。
- [x] **T-04 實作聊天 SSE 轉送** — 交付：有界 SSE event parser、model 別名改寫、逐事件 flush、完成標記、取消與閒置／總逾時處理；覆蓋：REQ-04 / AC-04.1、AC-04.2、AC-04.3；REQ-08 / AC-08.3；狀態：完成。
- [x] **T-05 實作視訊 multipart 有界代理** — 交付：`Content-Length` 檢查、YAML 本文上限、有界串流轉送及 multipart boundary 保留；覆蓋：REQ-05 / AC-05.1、AC-05.2、AC-05.3；REQ-07 / AC-07.1；狀態：完成。
- [x] **T-06 實作即時工作階段簽名句柄** — 交付：租戶綁定 HMAC 句柄、事件／停止路由驗證、`Last-Event-ID` 轉送與 Gateway URL 改寫；MediaMTX publish URL 保持原值；覆蓋：REQ-06 / AC-06.1、AC-06.2、AC-06.3、AC-06.4；REQ-11 / AC-11.4；狀態：完成。
- [x] **T-07 實作租戶限流與本文政策** — 交付：YAML 速率／突發值、程序內 tenant token bucket、`Retry-After`、本文長度與位元組上限檢查；覆蓋：REQ-07 / AC-07.1、AC-07.2、AC-07.3；狀態：完成。
- [x] **T-08 實作錯誤契約、上游逾時及失敗處理** — 交付：OpenAI 風格錯誤、上游 4xx/5xx 正規化、連線失敗 502、逾時 504、串流中斷日誌；覆蓋：REQ-08 / AC-08.1、AC-08.2、AC-08.3；狀態：完成。

### 3. 健康檢查、觀測與驗收

- [x] **T-09 實作健康與就緒端點** — 交付：免認證 `/healthz` 存活與 `/readyz` 上游模型就緒檢查；覆蓋：REQ-09 / AC-09.1、AC-09.2、AC-09.3；狀態：完成。
- [x] **T-10 加入安全日誌與服務指標** — 交付：request ID、路由／租戶雜湊／狀態／上游與總延遲日誌，Prometheus request/error/histogram 指標，不輸出 API Key、請求本文或租戶標籤；覆蓋：REQ-10 / AC-10.1、AC-10.2、AC-10.3；狀態：完成。
- [x] **T-11 建立分層測試、回復流程與文件同步** — 交付：`tests/gateway/` mock API 測試、`tests/inference/` 靜態契約測試、`tests/end2end/` 客戶端與設定相容測試、端到端性能工具、啟動／回復文件及 README 同步；覆蓋：REQ-12 / AC-12.1、AC-12.2、AC-12.3、AC-12.4；狀態：工具及文件完成；Qwen 與 Mage-VL 已有低樣本真實服務 smoke，正式性能基線及部署回復演練未完成。
- [x] **T-12 將 Gateway HTTP 能力拆分至職責套件** — 交付：`internal/route/` 路徑分派、`internal/upstream/` HTTP transport、`internal/stream/` SSE 轉送、`internal/videoproxy/` multipart 串流代理；由 `gateway.go` 組合各元件，不改變公開 API 契約；覆蓋：REQ-01 / AC-01.1、AC-01.2；REQ-03 / AC-03.1、AC-03.2；REQ-04 / AC-04.1、AC-04.2；REQ-05 / AC-05.1、AC-05.2、AC-05.3；REQ-08 / AC-08.1、AC-08.2、AC-08.3；REQ-09 / AC-09.2；REQ-12 / AC-12.1、AC-12.2；狀態：完成，`tests/gateway/` 整合測試通過。
- [x] **T-13 登記兩模型 capability 並路由至各自推論服務** — 交付：Gateway YAML 的 Mage／Qwen route、輸入與路由能力檢查、影片／Live 模型選擇、Live 句柄模型綁定、多上游 readiness；覆蓋：REQ-01 / AC-01.1、AC-01.2；REQ-03 / AC-03.1；REQ-04 / AC-04.2；REQ-06 / AC-06.2；REQ-09 / AC-09.2；REQ-13 / AC-13.1、AC-13.2、AC-13.3、AC-13.4、AC-13.5、AC-13.6；狀態：Gateway mock 測試、跨設定測試及 Spec 結構檢查通過；另有 Qwen 文字 JSON／SSE、Mage 圖片 JSON／SSE／視訊上傳、Mage Live／MediaMTX 合成 RTSP 真實 Gateway smoke。Live 與媒體授權僅完成單次功能檢查，未驗證失敗授權與到期邊界。

## 驗收記錄

| AC | 檢查方法 | 結果／證據 | 狀態 |
|---|---|---|---|
| AC-01.1、AC-01.2、AC-01.3；AC-02.1、AC-02.2；AC-03.1、AC-03.2 | `go test ./...`（`tests/gateway/`） | models alias、認證、上游模型映射、一般回應改寫及不轉送 API Key 測試 | 通過 mock 測試 |
| AC-04.1、AC-04.2 | `TestChatSSERewritesAliasAndPreservesDone` | SSE 分塊別名與 `[DONE]` 保留 | 通過 mock 測試 |
| AC-04.3 | 程式碼檢視 | request context 取消會關閉上游請求；沒有專門客戶端斷線測試 | 程式已實作，runtime case 待補 |
| AC-05.1、AC-05.2；AC-07.1、AC-07.2 | multipart、Content-Length、token bucket 整合測試 | boundary／本文相同、缺少長度 411、本文過大 413、速率超限 429 | 通過 mock 測試 |
| AC-05.3 | `tests/end2end/test_config_compatibility.py` | gateway chat/video/live request limits 均不高於 inference YAML 對應值 | 通過靜態設定檢查 |
| AC-06.1、AC-06.2、AC-06.3 | live handle 測試 | URL 改寫、租戶隔離、密鑰輪替和簽章竄改拒絕 | 通過 mock 測試 |
| AC-06.4 | Mage Live／MediaMTX 合成 RTSP 單次 smoke | Gateway 建立、事件與停止均 HTTP 200；FFmpeg 發布的影格經 MediaMTX 讀取並觸發 `analysis`。句柄過期、非作用中 session 拒絕及未授權 publish/read 未測；見 [`live smoke report`](../../../tests/end2end/results/mage-vl-4b-live-smoke-position-fix-2026-10-22.json) | 部分驗證；授權邊界待測 |
| AC-07.3、AC-08.1、AC-08.2、AC-09.1、AC-09.2 | Go handler 測試與程式碼檢視 | 程序內限流、OpenAI error、502、ready／health 有 mock 驗證；504 專門延遲 case 未執行 | 部分通過 |
| AC-08.3、AC-10.1、AC-10.2 | 日誌測試及 SSE 實作檢視 | request ID、上游延遲欄位存在；API Key／請求本文不出現在日誌 | 通過 mock 測試 |
| AC-10.3 | `/metrics` 整合測試與 `performance_baseline.py` 統計測試 | Gateway 請求、TTFT、串流及上傳完成 histogram 可輸出；Qwen text-sse 有 3 個請求，另有 JSON／SSE 各 1 次；Mage 圖片 JSON／SSE、視訊上傳及 Live／MediaMTX 合成 RTSP 各 1 次；Mage 設定見 [`checkpoints/val.md`](../../../checkpoints/val.md)、[`checkpoints/val.xlsx`](../../../checkpoints/val.xlsx)，其餘見 `tests/end2end/results/`；樣本不足以代表容量 | mock 測試與各項低樣本功能 smoke 通過 |
| AC-11.1、AC-11.2、AC-11.3、AC-11.4 | YAML 載入、設定相容檢查、mock handler 測試 | 已載入 checked-in YAML，測試 API key／HMAC 以程式注入；未以部署密鑰啟動服務 | 靜態／mock 通過；部署驗證待執行 |
| AC-12.1、AC-12.2 | 原始碼與目錄檢視 | Gateway 不匯入推論套件；測試分置 gateway、inference、end2end | 通過 |
| AC-12.3、AC-12.4 | 性能工具／回復文件檢視及真實服務 smoke | 已提供性能工具與回復步驟；Qwen text-sse、Mage 圖片／視訊低樣本 smoke 已連 GPU，Mage Live／MediaMTX 完成一次合成 RTSP smoke；未執行部署回復演練 | 工具／文件完成；功能 smoke 完成，正式容量及回復演練待執行 |
| T-12 涉及之 API／串流／上游驗收條件 | `go test -count=1 ./...`（`tests/gateway/`）及 `go test -count=1 ./...`（`src/gateway/`） | 新增模組編譯；既有 mock handler 測試涵蓋模型路由、SSE 別名、multipart boundary／本文、上游錯誤及健康檢查 | 通過本機 mock 測試；未連真實推論服務 |
| AC-13.1 至 AC-13.5 | Go Gateway 整合測試及真實服務 smoke | Qwen route／模態拒絕、Mage 圖片／影片路由、Live 句柄與多上游 readiness mock cases 通過；真實鏈路已完成 Qwen JSON／SSE、Mage 圖片 JSON／SSE／視訊上傳及 Live／MediaMTX RTSP smoke，報告見 [`tests/end2end/results/`](../../../tests/end2end/results/) 與 [`checkpoints/val.md`](../../../checkpoints/val.md) | Mock capability cases 與各已執行真實路徑通過；Live 單次 smoke，不代表授權邊界或容量驗收 |
| AC-13.6 | `python -m unittest discover -s tests/end2end -q` | 10 項測試通過，包含兩模型 alias、上游識別碼、能力和 network URL 一致性 | 通過靜態設定契約 |
| REQ-13 規格追蹤 | `validate_spec_group.py docs/Specifications/gateway` | 13 REQ、44 AC、13 T 的需求／驗收／任務映射完整 | 通過 |

## 實施偏差與決定

| 日期 | 偏差／新發現 | 規格／設計更新 | 依據 |
|---|---|---|---|
| 2026-10-20 | 原推論 middleware 將 live 建立及 MediaMTX callback 的 65,536 bytes 上限寫在 Python 常數 | 新增 `inference.yml` 的 `api.max_live_request_bytes`，兩路由共用該值；Gateway 設定相容測試同步檢查三類請求本文上限 | 符合所有可調推論參數留在推論 YAML 的專案規範，預設行為不變 |
| 2026-10-20 | Gateway v1 的路由、SSE、上游 HTTP 與影片 multipart 實作原先集中在 `src/gateway/gateway.go` | 已拆分至 `internal/route/`、`internal/stream/`、`internal/upstream/`、`internal/videoproxy/`，並同步 README 與技術設計 | 以職責邊界拆分；`gateway.go` 保留應用組裝與跨套件業務流程，既有 API mock 測試通過 |
| 2026-10-20 | 推論程式要求聊天至少含一個 `image_url`，但原 Gateway 規格概稱文字／圖片聊天 | README 與規格補註現行推論契約；Gateway 不改變此推論端行為 | 以 `service.py` 實際輸入整理邏輯為準，純文字契約仍待維護者決定 |

## 完成摘要

- 已完成：Go Gateway v1、YAML 設定、API Key／租戶、模型別名、聊天 JSON／SSE、視訊串流上傳、live 控制句柄、限流、錯誤、health/readiness、日誌／metrics、分層測試及客戶端／性能工具。
- 已完成套件拆分：`route`、`stream`、`upstream`、`videoproxy` 各自承接路由分派、SSE、上游 transport 與影片 multipart 代理。
- 已執行：Go Gateway 套件編譯與單元測試；`tests/inference/` 靜態來源檢查；`tests/end2end/` mock client、性能統計與 YAML 相容性檢查。具體結果在本次交付摘要記錄。
- 已執行：Qwen3-0.6B 真實 CUDA API 經 Gateway 的非串流／SSE smoke 與 3 個串行 text-sse 性能樣本；報告見 `tests/end2end/results/qwen3-0.6b-gateway-smoke-2026-10-22.json`。
- 已執行：Qwen3-0.6B Gateway JSON／SSE；Mage-VL Gateway 圖片 JSON／SSE／視訊上傳；Mage Live／MediaMTX 合成 RTSP 功能 smoke。Mage 圖片／影片設定與量測見 [`checkpoints/val.md`](../../../checkpoints/val.md)、[`checkpoints/val.xlsx`](../../../checkpoints/val.xlsx)，JSON 與 Live 報告見 `tests/end2end/results/`。仍未執行：未授權媒體存取／過期句柄測試、正式容量／高併發性能與部署回復演練；本次未下載模型權重。
- 未新增或修改：`checkpoints/val.md`、`checkpoints/val.xlsx`、`test.jpg`、`test.mp4`。
- 不代表：已選定第二個模型，或 SGLang／vLLM 已實作。
