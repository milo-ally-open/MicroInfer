# 實施任務：Go API 閘道

> 關聯規格：[spec.md](spec.md)；關聯設計：[design.md](design.md)；提案：[proposal.md](proposal.md)  
> 狀態：T-01 至 T-12 已完成實作或測試工具交付；本地 mock／靜態測試已執行。真實模型、GPU、MediaMTX 與性能基線尚未驗證。
> Go Gateway v1 已實作，不代表已部署或已連接真實推論服務。

## 任務清單

### 1. 設定與服務骨架

- [x] **T-01 建立 Gateway 設定載入與服務骨架** — 交付：`src/gateway/go.mod`、`internal/config/`、`cmd/gateway/main.go`、`configs/gateway.yml`；嚴格載入 YAML、拒絕無效設定、啟動／停止 HTTP server；覆蓋：REQ-09 / AC-09.1；REQ-11 / AC-11.1、AC-11.2；狀態：完成。

### 2. 認證、路由與轉送

- [x] **T-02 實作 API Key、租戶識別與秘密載入** — 交付：`internal/auth/` 的 Bearer API Key、tenant mapping、常數時間雜湊比對及輪替 HMAC 句柄；API Key 由掛載 JSON 檔載入，簽章密鑰由環境變數注入；覆蓋：REQ-02 / AC-02.1、AC-02.2、AC-02.3；REQ-11 / AC-11.3、AC-11.4；狀態：完成。
- [x] **T-03 實作模型別名、模型清單與一般聊天代理** — 交付：只公開已設定的 `mage-vl-4b`，將聊天請求改成 `microsoft/Mage-VL` 並把回應 model 欄位改回別名；覆蓋：REQ-01 / AC-01.1、AC-01.2、AC-01.3；REQ-03 / AC-03.1、AC-03.2；狀態：完成。
- [x] **T-04 實作聊天 SSE 轉送** — 交付：有界 SSE event parser、model 別名改寫、逐事件 flush、完成標記、取消與閒置／總逾時處理；覆蓋：REQ-04 / AC-04.1、AC-04.2、AC-04.3；REQ-08 / AC-08.3；狀態：完成。
- [x] **T-05 實作視訊 multipart 有界代理** — 交付：`Content-Length` 檢查、YAML 本文上限、有界串流轉送及 multipart boundary 保留；覆蓋：REQ-05 / AC-05.1、AC-05.2、AC-05.3；REQ-07 / AC-07.1；狀態：完成。
- [x] **T-06 實作即時工作階段簽名句柄** — 交付：租戶綁定 HMAC 句柄、事件／停止路由驗證、`Last-Event-ID` 轉送與 Gateway URL 改寫；MediaMTX publish URL 保持原值；覆蓋：REQ-06 / AC-06.1、AC-06.2、AC-06.3、AC-06.4；REQ-11 / AC-11.4；狀態：完成。
- [x] **T-07 實作租戶限流與本文政策** — 交付：YAML 速率／突發值、程序內 tenant token bucket、`Retry-After`、本文長度與位元組上限檢查；覆蓋：REQ-07 / AC-07.1、AC-07.2、AC-07.3；狀態：完成。
- [x] **T-08 實作錯誤契約、上游逾時及失敗處理** — 交付：OpenAI 風格錯誤、上游 4xx/5xx 正規化、連線失敗 502、逾時 504、串流中斷日誌；覆蓋：REQ-08 / AC-08.1、AC-08.2、AC-08.3；狀態：完成。

### 3. 健康檢查、觀測與驗收

- [x] **T-09 實作健康與就緒端點** — 交付：免認證 `/healthz` 存活與 `/readyz` 上游模型就緒檢查；覆蓋：REQ-09 / AC-09.1、AC-09.2、AC-09.3；狀態：完成。
- [x] **T-10 加入安全日誌與服務指標** — 交付：request ID、路由／租戶雜湊／狀態／上游與總延遲日誌，Prometheus request/error/histogram 指標，不輸出 API Key、請求本文或租戶標籤；覆蓋：REQ-10 / AC-10.1、AC-10.2、AC-10.3；狀態：完成。
- [x] **T-11 建立分層測試、回復流程與文件同步** — 交付：`tests/gateway/` mock API 測試、`tests/inference/` 靜態契約測試、`tests/end2end/` 客戶端與設定相容測試、端到端性能基線工具、啟動／回復文件及 README 同步；覆蓋：REQ-12 / AC-12.1、AC-12.2、AC-12.3、AC-12.4；狀態：工具及文件完成，真實性能／回復演練未執行。
- [x] **T-12 將 Gateway HTTP 能力拆分至職責套件** — 交付：`internal/route/` 路徑分派、`internal/upstream/` HTTP transport、`internal/stream/` SSE 轉送、`internal/videoproxy/` multipart 串流代理；由 `gateway.go` 組合各元件，不改變公開 API 契約；覆蓋：REQ-01 / AC-01.1、AC-01.2；REQ-03 / AC-03.1、AC-03.2；REQ-04 / AC-04.1、AC-04.2；REQ-05 / AC-05.1、AC-05.2、AC-05.3；REQ-08 / AC-08.1、AC-08.2、AC-08.3；REQ-09 / AC-09.2；REQ-12 / AC-12.1、AC-12.2；狀態：完成，`tests/gateway/` 整合測試通過。

## 驗收記錄

| AC | 檢查方法 | 結果／證據 | 狀態 |
|---|---|---|---|
| AC-01.1、AC-01.2、AC-01.3；AC-02.1、AC-02.2；AC-03.1、AC-03.2 | `go test ./...`（`tests/gateway/`） | models alias、認證、上游模型映射、一般回應改寫及不轉送 API Key 測試 | 通過 mock 測試 |
| AC-04.1、AC-04.2 | `TestChatSSERewritesAliasAndPreservesDone` | SSE 分塊別名與 `[DONE]` 保留 | 通過 mock 測試 |
| AC-04.3 | 程式碼檢視 | request context 取消會關閉上游請求；沒有專門客戶端斷線測試 | 程式已實作，runtime case 待補 |
| AC-05.1、AC-05.2；AC-07.1、AC-07.2 | multipart、Content-Length、token bucket 整合測試 | boundary／本文相同、缺少長度 411、本文過大 413、速率超限 429 | 通過 mock 測試 |
| AC-05.3 | `tests/end2end/test_config_compatibility.py` | gateway chat/video/live request limits 均不高於 inference YAML 對應值 | 通過靜態設定檢查 |
| AC-06.1、AC-06.2、AC-06.3 | live handle 測試 | URL 改寫、租戶隔離、密鑰輪替和簽章竄改拒絕 | 通過 mock 測試 |
| AC-06.4 | 尚未連接 MediaMTX | 推論服務 callback 拓撲未改動；真實 publish/read 授權未測 | 未執行 |
| AC-07.3、AC-08.1、AC-08.2、AC-09.1、AC-09.2 | Go handler 測試與程式碼檢視 | 程序內限流、OpenAI error、502、ready／health 有 mock 驗證；504 專門延遲 case 未執行 | 部分通過 |
| AC-08.3、AC-10.1、AC-10.2 | 日誌測試及 SSE 實作檢視 | request ID、上游延遲欄位存在；API Key／請求本文不出現在日誌 | 通過 mock 測試 |
| AC-10.3 | `/metrics` 整合測試與 `performance_baseline.py` 統計測試 | 請求、TTFT、串流及上傳完成 histogram 可輸出；真實 p50／p95／p99 尚無量測資料 | 工具測試通過；基線未執行 |
| AC-11.1、AC-11.2、AC-11.3、AC-11.4 | YAML 載入、設定相容檢查、mock handler 測試 | 已載入 checked-in YAML，測試 API key／HMAC 以程式注入；未以部署密鑰啟動服務 | 靜態／mock 通過；部署驗證待執行 |
| AC-12.1、AC-12.2 | 原始碼與目錄檢視 | Gateway 不匯入推論套件；測試分置 gateway、inference、end2end | 通過 |
| AC-12.3、AC-12.4 | 性能工具／回復文件檢視 | 已提供性能基線程式與回復步驟；未連 GPU/MediaMTX，也未做部署回復演練 | 工具／文件完成，實測未執行 |
| T-12 涉及之 API／串流／上游驗收條件 | `go test -count=1 ./...`（`tests/gateway/`）及 `go test -count=1 ./...`（`src/gateway/`） | 新增模組編譯；既有 mock handler 測試涵蓋模型路由、SSE 別名、multipart boundary／本文、上游錯誤及健康檢查 | 通過本機 mock 測試；未連真實推論服務 |

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
- 未執行：真實推論 API／GPU／MediaMTX 整合、真實端到端性能基線、部署回復演練；沒有下載模型權重。
- 未新增或修改：`checkpoints/val.md`、`checkpoints/val.xlsx`、`test.jpg`、`test.mp4`。
- 不代表：已選定第二個模型，或 SGLang／vLLM 已實作。
