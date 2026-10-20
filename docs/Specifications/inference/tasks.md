# 驗證任務：推論服務現況基線

> 關聯規格：[spec.md](spec.md)；關聯設計：[design.md](design.md)；提案：[proposal.md](proposal.md)  
> 現況：推論功能已存在；tests/inference 已包含靜態來源契約檢查，尚無 API runtime／模型行為測試。以下待辦主要補足回歸驗證，不代表推論服務待實作。  
> 本文件不表示任何 runtime 驗收已通過。

## 任務清單

### 1. 基線與啟動器

- [ ] **T-01 建立登錄表、CLI 與設定解析測試** — 目標：tests/inference/；交付：以臨時 YAML 驗證 list/start 選擇、預設 alias、路徑解析、CLI 覆寫與受保護環境變數；覆蓋：REQ-01 / AC-01.1、AC-01.2、AC-01.3；REQ-02 / AC-02.1、AC-02.2、AC-02.3、AC-02.4；依賴：無；狀態：未開始。
- [ ] **T-02 增加模型下載、載入、健康與關閉測試** — 目標：tests/inference/；交付：mock HF snapshot、processor、model、CUDA 與 lifespan teardown；覆蓋：REQ-03 / AC-03.1、AC-03.2、AC-03.3、AC-03.4；REQ-04 / AC-04.1、AC-04.2、AC-04.3；依賴：T-01；狀態：未開始。

### 2. 聊天與視訊 API

- [ ] **T-03 增加聊天 schema、圖片及生成參數測試** — 目標：tests/inference/；交付：使用本機合成圖片與 mock URL server 驗證 model、messages、圖片限制、生成邊界、上下文及回應 schema；覆蓋：REQ-05 / AC-05.1、AC-05.2、AC-05.3、AC-05.4；REQ-06 / AC-06.1、AC-06.2、AC-06.3、AC-06.4；REQ-07 / AC-07.1、AC-07.2、AC-07.3、AC-07.4；依賴：T-02；狀態：未開始。
- [ ] **T-04 增加 SSE、本文 middleware、上傳影片及前處理測試** — 目標：tests/inference/；交付：測試 SSE 成功／錯誤／DONE、Content-Length 邊界、frames fixture、codec 缺少依賴與輸出 metadata；覆蓋：REQ-08 / AC-08.1、AC-08.2、AC-08.3；REQ-09 / AC-09.1、AC-09.2、AC-09.3；REQ-10 / AC-10.1、AC-10.2、AC-10.3、AC-10.4；REQ-11 / AC-11.1、AC-11.2、AC-11.3、AC-11.4；REQ-12 / AC-12.1、AC-12.2、AC-12.3；依賴：T-03；狀態：未開始。

### 3. 即時視訊、安全差異與真機驗證

- [ ] **T-05 增加即時工作階段狀態與 MediaMTX callback 測試** — 目標：tests/inference/；交付：mock RTSP capture、時鐘及 analyzer 驗證建立、限制、重連、佇列上限、SSE replay、停止、cleanup 與授權 callback；覆蓋：REQ-13 / AC-13.1、AC-13.2、AC-13.3、AC-13.4；REQ-14 / AC-14.1、AC-14.2、AC-14.3、AC-14.4；REQ-15 / AC-15.1、AC-15.2、AC-15.3、AC-15.4；依賴：T-02、T-03；狀態：未開始。
- [ ] **T-06 解決規格中的聊天與安全待確認事項** — 目標：proposal.md、spec.md、design.md、src/inference/README.md；交付：由專案負責人決定純文字聊天契約及 image_url SSRF 防護方向，將決定同步至權威文件；覆蓋：REQ-05 / AC-05.4；REQ-06 / AC-06.2；REQ-16 / AC-16.1、AC-16.2、AC-16.3、AC-16.4；依賴：專案負責人決策；狀態：待決策。
- [ ] **T-07 在既有權重與授權硬體可用時執行受控 runtime smoke** — 目標：tests/inference/ 驗收紀錄；交付：記錄 GPU 型號、CUDA／PyTorch／Transformers 版本、模型 revision、啟動／health／關閉結果；不下載權重、不變更模型設定；覆蓋：REQ-03 / AC-03.2、AC-03.4；REQ-07 / AC-07.4；REQ-10 / AC-10.3；REQ-13 / AC-13.1；依賴：已有權重、已核准硬體與可控部署窗口；狀態：未開始。

### 4. 本次文件工作

本次工作建立本規格組並完成來源盤點。技能結構校驗器的結果記於交付摘要；它只驗證文件及 ID 連結，不驗證推論 runtime。以上 T-01 至 T-07 是後續測試、決策與真機驗收任務。

- [x] **T-08 建立推論來源靜態契約檢查** — 目標：tests/inference/test_source_contract.py；交付：檢查模型別名與上游 repo_id、推論路由宣告、Content-Length middleware 及 YAML 上限，測試不匯入模型 runtime；關聯：REQ-01 / AC-01.1（只檢查別名／repo_id 不變項）、REQ-09 / AC-09.3（只檢查設定值）；依賴：無；狀態：完成。此任務只提供來源防回歸檢查，不代表 API runtime 驗收已通過。

## 驗收記錄

| AC | 檢查方法 | 結果／證據 | 狀態 |
|---|---|---|---|
| REQ-01 靜態別名不變項；REQ-09 設定值 | `python -m unittest discover -s tests/inference -v` | 4 項標準函式庫測試通過；未匯入 FastAPI／推論服務 | 來源靜態檢查通過 |
| AC-01.1 至 AC-16.4 | 依各任務使用 mock、合成 fixture、API 測試或核准的 runtime smoke | 尚未執行推論 API runtime 測試；原始碼及 YAML 已檢視 | 待執行 |

## 實施偏差與決定

| 日期 | 偏差／新發現 | 規格／設計更新 | 決定人或依據 |
|---|---|---|---|
| 2026-10-20 | README 描述可接受文字／圖片，但來源碼要求至少一張圖片 | 已在 proposal.md、spec.md、design.md 標為待確認 | 以 main.py／service.py 現況為證，產品契約待維護者確認 |
| 2026-10-20 | image_url 可由推論主機抓取任意 HTTP(S) URL，無來源 allowlist | 已在安全風險及待決策記錄 | 依 service.py 的 URL 取得流程 |

## 完成摘要

- 已完成任務：本次反向盤點與四份 Spec 文件建立；T-08 來源靜態契約檢查。
- 未執行／未通過檢查：T-01 至 T-07 仍待完成；沒有載入模型、啟動 GPU 服務或 MediaMTX；推論 API runtime 行為尚未實測。
- 已同步的權威來源：本規格組引用 AGENTS.md、README.md、推論 README、launcher.py、model_registry.yaml、main.py、service.py 及三份 YAML。後續將 live 本文上限移入 inference.yml，並由 main.py 使用 service.py 暴露的設定值；預設值維持 65,536 bytes。
- 完成日期：2026-10-20

