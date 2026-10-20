# VideoStreamAPI 專案協作規範

## 專案目標

本專案用於建置視訊理解模型的 API 服務與模型無關的 API 閘道。閘道和推論服務是不同元件，必須分開維護：閘道處理外部客戶端接入，推論服務負責模型載入、GPU 推理及模型專屬功能。

## 目前狀態

- `src/inference/` 已有 Mage-VL-4B 的 Transformers 推論 API、模型登錄表及啟動器。
- `src/gateway/` 已實作 Go API 閘道 v1；目前能力與限制見 `src/gateway/README.md` 及 `docs/Specifications/gateway/`。
- 目前已登錄的模型為 `mage-vl-4b`，上游模型是 `microsoft/Mage-VL`。不要把專案別名和 Hugging Face `repo_id` 混為一談。
- SGLang、vLLM 和第二個模型尚未實作；不要將規劃中的功能寫成已完成。Go 閘道 v1 已實作，但仍須依實際驗收證據描述其能力，不代表已部署或通過真實模型整合。

## 重要目錄與文件

```text
checkpoints/                         # 本機模型權重與 Hugging Face 快取
docs/                                # 專案文件
scripts/download.py                  # 通用 Hugging Face 模型下載工具
scripts/README.md                    # 下載工具使用說明
src/gateway/                         # Go API 閘道與 route/stream/upstream/videoproxy 模組
src/inference/                       # 推論模型登錄、啟動器與 API
src/inference/model_registry.yaml    # 模型別名與設定檔登錄
src/inference/mage_vl_4b/            # Mage-VL-4B Transformers 服務
tests/                               # 測試目錄
```

開始修改前先閱讀根目錄 `README.md`，並依任務閱讀 `src/inference/README.md`、`src/gateway/README.md` 或 `scripts/README.md`。以實際工作樹內容為準；若文件和程式碼不一致，先查明差異，再只更新任務相關的內容。

## 架構邊界

- 閘道層負責客戶端認證與 API Key、租戶識別、請求政策、限流、路由、可觀測性，以及視訊請求和 SSE 回應的轉送。
- 推論層負責模型權重與生命週期、影像／視訊理解、生成及 GPU 資源管理。不要把模型推論或 GPU 管理移入閘道。
- MediaMTX 是即時推流時使用的 sidecar，不是模型服務或 API 閘道。圖片與上傳視訊檔案流程不應要求 MediaMTX。
- 推論 API 預設為 `127.0.0.1:8001`；Gateway 預設監聽 `127.0.0.1:8000`。實際端點以目前設定檔為準，不要在程式中另行硬編碼。
- `/v1/chat/completions` 支援文字／圖片輸入與 SSE 文字輸出。視訊檔案理解及即時視訊輸入依各自 API 逐段／單次回應；「流式輸出」指模型文字 token 的 SSE 輸出，不代表視訊輸入必須逐 token 或逐影格回傳。

## 設定與權重

- 模型及服務設定優先從 YAML 載入。`model_registry.yaml` 登錄模型別名、推論／網路／媒體設定路徑及服務入口。
- 模型推論參數放在模型的 `configs/inference.yml`；API 綁定位址及串流端點放在 `configs/network.yml`；MediaMTX 設定放在 `configs/mediamtx.yml`。新增或調整可調參數時，優先加入合適的 YAML，不要把實驗超參數散落在 Python 或 shell 腳本。
- API 啟動時會依固定 Hugging Face revision 載入模型；缺少權重時可自動下載至根目錄 `checkpoints/`。通用下載工具則使用 `scripts/download.py`，並要求明確指定 `--repo-id`、`--revision` 和 `--local-dir`。
- `checkpoints/` 內的大型權重及快取由 `.gitignore` 忽略；`checkpoints/val.md` 和 `checkpoints/val.xlsx` 必須保留為可追蹤文件。不要刪除、搬移或覆蓋這兩個工作流文件。
- 不要在 Git 中加入模型權重、快取、私密金鑰或 API token。認證資料由環境變數或安全設定提供。

## 修改與驗證方式

- 先檢查相關程式、設定、文件和現有工作樹修改，再進行最小範圍的變更。保留使用者已有的未提交修改，不要用整檔覆寫方式抹除它們。
- 不要自行下載大型模型、啟動 GPU 推論服務、修改遠端伺服器或推送遠端倉庫；只有在使用者明確要求時才執行。
- 不要自行提交或推送 Git 變更，除非使用者明確要求。不要未經要求建立新分支。
- 除非使用者明確要求測試／驗證，或任務明確要求交付驗證結果，否則不要新增或執行測試。若未執行，清楚說明；不得宣稱未做的檢查已通過。
- 執行任何刪除、搬移、清理或覆寫前，先核對目標路徑和影響範圍；涉及模型權重或工作流資料時格外小心。
- 完成後摘要實際修改、相關檔案、驗證方式與尚未處理的事項。不要把計畫描述成已實作。

## 語言與文件

- 與使用者溝通使用簡體中文。
- 專案文件、程式註解和面向開發者的說明使用繁體中文；程式識別字、命令、API 欄位及上游專有名稱維持原文。
- 文件中的啟動順序、連接埠、端點和設定路徑必須與實際程式及 YAML 一致。
