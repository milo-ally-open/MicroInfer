# 提案：推論服務現況規格基線

> 狀態：現況基線／待評審  
> 路徑：跨邊界高風險  
> 負責人：推論服務維護者  
> 上游依據：[專案規範](../../../AGENTS.md)、[專案 README](../../../README.md)、[推論服務 README](../../../src/inference/README.md)、[模型登錄表](../../../src/inference/model_registry.yaml)

## 背景與問題

專案已有 Mage-VL-4B Transformers 推論服務、模型登錄表與啟動器；初次反向盤點時尚無推論規格組或推論測試。現行行為散落在 FastAPI 路由、模型服務、啟動器、三份 YAML 與 README 中，後續變更需要可追溯的需求、驗收條件及測試任務。後續已新增來源碼／設定的靜態契約檢查，但尚無推論 API runtime 測試。

本規格組從目前原始碼與設定反向整理現況，不代表新增功能，也不把未實作的引擎或模型列為現有能力。驗收條件描述目前來源碼可觀察的行為；除已列明的結構校驗外，本次沒有啟動模型或執行 API 測試。

## 目標與成功觀察

- 將目前模型選擇、設定載入、權重生命週期、API 輸入／輸出、圖片／視訊處理、即時串流與 MediaMTX 回呼整理為推論領域基線。
- 以穩定的 REQ、AC、T 編號連結程式碼責任與後續 tests/inference 驗證任務。
- 清楚區分已實作行為、文件與程式不一致、尚未執行的驗證及需產品決定的契約。
- 維持推論服務與 Go Gateway 分離；Gateway 不承擔模型載入、GPU 推論或推論設定。

## 範圍

### 本次包含

- src/inference/launcher.py、model_registry.yaml、Mage-VL-4B FastAPI 服務及 inference.yml、network.yml、mediamtx.yml 的現況。
- 模型與處理器載入、權重快取、健康狀態、HTTP 請求限制、聊天與生成、SSE、上傳視訊、即時視訊及關閉流程。
- MediaMTX sidecar 的用途和 HTTP 授權回呼邊界。
- 後續直接推論測試的責任分層及現況缺口。

### 原始規格盤點階段不包含／保持不變

- 原始規格盤點階段不修改推論原始碼、YAML、依賴鎖定檔、README、Gateway、MediaMTX 或測試程式；後續為將 live 本文上限放入 YAML 而新增設定值及測試，詳見下方後續實作紀錄。
- 不下載模型、不啟動 GPU 推論服務、不連接即時媒體源、不改動遠端主機。
- 不把 SGLang、vLLM、第二個模型或多後端推論寫成已實作能力。
- 不執行模型資料集精度評測；本規格不引入 mAP、Video-MME 等品質分數。
- 不覆寫、移動或刪除 checkpoints/val.md 與 checkpoints/val.xlsx。

## 現況依據

### 已由原始碼與設定確認

- src/inference/model_registry.yaml 目前登錄 mage-vl-4b，預設後端為 transformers，服務入口為 src/inference/mage_vl_4b/main.py。
- 專案別名 mage-vl-4b 與 Hugging Face repo_id microsoft/Mage-VL 不同。推論 API 的模型清單回傳 repo_id；聊天請求的 model 可省略，若提供則必須等於該 repo_id。
- launcher.py 提供 list 與 start 命令；start 可覆寫模型別名、推論／網路／媒體 YAML、監聽 host／port 及 CUDA_VISIBLE_DEVICES。
- 推論 API 預設監聽 127.0.0.1:8001；模型快取位於 checkpoints/，缺少固定 revision 時依 runtime.local_files_only 設定下載。
- 模型載入要求 CUDA。模型及處理器使用同一固定 revision；目前設定 dtype 為 bfloat16、device_map 為 auto、量化 method 為 none，torch.compile 關閉。
- API 有聊天 SSE、視訊上傳及即時工作階段；即時推流才需要 MediaMTX。
- 初次反向盤點時，tests/inference、tests/gateway、tests/end2end 都只有 .gitkeep，沒有已追蹤的測試檔案；後續加入的靜態契約檢查見 tasks.md。

### 文件與程式差異

- README 與專案規範描述聊天 API 支援文字／圖片輸入；現行 _normalize_messages 實際要求每個請求至少包含一個 image_url，純文字請求回 HTTP 400。此規格先記錄來源碼行為，純文字是否屬於預期公開能力列為待確認，不能由本文替代產品決定。
- inference.yml 的 hardware.expected_gpu、runtime.conda_env、runtime.python_venv、runtime.startup_timeout_seconds、runtime.startup_poll_interval_seconds、runtime.startup_log_file、runtime.startup_result_file 目前沒有被 launcher.py 或 service.py 使用；它們不構成目前的執行時保證。
- 圖片 URL 只限制 http／https，程式會由推論主機直接下載，沒有來源網域白名單；對外部署前需處理 SSRF 風險。

## 提案與取捨

| 方案 | 收益 | 成本／風險 |
|---|---|---|
| 以現行程式與設定作為基線，對文件差異明確標記待確認 | 規格忠實反映實際行為，避免把規劃寫成已完成 | 純文字聊天契約仍需由專案負責人決定 |
| 直接把 README 的文字／圖片描述當成正式契約 | 文件表面一致 | 與目前程式碼衝突，會使驗收錯誤 |
| 直接改程式以接受純文字輸入 | 可能符合文件原意 | 超出本次文件範圍，需另行授權並驗收 |

建議採用第一列。現況基線以程式為證據；待確認的外部契約不暗中改寫，也不藉此修改程式。

## 影響面、依賴與風險

- 後續變更責任範圍：模型登錄與啟動器、Transforms 推論 API、模型／網路／MediaMTX YAML、推論測試。
- 模型權重與快取依賴 Hugging Face 固定 revision；權重與快取由 checkpoints/ 管理並排除於 Git。
- GPU、CUDA、Transformers、PyTorch、影片解碼器及 codec-video-prep 等環境因素會影響執行；純結構測試不能證明真機載入或性能。
- trust_remote_code 啟用上游模型程式碼；版本固定有助重現，仍需在升級 revision 前檢視來源。
- 即時工作階段、事件佇列與限流式生成名額是程序內狀態，服務重啟會停止工作階段並清除狀態。
- 安全風險包含任意 HTTP(S) image_url 抓取、健康回應中的模型／GPU 資訊，以及異常回應中可能包含底層錯誤細節；目前服務預設只綁定 loopback，仍需在對外暴露前由 Gateway 或網路政策保護。

## 未決問題

| 問題 | 影響 | 負責人／驗證方式 | 狀態 |
|---|---|---|---|
| 聊天 API 是否應接受完全沒有 image_url 的純文字請求？ | README 契約與程式行為不一致，影響公開請求驗收 | 專案負責人確認後更新 spec 與 README，必要時另立程式修正任務 | [待確認] |
| image_url 是否需限制主機、IP 範圍或下載來源？ | 任意 URL 由模型服務主機抓取，可能造成 SSRF | 安全設計決定後另建需求與測試 | [待確認] |
| 環境與硬體資訊欄位是否要由 launcher 真正執行驗證？ | 部分 YAML 欄位目前未被使用 | 維護者依部署需求決定是否清理或實作 | [待確認] |

## 變更記錄

| 日期 | 內容 | 決定／依據 |
|---|---|---|
| 2026-10-20 | 建立推論服務現況規格組 | 依目前工作樹原始碼與 YAML 反向整理；未執行模型或 API |

> 後續實作紀錄：為讓 live 本文上限遵循推論 YAML，`api.max_live_request_bytes` 已加入 `inference.yml`，FastAPI middleware 改讀該值；來源契約測試已加入 `tests/inference/`。此變更維持原本 65,536 bytes 預設值，未啟動推論服務。

