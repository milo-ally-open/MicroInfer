# 推論測試

本目錄目前提供不載入推論套件的靜態來源契約檢查：驗證模型別名與上游 ID、路由宣告及 YAML 請求限制。

```powershell
python -m unittest discover -s tests/inference -v
```

此測試不匯入 `service.py`，不會載入 CUDA、模型權重、MediaMTX 或影片解碼器。它只能檢查程式與設定中的靜態契約；模型載入、API 輸入、生成、SSE、視訊前處理與 live session 的 runtime 測試仍待後續補齊。執行 GPU smoke 前須有維護者提供的權重及可用 GPU，且不得藉由測試自動下載權重。
