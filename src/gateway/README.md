# API 閘道骨架

此目錄預留給 Go API 閘道。本階段僅建立套件目錄，尚未加入 Go 原始碼。

閘道預計負責外部 API 身分驗證、請求政策、路由、可觀測性，以及視訊上傳與 SSE 回應的透明轉送。GPU 模型載入與推論仍由 `src/inference/` 負責。

## 目前目錄

```text
src/gateway/
├── cmd/
│   └── gateway/       # 閘道執行程式入口
└── internal/
    ├── auth/          # API Key 與客戶端身分驗證
    ├── config/        # 閘道設定載入
    ├── middleware/    # 共用 HTTP 中介處理
    ├── route/         # API 路由與處理器
    ├── stream/        # SSE 等串流回應轉送
    ├── upstream/      # 下游模型服務呼叫
    └── videoproxy/    # 視訊上傳與請求轉送
```

以上目錄目前以 `.gitkeep` 保留，尚未包含 Go 實作。新增程式時，請沿用這份目錄命名與職責劃分。
