package main

import (
	"context"
	"flag"
	"fmt"
	"log/slog"
	"net"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"gitee.com/milo-ally-intern/LYGCloudModelAPI/src/gateway"
)

func main() {
	configPath := flag.String("config", "src/gateway/configs/gateway.yml", "Gateway YAML 設定路徑")
	flag.Parse()
	cfg, err := gateway.LoadConfig(*configPath)
	if err != nil {
		fatal("載入 Gateway 設定失敗", err)
	}
	apiKeyPath := strings.TrimSpace(os.Getenv(cfg.Auth.APIKeysFileEnv))
	if apiKeyPath == "" {
		fatal(fmt.Sprintf("環境變數 %s 未設定", cfg.Auth.APIKeysFileEnv), nil)
	}
	apiKeys, err := gateway.LoadAPIKeys(apiKeyPath)
	if err != nil {
		fatal("載入 API Key 檔案失敗", err)
	}
	hmacKeys, err := gateway.ParseHMACKeys(os.Getenv(cfg.Session.HMACKeysEnv))
	if err != nil {
		fatal("載入 session HMAC 金鑰失敗", err)
	}
	logger := slog.New(slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{Level: parseLevel(cfg.Observability.LogLevel)}))
	handler, err := gateway.NewHandler(cfg, gateway.Options{APIKeys: apiKeys, HMACKeys: hmacKeys, Logger: logger})
	if err != nil {
		fatal("建立 Gateway 失敗", err)
	}
	server := &http.Server{
		Addr:              net.JoinHostPort(cfg.Listener.Host, fmt.Sprint(cfg.Listener.Port)),
		Handler:           handler,
		ReadHeaderTimeout: time.Duration(cfg.Timeouts.ReadHeaderSeconds) * time.Second,
		IdleTimeout:       time.Duration(cfg.Timeouts.IdleConnectionSeconds) * time.Second,
		MaxHeaderBytes:    cfg.Limits.MaxHeaderBytes,
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	go func() {
		<-ctx.Done()
		shutdownCtx, cancel := context.WithTimeout(context.Background(), time.Duration(cfg.Timeouts.ShutdownGraceSeconds)*time.Second)
		defer cancel()
		if err := server.Shutdown(shutdownCtx); err != nil {
			logger.Error("Gateway 優雅停止失敗", "error", err)
		}
	}()
	logger.Info("Gateway 已啟動", "address", server.Addr, "upstream", cfg.Upstream.BaseURL)
	if err := server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
		fatal("Gateway 啟動失敗", err)
	}
}

func parseLevel(value string) slog.Level {
	switch strings.ToLower(strings.TrimSpace(value)) {
	case "debug":
		return slog.LevelDebug
	case "warn", "warning":
		return slog.LevelWarn
	case "error":
		return slog.LevelError
	default:
		return slog.LevelInfo
	}
}

func fatal(message string, err error) {
	if err != nil {
		fmt.Fprintf(os.Stderr, "%s：%v\n", message, err)
	} else {
		fmt.Fprintln(os.Stderr, message)
	}
	os.Exit(1)
}
