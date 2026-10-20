package upstream

import (
	"context"
	"io"
	"net"
	"net/http"
	"strings"
	"time"

	"gitee.com/milo-ally-intern/LYGCloudModelAPI/src/gateway/internal/config"
)

// Client 管理連線推論服務時共用的 HTTP transport。
type Client struct {
	httpClient *http.Client
}

func NewClient(cfg config.Config) *Client {
	transport := &http.Transport{
		Proxy:                 http.ProxyFromEnvironment,
		DialContext:           (&net.Dialer{Timeout: time.Duration(cfg.Timeouts.ConnectSeconds) * time.Second, KeepAlive: time.Duration(cfg.Upstream.TCPKeepAliveSeconds) * time.Second}).DialContext,
		TLSHandshakeTimeout:   time.Duration(cfg.Timeouts.ConnectSeconds) * time.Second,
		ResponseHeaderTimeout: time.Duration(cfg.Timeouts.VideoRequestSeconds) * time.Second,
		ExpectContinueTimeout: time.Duration(cfg.Upstream.ExpectContinueSeconds) * time.Second,
		MaxIdleConns:          cfg.Upstream.MaxIdleConnections,
		MaxIdleConnsPerHost:   cfg.Upstream.MaxIdleConnectionsPerHost,
		IdleConnTimeout:       time.Duration(cfg.Upstream.IdleConnectionSeconds) * time.Second,
	}
	return &Client{httpClient: &http.Client{Transport: transport}}
}

func (client *Client) Forward(incoming *http.Request, requestID, method, path, baseURL string, body io.Reader, contentLength int64, contentType string, timeout time.Duration) (*http.Response, context.Context, context.CancelFunc, time.Duration, error) {
	ctx, cancel := context.WithTimeout(incoming.Context(), timeout)
	request, err := http.NewRequestWithContext(ctx, method, URL(baseURL, path), body)
	if err != nil {
		cancel()
		return nil, nil, nil, 0, err
	}
	if body != nil {
		request.ContentLength = contentLength
	}
	if contentType != "" {
		request.Header.Set("Content-Type", contentType)
	}
	if accept := incoming.Header.Get("Accept"); accept != "" {
		request.Header.Set("Accept", accept)
	}
	if lastID := incoming.Header.Get("Last-Event-ID"); lastID != "" {
		request.Header.Set("Last-Event-ID", lastID)
	}
	if requestID != "" {
		request.Header.Set("X-Request-ID", requestID)
	}
	started := time.Now()
	response, err := client.httpClient.Do(request)
	elapsed := time.Since(started)
	if err != nil {
		cancel()
		return nil, nil, nil, elapsed, err
	}
	return response, ctx, cancel, elapsed, nil
}

func URL(baseURL, path string) string {
	return strings.TrimRight(baseURL, "/") + path
}

func BaseURL(cfg config.Config, route config.ModelRoute) string {
	if route.BaseURL != "" {
		return route.BaseURL
	}
	return cfg.Upstream.BaseURL
}
