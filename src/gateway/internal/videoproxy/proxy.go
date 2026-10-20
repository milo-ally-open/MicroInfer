package videoproxy

import (
	"context"
	"io"
	"net/http"
	"time"

	"gitee.com/milo-ally-intern/LYGCloudModelAPI/src/gateway/internal/upstream"
)

type Metrics interface {
	Observe(name, route string, seconds float64)
}

type RequestError struct {
	Status  int
	Code    string
	Message string
}

func (requestError *RequestError) Error() string { return requestError.Message }

type Proxy struct {
	upstream        *upstream.Client
	baseURL         string
	maxRequestSize  int64
	responseTimeout time.Duration
	metrics         Metrics
}

func New(upstreamClient *upstream.Client, baseURL string, maxRequestSize int64, responseTimeout time.Duration, metrics Metrics) *Proxy {
	return &Proxy{upstream: upstreamClient, baseURL: baseURL, maxRequestSize: maxRequestSize, responseTimeout: responseTimeout, metrics: metrics}
}

func (proxy *Proxy) Forward(r *http.Request, requestID string) (*http.Response, context.Context, context.CancelFunc, error) {
	if err := ValidateContentLength(r.ContentLength, proxy.maxRequestSize); err != nil {
		return nil, nil, nil, err
	}
	body := &trackedBody{ReadCloser: r.Body, expected: r.ContentLength, started: time.Now(), metrics: proxy.metrics}
	response, ctx, cancel, _, err := proxy.upstream.Forward(r, requestID, http.MethodPost, "/v1/videos/understand", proxy.baseURL, body, r.ContentLength, r.Header.Get("Content-Type"), proxy.responseTimeout)
	return response, ctx, cancel, err
}

func ValidateContentLength(contentLength, limit int64) error {
	if contentLength < -1 {
		return &RequestError{Status: http.StatusBadRequest, Code: "invalid_content_length", Message: "Content-Length must be a non-negative integer."}
	}
	if contentLength == -1 {
		return &RequestError{Status: http.StatusLengthRequired, Code: "content_length_required", Message: "Content-Length is required for this endpoint."}
	}
	if contentLength > limit {
		return &RequestError{Status: http.StatusRequestEntityTooLarge, Code: "request_too_large", Message: "The request body exceeds the configured size limit."}
	}
	return nil
}

type trackedBody struct {
	io.ReadCloser
	expected int64
	read     int64
	started  time.Time
	done     bool
	metrics  Metrics
}

func (body *trackedBody) Read(data []byte) (int, error) {
	n, err := body.ReadCloser.Read(data)
	body.read += int64(n)
	if !body.done && ((body.expected >= 0 && body.read >= body.expected) || err == io.EOF) {
		body.done = true
		body.metrics.Observe("video_upload_duration_seconds", "/v1/videos/understand", time.Since(body.started).Seconds())
	}
	return n, err
}
