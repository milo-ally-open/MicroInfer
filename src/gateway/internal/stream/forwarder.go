package stream

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net"
	"net/http"
	"strings"
	"time"
)

type Metrics interface {
	Observe(name, route string, seconds float64)
	AddUpstreamError(kind string)
}

type ErrorWriter func(http.ResponseWriter, int, string, string, string)

type Config struct {
	EventBytes  int64
	BufferBytes int
	IdleTimeout time.Duration
}

type Forwarder struct {
	config     Config
	metrics    Metrics
	logger     *slog.Logger
	writeError ErrorWriter
}

func New(config Config, metrics Metrics, logger *slog.Logger, writeError ErrorWriter) *Forwarder {
	return &Forwarder{config: config, metrics: metrics, logger: logger, writeError: writeError}
}

func (forwarder *Forwarder) Chat(w http.ResponseWriter, r *http.Request, ctx context.Context, response *http.Response, alias, requestID string, requestStarted time.Time) {
	if response.Header.Get("Content-Type") != "" {
		w.Header().Set("Content-Type", response.Header.Get("Content-Type"))
	} else {
		w.Header().Set("Content-Type", "text/event-stream; charset=utf-8")
	}
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("X-Accel-Buffering", "no")
	copySelectedHeaders(w, response.Header, "Cache-Control", "X-Accel-Buffering", "X-Request-ID")
	started := time.Now()
	defer func() {
		forwarder.metrics.Observe("sse_stream_duration_seconds", "/v1/chat/completions", time.Since(started).Seconds())
	}()

	reads := make(chan eventRead, 1)
	go func() {
		reader := bufio.NewReaderSize(response.Body, forwarder.config.BufferBytes)
		for {
			event, err := readSSEEvent(reader, forwarder.config.EventBytes)
			if len(event) > 0 {
				select {
				case reads <- eventRead{data: event}:
				case <-ctx.Done():
					return
				case <-r.Context().Done():
					return
				}
			}
			if err != nil {
				select {
				case reads <- eventRead{err: err}:
				case <-ctx.Done():
				case <-r.Context().Done():
				}
				return
			}
		}
	}()

	committed := false
	firstTokenRecorded := false
	for {
		idle := time.NewTimer(forwarder.config.IdleTimeout)
		select {
		case item := <-reads:
			idle.Stop()
			if item.err != nil {
				if item.err != io.EOF {
					forwarder.logger.Error("chat SSE upstream stream ended", "request_id", requestID, "error_class", classifyError(item.err))
				}
				if !committed {
					forwarder.writeError(w, http.StatusBadGateway, "upstream_error", "upstream_stream_failed", "The inference service stream failed before producing an event.")
				}
				return
			}
			if !committed {
				w.WriteHeader(response.StatusCode)
				committed = true
			}
			if !firstTokenRecorded && hasChatToken(item.data) {
				forwarder.metrics.Observe("sse_time_to_first_token_seconds", "/v1/chat/completions", time.Since(requestStarted).Seconds())
				firstTokenRecorded = true
			}
			rewritten := rewriteSSEModel(item.data, alias)
			if _, err := w.Write(rewritten); err != nil {
				return
			}
			if flusher, ok := w.(http.Flusher); ok {
				flusher.Flush()
			}
		case <-idle.C:
			forwarder.logger.Error("chat SSE idle timeout", "request_id", requestID)
			forwarder.metrics.AddUpstreamError("stream_idle_timeout")
			if !committed {
				forwarder.writeError(w, http.StatusGatewayTimeout, "upstream_error", "upstream_timeout", "The inference service stream timed out.")
			}
			return
		case <-ctx.Done():
			idle.Stop()
			if ctx.Err() == context.DeadlineExceeded {
				forwarder.metrics.AddUpstreamError("stream_timeout")
				if !committed {
					forwarder.writeError(w, http.StatusGatewayTimeout, "upstream_error", "upstream_timeout", "The inference service stream timed out.")
				}
			}
			return
		case <-r.Context().Done():
			idle.Stop()
			return
		}
	}
}

func (forwarder *Forwarder) Copy(w http.ResponseWriter, r *http.Request, ctx context.Context, body io.Reader, route, requestID string) {
	started := time.Now()
	defer func() { forwarder.metrics.Observe("sse_stream_duration_seconds", route, time.Since(started).Seconds()) }()
	first := true
	type readResult struct {
		data []byte
		err  error
	}
	chunks := make(chan readResult, 1)
	go func() {
		buffer := make([]byte, forwarder.config.BufferBytes)
		for {
			count, err := body.Read(buffer)
			if count > 0 {
				chunk := append([]byte(nil), buffer[:count]...)
				select {
				case chunks <- readResult{data: chunk}:
				case <-ctx.Done():
					return
				case <-r.Context().Done():
					return
				}
			}
			if err != nil {
				select {
				case chunks <- readResult{err: err}:
				case <-ctx.Done():
				case <-r.Context().Done():
				}
				return
			}
		}
	}()

	idle := time.NewTimer(forwarder.config.IdleTimeout)
	defer idle.Stop()
	for {
		select {
		case result := <-chunks:
			if !idle.Stop() {
				select {
				case <-idle.C:
				default:
				}
			}
			idle.Reset(forwarder.config.IdleTimeout)
			if len(result.data) > 0 {
				if first {
					forwarder.metrics.Observe("sse_time_to_first_event_seconds", route, time.Since(started).Seconds())
					first = false
				}
				if _, writeErr := w.Write(result.data); writeErr != nil {
					return
				}
				if flusher, ok := w.(http.Flusher); ok {
					flusher.Flush()
				}
			}
			if result.err != nil {
				if result.err != io.EOF {
					forwarder.logger.Error("live SSE upstream stream ended", "request_id", requestID, "error_class", classifyError(result.err))
				}
				return
			}
		case <-idle.C:
			forwarder.logger.Error("live SSE idle timeout", "request_id", requestID)
			forwarder.metrics.AddUpstreamError("stream_idle_timeout")
			return
		case <-ctx.Done():
			if ctx.Err() == context.DeadlineExceeded {
				forwarder.metrics.AddUpstreamError("stream_timeout")
			}
			return
		case <-r.Context().Done():
			return
		}
	}
}

type eventRead struct {
	data []byte
	err  error
}

func readSSEEvent(reader *bufio.Reader, maxBytes int64) ([]byte, error) {
	var event bytes.Buffer
	for {
		fragment, err := reader.ReadSlice('\n')
		if int64(event.Len()+len(fragment)) > maxBytes {
			return nil, fmt.Errorf("SSE event exceeds configured limit")
		}
		_, _ = event.Write(fragment)
		if err == bufio.ErrBufferFull {
			continue
		}
		if err != nil {
			if err == io.EOF && event.Len() > 0 {
				return event.Bytes(), io.EOF
			}
			return nil, err
		}
		if bytes.Equal(fragment, []byte("\n")) || bytes.Equal(fragment, []byte("\r\n")) {
			return event.Bytes(), nil
		}
	}
}

func rewriteSSEModel(event []byte, alias string) []byte {
	lines := bytes.Split(event, []byte("\n"))
	dataValues := make([]string, 0, 1)
	for _, rawLine := range lines {
		trimmed := strings.TrimSuffix(string(rawLine), "\r")
		if strings.HasPrefix(trimmed, "data:") {
			value := strings.TrimPrefix(trimmed, "data:")
			value = strings.TrimPrefix(value, " ")
			dataValues = append(dataValues, value)
		}
	}
	if len(dataValues) == 0 {
		return event
	}
	joined := strings.Join(dataValues, "\n")
	if joined == "[DONE]" {
		return event
	}
	var payload map[string]any
	if json.Unmarshal([]byte(joined), &payload) != nil || payload == nil {
		return event
	}
	payload["model"] = alias
	encoded, err := json.Marshal(payload)
	if err != nil {
		return event
	}
	var out bytes.Buffer
	dataWritten := false
	for _, rawLine := range lines {
		line := strings.TrimSuffix(string(rawLine), "\r")
		if line == "" {
			continue
		}
		if strings.HasPrefix(line, "data:") {
			if !dataWritten {
				out.WriteString("data: ")
				out.Write(encoded)
				out.WriteByte('\n')
				dataWritten = true
			}
			continue
		}
		out.WriteString(line)
		out.WriteByte('\n')
	}
	out.WriteByte('\n')
	return out.Bytes()
}

func hasChatToken(event []byte) bool {
	for _, line := range bytes.Split(event, []byte("\n")) {
		line = bytes.TrimSuffix(line, []byte("\r"))
		if !bytes.HasPrefix(line, []byte("data:")) {
			continue
		}
		data := bytes.TrimSpace(bytes.TrimPrefix(line, []byte("data:")))
		if len(data) == 0 || bytes.Equal(data, []byte("[DONE]")) {
			continue
		}
		var payload struct {
			Choices []struct {
				Delta struct {
					Content any `json:"content"`
				} `json:"delta"`
			} `json:"choices"`
		}
		if json.Unmarshal(data, &payload) != nil {
			continue
		}
		for _, choice := range payload.Choices {
			if content, ok := choice.Delta.Content.(string); ok && content != "" {
				return true
			}
		}
	}
	return false
}

func copySelectedHeaders(w http.ResponseWriter, source http.Header, names ...string) {
	for _, name := range names {
		if values := source.Values(name); len(values) > 0 {
			w.Header()[http.CanonicalHeaderKey(name)] = append([]string(nil), values...)
		}
	}
}

func classifyError(err error) string {
	if errors.Is(err, context.DeadlineExceeded) {
		return "timeout"
	}
	var networkError net.Error
	if errors.As(err, &networkError) && networkError.Timeout() {
		return "timeout"
	}
	return "stream_error"
}
