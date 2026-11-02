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
			rewritten, hasToken := rewriteSSEModel(item.data, alias)
			if !firstTokenRecorded && hasToken {
				forwarder.metrics.Observe("sse_time_to_first_token_seconds", "/v1/chat/completions", time.Since(requestStarted).Seconds())
				firstTokenRecorded = true
			}
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

func rewriteSSEModel(event []byte, alias string) ([]byte, bool) {
	var data bytes.Buffer
	dataCount := 0
	for start := 0; start <= len(event); {
		end := len(event)
		next := len(event) + 1
		if offset := bytes.IndexByte(event[start:], '\n'); offset >= 0 {
			end = start + offset
			next = end + 1
		}
		line := bytes.TrimSuffix(event[start:end], []byte{'\r'})
		if bytes.HasPrefix(line, []byte("data:")) {
			value := bytes.TrimPrefix(line, []byte("data:"))
			if len(value) > 0 && value[0] == ' ' {
				value = value[1:]
			}
			if dataCount > 0 {
				data.WriteByte('\n')
			}
			_, _ = data.Write(value)
			dataCount++
		}
		start = next
	}
	if dataCount == 0 {
		return event, false
	}
	joined := data.Bytes()
	if bytes.Equal(joined, []byte("[DONE]")) {
		return event, false
	}
	var payload map[string]json.RawMessage
	if json.Unmarshal([]byte(joined), &payload) != nil || payload == nil {
		return event, false
	}
	model, err := json.Marshal(alias)
	if err != nil {
		return event, false
	}
	payload["model"] = model
	encoded, err := json.Marshal(payload)
	if err != nil {
		return event, false
	}
	hasToken := hasChatToken(payload["choices"])
	var out bytes.Buffer
	dataWritten := false
	for start := 0; start <= len(event); {
		end := len(event)
		next := len(event) + 1
		if offset := bytes.IndexByte(event[start:], '\n'); offset >= 0 {
			end = start + offset
			next = end + 1
		}
		line := bytes.TrimSuffix(event[start:end], []byte{'\r'})
		if len(line) == 0 {
			start = next
			continue
		}
		if bytes.HasPrefix(line, []byte("data:")) {
			if !dataWritten {
				out.WriteString("data: ")
				out.Write(encoded)
				out.WriteByte('\n')
				dataWritten = true
			}
			start = next
			continue
		}
		out.Write(line)
		out.WriteByte('\n')
		start = next
	}
	out.WriteByte('\n')
	return out.Bytes(), hasToken
}

func hasChatToken(rawChoices json.RawMessage) bool {
	var choices []struct {
		Delta struct {
			Content   json.RawMessage `json:"content"`
			ToolCalls json.RawMessage `json:"tool_calls"`
		} `json:"delta"`
	}
	if len(rawChoices) == 0 || json.Unmarshal(rawChoices, &choices) != nil {
		return false
	}
	for _, choice := range choices {
		var content string
		if json.Unmarshal(choice.Delta.Content, &content) == nil && content != "" {
			return true
		}
		var calls []json.RawMessage
		if json.Unmarshal(choice.Delta.ToolCalls, &calls) == nil && len(calls) > 0 {
			return true
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
