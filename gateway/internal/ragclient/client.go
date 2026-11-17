// Package ragclient calls the Python RAG service with timeouts and retries.
package ragclient

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"time"
)

type QueryRequest struct {
	Question string `json:"question"`
	Profile  string `json:"profile,omitempty"`
}

// StatusError is a non-2xx response from the RAG service.
type StatusError struct {
	Code int
	Body string
}

func (e *StatusError) Error() string { return fmt.Sprintf("rag service returned %d: %s", e.Code, e.Body) }

type Client struct {
	baseURL    string
	http       *http.Client
	maxRetries int
	backoff    time.Duration
}

func New(baseURL string, timeout time.Duration, maxRetries int) *Client {
	return &Client{
		baseURL:    baseURL,
		http:       &http.Client{Timeout: timeout},
		maxRetries: maxRetries,
		backoff:    200 * time.Millisecond,
	}
}

// Query forwards a question and returns the raw JSON answer. Connection errors
// and 5xx responses are retried with exponential backoff; 4xx are not.
func (c *Client) Query(ctx context.Context, req QueryRequest) ([]byte, error) {
	payload, err := json.Marshal(req)
	if err != nil {
		return nil, err
	}
	var lastErr error
	for attempt := 0; attempt <= c.maxRetries; attempt++ {
		if attempt > 0 {
			select {
			case <-time.After(c.backoff << (attempt - 1)):
			case <-ctx.Done():
				return nil, ctx.Err()
			}
		}
		body, err := c.do(ctx, payload)
		if err == nil {
			return body, nil
		}
		lastErr = err
		var se *StatusError
		if errors.As(err, &se) && se.Code < 500 {
			return nil, err
		}
		if ctx.Err() != nil {
			return nil, ctx.Err()
		}
	}
	return nil, lastErr
}

func (c *Client) do(ctx context.Context, payload []byte) ([]byte, error) {
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+"/query", bytes.NewReader(payload))
	if err != nil {
		return nil, err
	}
	httpReq.Header.Set("Content-Type", "application/json")
	if id, ok := ctx.Value(requestIDKey{}).(string); ok {
		httpReq.Header.Set("X-Request-ID", id)
	}
	resp, err := c.http.Do(httpReq)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if err != nil {
		return nil, err
	}
	if resp.StatusCode >= 300 {
		return nil, &StatusError{Code: resp.StatusCode, Body: string(body)}
	}
	return body, nil
}

// Ready checks the RAG service readiness endpoint.
func (c *Client) Ready(ctx context.Context) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.baseURL+"/readyz", nil)
	if err != nil {
		return err
	}
	resp, err := c.http.Do(req)
	if err != nil {
		return err
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return &StatusError{Code: resp.StatusCode}
	}
	return nil
}

type requestIDKey struct{}

// WithRequestID attaches a request ID that is propagated to the RAG service.
func WithRequestID(ctx context.Context, id string) context.Context {
	return context.WithValue(ctx, requestIDKey{}, id)
}
