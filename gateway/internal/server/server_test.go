package server

import (
	"context"
	"errors"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/medrag/gateway/internal/cache"
	"github.com/medrag/gateway/internal/ragclient"
	"github.com/medrag/gateway/internal/ratelimit"
)

type fakeRAG struct {
	calls int
	err   error
	last  ragclient.QueryRequest
}

func (f *fakeRAG) Query(_ context.Context, req ragclient.QueryRequest) ([]byte, error) {
	f.calls++
	f.last = req
	if f.err != nil {
		return nil, f.err
	}
	return []byte(`{"answer":"Metformin [1]"}`), nil
}

func (f *fakeRAG) Ready(context.Context) error { return f.err }

func newTestServer(rag RAG, limiter *ratelimit.Limiter) http.Handler {
	return New(rag, Options{
		MaxQuestionLen: 100,
		Limiter:        limiter,
		Cache:          cache.New(10, time.Minute),
		Logger:         slog.New(slog.NewTextHandler(io.Discard, nil)),
	}).Handler()
}

func post(h http.Handler, body string) *httptest.ResponseRecorder {
	req := httptest.NewRequest(http.MethodPost, "/v1/query", strings.NewReader(body))
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	return rec
}

func TestQueryForwardsAndCaches(t *testing.T) {
	rag := &fakeRAG{}
	h := newTestServer(rag, nil)

	first := post(h, `{"question":"  What is first-line for diabetes? "}`)
	if first.Code != http.StatusOK || first.Header().Get("X-Cache") != "MISS" {
		t.Fatalf("code=%d cache=%s", first.Code, first.Header().Get("X-Cache"))
	}
	if rag.last.Question != "What is first-line for diabetes?" {
		t.Fatalf("question not trimmed: %q", rag.last.Question)
	}
	if first.Header().Get("X-Request-ID") == "" {
		t.Fatal("missing request id")
	}

	second := post(h, `{"question":"what is FIRST-LINE  for diabetes?"}`)
	if second.Header().Get("X-Cache") != "HIT" || rag.calls != 1 {
		t.Fatalf("expected normalized cache hit; cache=%s calls=%d", second.Header().Get("X-Cache"), rag.calls)
	}
}

func TestValidation(t *testing.T) {
	h := newTestServer(&fakeRAG{}, nil)
	cases := map[string]string{
		"bad json":      `{"question":`,
		"too short":     `{"question":"hi"}`,
		"too long":      `{"question":"` + strings.Repeat("a", 101) + `"}`,
		"unknown field": `{"question":"valid question","x":1}`,
	}
	for name, body := range cases {
		if rec := post(h, body); rec.Code != http.StatusBadRequest {
			t.Errorf("%s: code=%d, want 400", name, rec.Code)
		}
	}
}

func TestRateLimit(t *testing.T) {
	h := newTestServer(&fakeRAG{}, ratelimit.New(0.001, 1))
	if rec := post(h, `{"question":"first question"}`); rec.Code != http.StatusOK {
		t.Fatalf("first request code=%d", rec.Code)
	}
	rec := post(h, `{"question":"second question"}`)
	if rec.Code != http.StatusTooManyRequests || rec.Header().Get("Retry-After") == "" {
		t.Fatalf("code=%d, want 429 with Retry-After", rec.Code)
	}
}

func TestUpstreamErrorMapping(t *testing.T) {
	cases := []struct {
		err  error
		want int
	}{
		{&ragclient.StatusError{Code: 400, Body: `{"detail":"bad profile"}`}, http.StatusBadRequest},
		{context.DeadlineExceeded, http.StatusGatewayTimeout},
		{errors.New("connection refused"), http.StatusBadGateway},
	}
	for _, tc := range cases {
		h := newTestServer(&fakeRAG{err: tc.err}, nil)
		if rec := post(h, `{"question":"valid question"}`); rec.Code != tc.want {
			t.Errorf("err=%v: code=%d, want %d", tc.err, rec.Code, tc.want)
		}
	}
}

func TestHealthReadyMetrics(t *testing.T) {
	h := newTestServer(&fakeRAG{}, nil)
	post(h, `{"question":"valid question"}`)
	for path, want := range map[string]int{"/healthz": 200, "/readyz": 200, "/metrics": 200} {
		rec := httptest.NewRecorder()
		h.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, path, nil))
		if rec.Code != want {
			t.Errorf("%s: code=%d", path, rec.Code)
		}
		if path == "/metrics" && !strings.Contains(rec.Body.String(), "gateway_requests_total") {
			t.Error("metrics missing gateway_requests_total")
		}
	}

	notReady := newTestServer(&fakeRAG{err: errors.New("down")}, nil)
	rec := httptest.NewRecorder()
	notReady.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/readyz", nil))
	if rec.Code != http.StatusServiceUnavailable {
		t.Errorf("readyz with down upstream: code=%d", rec.Code)
	}
}
