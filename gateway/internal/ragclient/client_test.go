package ragclient

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"
)

func TestRetriesServerErrorsThenSucceeds(t *testing.T) {
	var calls atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("X-Request-ID") != "req-1" {
			t.Errorf("request id not propagated")
		}
		if calls.Add(1) < 3 {
			w.WriteHeader(http.StatusBadGateway)
			return
		}
		w.Write([]byte(`{"answer":"ok"}`))
	}))
	defer srv.Close()

	c := New(srv.URL, time.Second, 2)
	c.backoff = time.Millisecond
	body, err := c.Query(WithRequestID(context.Background(), "req-1"), QueryRequest{Question: "q"})
	if err != nil {
		t.Fatal(err)
	}
	if string(body) != `{"answer":"ok"}` || calls.Load() != 3 {
		t.Fatalf("body=%s calls=%d", body, calls.Load())
	}
}

func TestDoesNotRetryClientErrors(t *testing.T) {
	var calls atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		http.Error(w, "bad profile", http.StatusBadRequest)
	}))
	defer srv.Close()

	c := New(srv.URL, time.Second, 3)
	_, err := c.Query(context.Background(), QueryRequest{Question: "q"})
	var se *StatusError
	if !errors.As(err, &se) || se.Code != http.StatusBadRequest {
		t.Fatalf("want 400 StatusError, got %v", err)
	}
	if calls.Load() != 1 {
		t.Fatalf("4xx should not be retried, got %d calls", calls.Load())
	}
}
