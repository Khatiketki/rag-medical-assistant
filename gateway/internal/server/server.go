// Package server is the public HTTP API in front of the RAG service.
package server

import (
	"context"
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"log/slog"
	"net"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"

	"github.com/medrag/gateway/internal/cache"
	"github.com/medrag/gateway/internal/ragclient"
	"github.com/medrag/gateway/internal/ratelimit"
)

// RAG is the subset of the RAG client the server depends on.
type RAG interface {
	Query(ctx context.Context, req ragclient.QueryRequest) ([]byte, error)
	Ready(ctx context.Context) error
}

type Options struct {
	MaxQuestionLen int
	Limiter        *ratelimit.Limiter
	Cache          *cache.Cache
	Registry       *prometheus.Registry
	Logger         *slog.Logger
}

type Server struct {
	rag     RAG
	opts    Options
	metrics *metrics
	log     *slog.Logger
}

func New(rag RAG, opts Options) *Server {
	if opts.Registry == nil {
		opts.Registry = prometheus.NewRegistry()
	}
	if opts.Logger == nil {
		opts.Logger = slog.Default()
	}
	return &Server{rag: rag, opts: opts, metrics: newMetrics(opts.Registry), log: opts.Logger}
}

func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.Handle("POST /v1/query", s.instrument("query", http.HandlerFunc(s.handleQuery)))
	mux.HandleFunc("GET /healthz", func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
	})
	mux.HandleFunc("GET /readyz", s.handleReady)
	mux.Handle("GET /metrics", promhttp.HandlerFor(s.opts.Registry, promhttp.HandlerOpts{}))
	return s.recoverer(s.requestID(mux))
}

type queryBody struct {
	Question string `json:"question"`
	Profile  string `json:"profile,omitempty"`
}

func (s *Server) handleQuery(w http.ResponseWriter, r *http.Request) {
	if s.opts.Limiter != nil && !s.opts.Limiter.Allow(clientIP(r)) {
		s.metrics.rateLimited.Inc()
		w.Header().Set("Retry-After", "1")
		writeError(w, http.StatusTooManyRequests, "rate limit exceeded")
		return
	}

	var body queryBody
	dec := json.NewDecoder(http.MaxBytesReader(w, r.Body, 16<<10))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&body); err != nil {
		writeError(w, http.StatusBadRequest, "invalid JSON body")
		return
	}
	body.Question = strings.TrimSpace(body.Question)
	switch {
	case len(body.Question) < 3:
		writeError(w, http.StatusBadRequest, "question must be at least 3 characters")
		return
	case len(body.Question) > s.opts.MaxQuestionLen:
		writeError(w, http.StatusBadRequest, "question exceeds "+strconv.Itoa(s.opts.MaxQuestionLen)+" characters")
		return
	}

	key := cacheKey(body)
	if s.opts.Cache != nil {
		if cached, ok := s.opts.Cache.Get(key); ok {
			s.metrics.cacheHits.Inc()
			w.Header().Set("X-Cache", "HIT")
			writeRaw(w, http.StatusOK, cached)
			return
		}
	}

	answer, err := s.rag.Query(r.Context(), ragclient.QueryRequest{Question: body.Question, Profile: body.Profile})
	if err != nil {
		var se *ragclient.StatusError
		switch {
		case errors.As(err, &se) && se.Code < 500:
			s.metrics.upstreamErr.WithLabelValues("client").Inc()
			writeRaw(w, se.Code, []byte(se.Body))
		case errors.Is(err, context.DeadlineExceeded):
			s.metrics.upstreamErr.WithLabelValues("timeout").Inc()
			writeError(w, http.StatusGatewayTimeout, "RAG service timed out")
		default:
			s.metrics.upstreamErr.WithLabelValues("unavailable").Inc()
			s.log.Error("rag query failed", "err", err, "request_id", requestIDFrom(r.Context()))
			writeError(w, http.StatusBadGateway, "RAG service unavailable")
		}
		return
	}
	if s.opts.Cache != nil {
		s.opts.Cache.Set(key, answer)
	}
	w.Header().Set("X-Cache", "MISS")
	writeRaw(w, http.StatusOK, answer)
}

func (s *Server) handleReady(w http.ResponseWriter, r *http.Request) {
	ctx, cancel := context.WithTimeout(r.Context(), 2*time.Second)
	defer cancel()
	if err := s.rag.Ready(ctx); err != nil {
		writeError(w, http.StatusServiceUnavailable, "RAG service not ready")
		return
	}
	writeJSON(w, http.StatusOK, map[string]string{"status": "ready"})
}

// cacheKey normalizes case and whitespace so trivially different phrasings share an entry.
func cacheKey(b queryBody) string {
	norm := strings.Join(strings.Fields(strings.ToLower(b.Question)), " ")
	sum := sha256.Sum256([]byte(b.Profile + "|" + norm))
	return hex.EncodeToString(sum[:])
}

func clientIP(r *http.Request) string {
	if fwd := r.Header.Get("X-Forwarded-For"); fwd != "" {
		return strings.TrimSpace(strings.Split(fwd, ",")[0])
	}
	host, _, err := net.SplitHostPort(r.RemoteAddr)
	if err != nil {
		return r.RemoteAddr
	}
	return host
}

// --- middleware ---

type statusRecorder struct {
	http.ResponseWriter
	code int
}

func (r *statusRecorder) WriteHeader(code int) {
	r.code = code
	r.ResponseWriter.WriteHeader(code)
}

func (s *Server) instrument(route string, next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		rec := &statusRecorder{ResponseWriter: w, code: http.StatusOK}
		next.ServeHTTP(rec, r)
		elapsed := time.Since(start)
		s.metrics.requests.WithLabelValues(route, strconv.Itoa(rec.code)).Inc()
		s.metrics.latency.WithLabelValues(route).Observe(elapsed.Seconds())
		s.log.Info("request", "route", route, "code", rec.code,
			"duration_ms", elapsed.Milliseconds(), "request_id", requestIDFrom(r.Context()))
	})
}

func (s *Server) requestID(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		id := r.Header.Get("X-Request-ID")
		if id == "" {
			buf := make([]byte, 8)
			_, _ = rand.Read(buf)
			id = hex.EncodeToString(buf)
		}
		w.Header().Set("X-Request-ID", id)
		next.ServeHTTP(w, r.WithContext(ragclient.WithRequestID(withRequestID(r.Context(), id), id)))
	})
}

func (s *Server) recoverer(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		defer func() {
			if v := recover(); v != nil {
				s.log.Error("panic", "value", v, "request_id", requestIDFrom(r.Context()))
				writeError(w, http.StatusInternalServerError, "internal error")
			}
		}()
		next.ServeHTTP(w, r)
	})
}

type ctxKey struct{}

func withRequestID(ctx context.Context, id string) context.Context {
	return context.WithValue(ctx, ctxKey{}, id)
}

func requestIDFrom(ctx context.Context) string {
	id, _ := ctx.Value(ctxKey{}).(string)
	return id
}

// --- responses ---

func writeJSON(w http.ResponseWriter, code int, v any) {
	b, _ := json.Marshal(v)
	writeRaw(w, code, b)
}

func writeRaw(w http.ResponseWriter, code int, b []byte) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	_, _ = w.Write(b)
}

func writeError(w http.ResponseWriter, code int, msg string) {
	writeJSON(w, code, map[string]string{"error": msg})
}
