package ratelimit

import (
	"testing"
	"time"
)

func TestBurstThenRefill(t *testing.T) {
	now := time.Unix(0, 0)
	l := New(1, 3)
	l.now = func() time.Time { return now }

	for i := 0; i < 3; i++ {
		if !l.Allow("a") {
			t.Fatalf("request %d within burst was rejected", i)
		}
	}
	if l.Allow("a") {
		t.Fatal("request beyond burst was allowed")
	}
	if !l.Allow("b") {
		t.Fatal("keys must have independent buckets")
	}

	now = now.Add(1500 * time.Millisecond)
	if !l.Allow("a") {
		t.Fatal("token should have refilled after 1.5s at 1 rps")
	}
	if l.Allow("a") {
		t.Fatal("only one token should have refilled")
	}
}

func TestSweepRemovesIdleBuckets(t *testing.T) {
	now := time.Unix(0, 0)
	l := New(1, 1)
	l.now = func() time.Time { return now }
	l.Allow("old")
	now = now.Add(time.Hour)
	l.Allow("new")
	l.Sweep(time.Minute)
	if l.Len() != 1 {
		t.Fatalf("expected 1 bucket after sweep, got %d", l.Len())
	}
}
