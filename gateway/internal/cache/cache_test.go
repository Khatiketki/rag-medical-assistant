package cache

import (
	"testing"
	"time"
)

func TestEvictsLeastRecentlyUsed(t *testing.T) {
	c := New(2, time.Minute)
	c.Set("a", []byte("1"))
	c.Set("b", []byte("2"))
	c.Get("a") // a is now most recent
	c.Set("c", []byte("3"))

	if _, ok := c.Get("b"); ok {
		t.Fatal("b should have been evicted")
	}
	if v, ok := c.Get("a"); !ok || string(v) != "1" {
		t.Fatal("a should still be cached")
	}
	if c.Len() != 2 {
		t.Fatalf("len = %d, want 2", c.Len())
	}
}

func TestEntriesExpire(t *testing.T) {
	now := time.Unix(0, 0)
	c := New(10, time.Second)
	c.now = func() time.Time { return now }
	c.Set("k", []byte("v"))
	now = now.Add(2 * time.Second)
	if _, ok := c.Get("k"); ok {
		t.Fatal("entry should have expired")
	}
	if c.Len() != 0 {
		t.Fatal("expired entry should be removed")
	}
}
