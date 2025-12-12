package gemini

import (
	"context"
	"testing"

	"github.com/montanaflynn/grail"
)

// Compile-time check that Provider implements grail.Provider.
var _ grail.Provider = (*Provider)(nil)

func TestGemini_New_APIKeyHandling(t *testing.T) {
	t.Run("explicit empty key errors", func(t *testing.T) {
		_, err := New(context.Background(), WithAPIKey(""))
		if err == nil {
			t.Fatalf("expected error for empty explicit key")
		}
		if err != ErrAPIKeyRequired {
			t.Fatalf("expected ErrAPIKeyRequired, got %v", err)
		}
	})

	t.Run("explicit non-empty key ok", func(t *testing.T) {
		_, err := New(context.Background(), WithAPIKey("dummy"))
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
	})

	t.Run("fallback env set", func(t *testing.T) {
		t.Setenv("GEMINI_API_KEY", "dummy")
		_, err := New(context.Background())
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
	})

	t.Run("fallback env missing errors", func(t *testing.T) {
		t.Setenv("GEMINI_API_KEY", "")
		_, err := New(context.Background())
		if err == nil {
			t.Fatalf("expected error for missing env key")
		}
		if err != ErrAPIKeyRequired {
			t.Fatalf("expected ErrAPIKeyRequired, got %v", err)
		}
	})
}
