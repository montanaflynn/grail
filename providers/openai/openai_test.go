package openai

import (
	"testing"

	"github.com/montanaflynn/grail"
)

// Compile-time check that Provider implements grail.Provider.
var _ grail.Provider = (*Provider)(nil)

// Note: We avoid real API calls by using dummy keys; New does not make network requests.

func TestOpenAI_New_APIKeyHandling(t *testing.T) {
	t.Run("explicit empty key errors", func(t *testing.T) {
		_, err := New(WithAPIKey(""))
		if err == nil {
			t.Fatalf("expected error for empty explicit key")
		}
		if err != ErrAPIKeyRequired {
			t.Fatalf("expected ErrAPIKeyRequired, got %v", err)
		}
	})

	t.Run("explicit non-empty key ok", func(t *testing.T) {
		_, err := New(WithAPIKey("dummy"))
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
	})

	t.Run("fallback env set", func(t *testing.T) {
		t.Setenv("OPENAI_API_KEY", "dummy")
		_, err := New()
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
	})

	t.Run("fallback env missing errors", func(t *testing.T) {
		t.Setenv("OPENAI_API_KEY", "")
		_, err := New()
		if err == nil {
			t.Fatalf("expected error for missing env key")
		}
		if err != ErrAPIKeyRequired {
			t.Fatalf("expected ErrAPIKeyRequired, got %v", err)
		}
	})
}
