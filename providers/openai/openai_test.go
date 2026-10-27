package openai

import (
	"go/ast"
	"go/parser"
	"go/token"
	"testing"
	"time"

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

// Compile-time check that Provider implements grail.KnownModelLister.
var _ grail.KnownModelLister = (*Provider)(nil)

// TestKnownModels_CoversExportedModels parses models.go and checks that every
// exported grail.Model variable is returned by KnownModels exactly once.
func TestKnownModels_CoversExportedModels(t *testing.T) {
	f, err := parser.ParseFile(token.NewFileSet(), "models.go", nil, 0)
	if err != nil {
		t.Fatalf("parse models.go: %v", err)
	}
	var exported int
	for _, decl := range f.Decls {
		gd, ok := decl.(*ast.GenDecl)
		if !ok || gd.Tok != token.VAR {
			continue
		}
		for _, spec := range gd.Specs {
			vs := spec.(*ast.ValueSpec)
			for i, name := range vs.Names {
				if !name.IsExported() || i >= len(vs.Values) {
					continue
				}
				lit, ok := vs.Values[i].(*ast.CompositeLit)
				if !ok {
					continue
				}
				if sel, ok := lit.Type.(*ast.SelectorExpr); ok && sel.Sel.Name == "Model" {
					exported++
				}
			}
		}
	}

	known := (&Provider{}).KnownModels()
	seen := make(map[string]bool, len(known))
	for _, m := range known {
		if seen[m.Name] {
			t.Errorf("KnownModels lists %q more than once", m.Name)
		}
		seen[m.Name] = true
	}
	if len(known) != exported {
		t.Fatalf("KnownModels returned %d models, models.go exports %d", len(known), exported)
	}
}

func TestKnownModels_ReplacementsAreCurrent(t *testing.T) {
	known := make(map[string]grail.Model)
	for _, m := range (&Provider{}).KnownModels() {
		known[m.Name] = m
	}
	for _, m := range known {
		if !m.IsDeprecated() {
			continue
		}
		r, ok := known[m.Deprecation.Replacement]
		if !ok {
			t.Errorf("%s: replacement %q is not in KnownModels", m.Name, m.Deprecation.Replacement)
			continue
		}
		if r.IsDeprecated() {
			t.Errorf("%s: replacement %q is itself deprecated", m.Name, r.Name)
		}
		if r.Role != m.Role {
			t.Errorf("%s: replacement %q has role %s, want %s", m.Name, r.Name, r.Role, m.Role)
		}
	}
}

func TestKnownModels_DefaultsAreCurrent(t *testing.T) {
	for _, m := range defaultModelsForTest(t) {
		if m.IsDeprecated() {
			t.Errorf("default model %s is deprecated", m.Name)
		}
	}
}

type wantDeprecation struct {
	announced, shutdown string // YYYY-MM-DD, empty for zero
	replacement         string
}

func checkDeprecations(t *testing.T, want map[string]wantDeprecation) {
	t.Helper()
	parse := func(s string) time.Time {
		if s == "" {
			return time.Time{}
		}
		d, err := time.Parse(time.DateOnly, s)
		if err != nil {
			t.Fatalf("bad date %q: %v", s, err)
		}
		return d
	}
	for _, m := range (&Provider{}).KnownModels() {
		w, ok := want[m.Name]
		if !ok {
			if m.IsDeprecated() {
				t.Errorf("%s: unexpected deprecation record %+v", m.Name, *m.Deprecation)
			}
			continue
		}
		delete(want, m.Name)
		if !m.IsDeprecated() {
			t.Errorf("%s: expected a deprecation record", m.Name)
			continue
		}
		d := m.Deprecation
		if !d.Announced.Equal(parse(w.announced)) {
			t.Errorf("%s: Announced = %v, want %q", m.Name, d.Announced, w.announced)
		}
		if !d.Shutdown.Equal(parse(w.shutdown)) {
			t.Errorf("%s: Shutdown = %v, want %q", m.Name, d.Shutdown, w.shutdown)
		}
		if d.Replacement != w.replacement {
			t.Errorf("%s: Replacement = %q, want %q", m.Name, d.Replacement, w.replacement)
		}
	}
	for name := range want {
		t.Errorf("%s: not in KnownModels", name)
	}
}

func defaultModelsForTest(t *testing.T) []grail.Model {
	p, err := New(WithAPIKey("dummy"))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return p.AllModels()
}

func TestKnownModels_DeprecationRecords(t *testing.T) {
	checkDeprecations(t, map[string]wantDeprecation{
		GPTImage1.Name:     {shutdown: "2026-10-23", replacement: "gpt-image-2.5-sunburst"},
		GPTImage1Mini.Name: {shutdown: "2026-12-01", replacement: "gpt-image-2.5-flare"},
		GPT5_4Nano.Name:    {shutdown: "2027-04-01", replacement: "gpt-6-luna"},
		GPT4o.Name:         {shutdown: "2027-01-14", replacement: "gpt-6-luna"},
	})
}
