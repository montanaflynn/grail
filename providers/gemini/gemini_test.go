package gemini

import (
	"context"
	"encoding/json"
	"go/ast"
	"go/parser"
	"go/token"
	"net/http"
	"net/http/httptest"
	"reflect"
	"testing"
	"time"

	"github.com/montanaflynn/grail"

	"google.golang.org/genai"
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
	p, err := New(context.Background(), WithAPIKey("dummy"))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return p.AllModels()
}

func TestKnownModels_DeprecationRecords(t *testing.T) {
	checkDeprecations(t, map[string]wantDeprecation{
		Gemini3Pro.Name:                 {shutdown: "2026-03-09", replacement: "gemini-4-argon"},
		Gemini3ProImagePreview.Name:     {shutdown: "2026-06-25", replacement: "gemini-3-pro-image"},
		Gemini3_1FlashImagePreview.Name: {shutdown: "2026-06-25", replacement: "gemini-nano-banana-2.1"},
		Gemini3_5Flash.Name:             {announced: "2026-10-08", replacement: "gemini-3.8-flash"},
		Gemini3_1FlashImage.Name:        {announced: "2026-10-06", replacement: "gemini-nano-banana-2.1"},
		Gemini3Flash.Name:               {replacement: "gemini-3.8-flash"},
	})
}

// newTestProvider returns a provider whose client talks to an httptest server
// that records each request body and replies with a candidate containing text.
func newTestProvider(t *testing.T, text string) (*Provider, *[]map[string]any) {
	t.Helper()
	var bodies []map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Errorf("decode request body: %v", err)
		}
		bodies = append(bodies, body)
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"candidates": []any{map[string]any{
				"content": map[string]any{
					"role":  "model",
					"parts": []any{map[string]any{"text": text}},
				},
			}},
		})
	}))
	t.Cleanup(srv.Close)

	ctx := context.Background()
	p, err := New(ctx, WithAPIKey("dummy"))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	p.client, err = genai.NewClient(ctx, &genai.ClientConfig{
		APIKey:      "dummy",
		Backend:     genai.BackendGeminiAPI,
		HTTPOptions: genai.HTTPOptions{BaseURL: srv.URL},
	})
	if err != nil {
		t.Fatalf("genai.NewClient: %v", err)
	}
	return p, &bodies
}

func generationConfig(t *testing.T, body map[string]any) map[string]any {
	t.Helper()
	cfg, ok := body["generationConfig"].(map[string]any)
	if !ok {
		t.Fatalf("request has no generationConfig: %v", body)
	}
	return cfg
}

func TestGemini_GenerateJSON_SendsSchema(t *testing.T) {
	schema := map[string]any{
		"type":                 "object",
		"properties":           map[string]any{"name": map[string]any{"type": "string"}},
		"required":             []any{"name"},
		"additionalProperties": false,
	}
	raw, err := json.Marshal(schema)
	if err != nil {
		t.Fatal(err)
	}

	tests := []struct {
		name   string
		schema any
	}{
		{"map schema", schema},
		{"raw message schema", json.RawMessage(raw)},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p, bodies := newTestProvider(t, `{"name":"grail"}`)
			res, err := p.DoGenerate(context.Background(), grail.Request{
				Inputs: []grail.Input{grail.InputText("name?")},
				Output: grail.OutputJSON(tt.schema),
			})
			if err != nil {
				t.Fatalf("DoGenerate: %v", err)
			}
			if len(*bodies) != 1 {
				t.Fatalf("expected 1 request, got %d", len(*bodies))
			}
			cfg := generationConfig(t, (*bodies)[0])
			if cfg["responseMimeType"] != "application/json" {
				t.Errorf("responseMimeType = %v, want application/json", cfg["responseMimeType"])
			}
			if !reflect.DeepEqual(cfg["responseJsonSchema"], schema) {
				t.Errorf("responseJsonSchema = %v, want %v", cfg["responseJsonSchema"], schema)
			}
			if _, ok := cfg["responseSchema"]; ok {
				t.Errorf("responseSchema must be omitted when responseJsonSchema is set")
			}
			var got struct{ Name string }
			if err := res.DecodeJSON(&got); err != nil || got.Name != "grail" {
				t.Errorf("DecodeJSON = %+v, %v", got, err)
			}
		})
	}
}

func TestGemini_GenerateJSON_NilSchema(t *testing.T) {
	p, bodies := newTestProvider(t, `{"ok":true}`)
	_, err := p.DoGenerate(context.Background(), grail.Request{
		Inputs: []grail.Input{grail.InputText("hi")},
		Output: grail.OutputJSON(nil),
	})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}
	cfg := generationConfig(t, (*bodies)[0])
	if cfg["responseMimeType"] != "application/json" {
		t.Errorf("responseMimeType = %v, want application/json", cfg["responseMimeType"])
	}
	if _, ok := cfg["responseJsonSchema"]; ok {
		t.Errorf("nil schema should not send responseJsonSchema: %v", cfg)
	}
}

func TestGemini_GenerateJSON_MalformedOutput(t *testing.T) {
	p, _ := newTestProvider(t, `{"name":`)
	_, err := p.DoGenerate(context.Background(), grail.Request{
		Inputs: []grail.Input{grail.InputText("name?")},
		Output: grail.OutputJSON(map[string]any{"type": "object"}),
	})
	if code := grail.GetErrorCode(err); code != grail.OutputInvalid {
		t.Fatalf("error code = %q (%v), want %q", code, err, grail.OutputInvalid)
	}
}

func TestGemini_GenerateJSON_NonObjectSchema(t *testing.T) {
	p, bodies := newTestProvider(t, `{}`)
	_, err := p.DoGenerate(context.Background(), grail.Request{
		Inputs: []grail.Input{grail.InputText("hi")},
		Output: grail.OutputJSON("not a schema"),
	})
	if code := grail.GetErrorCode(err); code != grail.InvalidArgument {
		t.Fatalf("error code = %q (%v), want %q", code, err, grail.InvalidArgument)
	}
	if len(*bodies) != 0 {
		t.Fatalf("expected no request, got %d", len(*bodies))
	}
}
