package openai

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

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
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

// newTestProvider returns a provider whose client talks to an httptest server
// that records each request body and replies with a message containing text.
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
			"id":     "resp_test",
			"object": "response",
			"status": "completed",
			"output": []any{map[string]any{
				"type":   "message",
				"id":     "msg_test",
				"role":   "assistant",
				"status": "completed",
				"content": []any{map[string]any{
					"type":        "output_text",
					"text":        text,
					"annotations": []any{},
				}},
			}},
		})
	}))
	t.Cleanup(srv.Close)

	p, err := New(WithAPIKey("dummy"))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	p.client = openai.NewClient(option.WithAPIKey("dummy"), option.WithBaseURL(srv.URL), option.WithMaxRetries(0))
	return p, &bodies
}

func textFormat(t *testing.T, body map[string]any) map[string]any {
	t.Helper()
	text, ok := body["text"].(map[string]any)
	if !ok {
		t.Fatalf("request has no text config: %v", body)
	}
	format, ok := text["format"].(map[string]any)
	if !ok {
		t.Fatalf("request has no text.format: %v", text)
	}
	return format
}

func TestOpenAI_GenerateJSON_SendsSchema(t *testing.T) {
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
		name       string
		output     grail.Output
		wantStrict bool
	}{
		{"map schema strict", grail.OutputJSON(schema), true},
		{"raw message schema non-strict", grail.OutputJSON(json.RawMessage(raw), grail.WithStrictJSON(false)), false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p, bodies := newTestProvider(t, `{"name":"grail"}`)
			res, err := p.DoGenerate(context.Background(), grail.Request{
				Inputs: []grail.Input{grail.InputText("name?")},
				Output: tt.output,
			})
			if err != nil {
				t.Fatalf("DoGenerate: %v", err)
			}
			if len(*bodies) != 1 {
				t.Fatalf("expected 1 request, got %d", len(*bodies))
			}
			format := textFormat(t, (*bodies)[0])
			if format["type"] != "json_schema" {
				t.Errorf("format type = %v, want json_schema", format["type"])
			}
			if name, _ := format["name"].(string); name == "" {
				t.Errorf("format name missing: %v", format)
			}
			if format["strict"] != tt.wantStrict {
				t.Errorf("format strict = %v, want %v", format["strict"], tt.wantStrict)
			}
			if !reflect.DeepEqual(format["schema"], schema) {
				t.Errorf("format schema = %v, want %v", format["schema"], schema)
			}
			var got struct{ Name string }
			if err := res.DecodeJSON(&got); err != nil || got.Name != "grail" {
				t.Errorf("DecodeJSON = %+v, %v", got, err)
			}
		})
	}
}

func TestOpenAI_GenerateJSON_NilSchemaUsesJSONObject(t *testing.T) {
	p, bodies := newTestProvider(t, `{"ok":true}`)
	_, err := p.DoGenerate(context.Background(), grail.Request{
		Inputs: []grail.Input{grail.InputText("hi")},
		Output: grail.OutputJSON(nil),
	})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}
	format := textFormat(t, (*bodies)[0])
	if format["type"] != "json_object" {
		t.Errorf("format type = %v, want json_object", format["type"])
	}
	if _, ok := format["schema"]; ok {
		t.Errorf("json_object format should not carry a schema: %v", format)
	}
}

func TestOpenAI_GenerateJSON_MalformedOutput(t *testing.T) {
	p, _ := newTestProvider(t, `{"name":`)
	_, err := p.DoGenerate(context.Background(), grail.Request{
		Inputs: []grail.Input{grail.InputText("name?")},
		Output: grail.OutputJSON(map[string]any{"type": "object"}),
	})
	if code := grail.GetErrorCode(err); code != grail.OutputInvalid {
		t.Fatalf("error code = %q (%v), want %q", code, err, grail.OutputInvalid)
	}
}

func TestOpenAI_GenerateJSON_NonObjectSchema(t *testing.T) {
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
