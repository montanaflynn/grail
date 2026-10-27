package anthropic

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"go/ast"
	"go/parser"
	"go/token"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/montanaflynn/grail"

	"github.com/anthropics/anthropic-sdk-go/v2"
	"github.com/anthropics/anthropic-sdk-go/v2/option"
)

// Compile-time interface checks.
var (
	_ grail.ProviderExecutor = (*Provider)(nil)
	_ grail.ModelResolver    = (*Provider)(nil)
	_ grail.ModelLister      = (*Provider)(nil)
	_ grail.ModelCatalog     = (*Provider)(nil)
	_ grail.KnownModelLister = (*Provider)(nil)
	_ grail.ModelDescriber   = (*Provider)(nil)
	_ grail.LoggerAware      = (*Provider)(nil)
)

func TestAnthropic_New_APIKeyHandling(t *testing.T) {
	t.Run("explicit empty key errors", func(t *testing.T) {
		if _, err := New(WithAPIKey("")); err != ErrAPIKeyRequired {
			t.Fatalf("expected ErrAPIKeyRequired, got %v", err)
		}
	})

	t.Run("explicit non-empty key ok", func(t *testing.T) {
		if _, err := New(WithAPIKey("dummy")); err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
	})

	t.Run("custom env var", func(t *testing.T) {
		t.Setenv("MY_CLAUDE_KEY", "dummy")
		if _, err := New(WithAPIKeyFromEnv("MY_CLAUDE_KEY")); err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
	})

	t.Run("custom env var missing errors", func(t *testing.T) {
		t.Setenv("MY_CLAUDE_KEY", "")
		if _, err := New(WithAPIKeyFromEnv("MY_CLAUDE_KEY")); err != ErrAPIKeyRequired {
			t.Fatalf("expected ErrAPIKeyRequired, got %v", err)
		}
	})

	t.Run("fallback env set", func(t *testing.T) {
		t.Setenv("ANTHROPIC_API_KEY", "dummy")
		if _, err := New(); err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
	})

	t.Run("fallback env missing errors", func(t *testing.T) {
		t.Setenv("ANTHROPIC_API_KEY", "")
		if _, err := New(); err != ErrAPIKeyRequired {
			t.Fatalf("expected ErrAPIKeyRequired, got %v", err)
		}
	})
}

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
		if m.Role != grail.ModelRoleText || m.Capabilities.ImageGeneration {
			t.Errorf("%s: Claude models are text-only, got role %s, caps %+v", m.Name, m.Role, m.Capabilities)
		}
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
	}
}

func TestKnownModels_DeprecationRecords(t *testing.T) {
	type want struct{ announced, shutdown, replacement string }
	wants := map[string]want{
		"claude-mythos-preview":      {"2026-06-09", "", "claude-mythos-5"},
		"claude-sonnet-4-5":          {"2026-09-30", "2026-11-30", "claude-sonnet-5-5"},
		"claude-sonnet-4-5-20250929": {"2026-09-30", "2026-11-30", "claude-sonnet-5-5"},
	}
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
		w, ok := wants[m.Name]
		if !ok {
			if m.IsDeprecated() {
				t.Errorf("%s: unexpected deprecation record %+v", m.Name, *m.Deprecation)
			}
			continue
		}
		delete(wants, m.Name)
		if !m.IsDeprecated() {
			t.Errorf("%s: expected a deprecation record", m.Name)
			continue
		}
		d := m.Deprecation
		if !d.Announced.Equal(parse(w.announced)) || !d.Shutdown.Equal(parse(w.shutdown)) || d.Replacement != w.replacement {
			t.Errorf("%s: deprecation = %+v, want %+v", m.Name, *d, w)
		}
	}
	for name := range wants {
		t.Errorf("%s: not in KnownModels", name)
	}
}

func TestResolveModel(t *testing.T) {
	p, err := New(WithAPIKey("dummy"))
	if err != nil {
		t.Fatal(err)
	}
	for _, m := range p.AllModels() {
		if m.IsDeprecated() {
			t.Errorf("default model %s is deprecated", m.Name)
		}
	}

	tests := []struct {
		role grail.ModelRole
		tier grail.ModelTier
		want string
	}{
		{grail.ModelRoleText, grail.ModelTierBest, "claude-fable-5-1"},
		{grail.ModelRoleText, grail.ModelTierFast, "claude-haiku-5-5"},
	}
	for _, tt := range tests {
		got, err := p.ResolveModel(tt.role, tt.tier)
		if err != nil || got != tt.want {
			t.Errorf("ResolveModel(%s, %s) = %q, %v; want %q", tt.role, tt.tier, got, err, tt.want)
		}
	}

	for _, tier := range []grail.ModelTier{grail.ModelTierBest, grail.ModelTierFast} {
		_, err := p.ResolveModel(grail.ModelRoleImage, tier)
		if code := grail.GetErrorCode(err); code != grail.Unsupported {
			t.Errorf("ResolveModel(image, %s) code = %q (%v), want %q", tier, code, err, grail.Unsupported)
		}
	}
}

// testServer is an httptest Messages API that records request bodies and
// replies with status and body.
type testServer struct {
	bodies []map[string]any
	status int
	reply  map[string]any
}

// messageReply builds a Messages API response with one text block.
func messageReply(text, stopReason string) map[string]any {
	return map[string]any{
		"id":            "msg_test",
		"type":          "message",
		"role":          "assistant",
		"model":         "claude-fable-5-1",
		"content":       []any{map[string]any{"type": "text", "text": text}},
		"stop_reason":   stopReason,
		"stop_sequence": nil,
		"usage":         map[string]any{"input_tokens": 12, "output_tokens": 5},
	}
}

func newTestProvider(t *testing.T, srv *testServer) *Provider {
	t.Helper()
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/messages" {
			t.Errorf("unexpected path %s", r.URL.Path)
		}
		var body map[string]any
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Errorf("decode request body: %v", err)
		}
		srv.bodies = append(srv.bodies, body)
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("Request-Id", "req_test")
		if srv.status != 0 {
			w.WriteHeader(srv.status)
		}
		_ = json.NewEncoder(w).Encode(srv.reply)
	}))
	t.Cleanup(ts.Close)

	p, err := New(WithAPIKey("dummy"))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	p.client = anthropic.NewClient(
		option.WithoutEnvironmentDefaults(),
		option.WithAPIKey("dummy"),
		option.WithBaseURL(ts.URL),
		option.WithMaxRetries(0),
	)
	return p
}

// userContent returns the content blocks of the request's single user message.
func userContent(t *testing.T, body map[string]any) []map[string]any {
	t.Helper()
	msgs, _ := body["messages"].([]any)
	if len(msgs) != 1 {
		t.Fatalf("expected 1 message, got %v", body["messages"])
	}
	msg := msgs[0].(map[string]any)
	if msg["role"] != "user" {
		t.Fatalf("message role = %v, want user", msg["role"])
	}
	raw, _ := msg["content"].([]any)
	out := make([]map[string]any, len(raw))
	for i, c := range raw {
		out[i] = c.(map[string]any)
	}
	return out
}

func systemPrompt(body map[string]any) string {
	blocks, _ := body["system"].([]any)
	var parts []string
	for _, b := range blocks {
		parts = append(parts, b.(map[string]any)["text"].(string))
	}
	return strings.Join(parts, "\n")
}

func TestAnthropic_GenerateText(t *testing.T) {
	srv := &testServer{reply: messageReply("Hello there.", "end_turn")}
	p := newTestProvider(t, srv)

	res, err := p.DoGenerate(context.Background(), grail.Request{
		Inputs: []grail.Input{grail.InputText("Say hello.")},
		Output: grail.OutputText(),
	})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}
	if text, _ := res.Text(); text != "Hello there." {
		t.Errorf("text = %q", text)
	}
	if res.Usage != (grail.Usage{InputTokens: 12, OutputTokens: 5, TotalTokens: 17}) {
		t.Errorf("usage = %+v", res.Usage)
	}
	if res.Provider.Name != "anthropic" || res.Provider.Models[0].Name != "claude-fable-5-1" {
		t.Errorf("provider = %+v", res.Provider)
	}
	if res.RequestID != "msg_test" || len(res.Warnings) != 0 {
		t.Errorf("request id = %q, warnings = %v", res.RequestID, res.Warnings)
	}

	body := srv.bodies[0]
	if body["model"] != "claude-fable-5-1" {
		t.Errorf("model = %v", body["model"])
	}
	if body["max_tokens"] != float64(DefaultMaxTokens) {
		t.Errorf("max_tokens = %v, want %d", body["max_tokens"], DefaultMaxTokens)
	}
	if _, ok := body["system"]; ok {
		t.Errorf("unexpected system prompt: %v", body["system"])
	}
	for _, k := range []string{"temperature", "top_p", "top_k", "output_config"} {
		if _, ok := body[k]; ok {
			t.Errorf("unexpected %s in request", k)
		}
	}
	content := userContent(t, body)
	if len(content) != 1 || content[0]["type"] != "text" || content[0]["text"] != "Say hello." {
		t.Errorf("content = %v", content)
	}
}

func TestAnthropic_GenerateText_Options(t *testing.T) {
	t.Run("text options", func(t *testing.T) {
		srv := &testServer{reply: messageReply("ok", "end_turn")}
		p := newTestProvider(t, srv)
		_, err := p.DoGenerate(context.Background(), grail.Request{
			Inputs: []grail.Input{grail.InputText("hi")},
			Output: grail.OutputText(),
			ProviderOptions: []grail.ProviderOption{TextOptions{
				Model:        "claude-sonnet-5-5",
				MaxTokens:    grail.Pointer[int32](256),
				SystemPrompt: "Be brief.",
			}},
		})
		if err != nil {
			t.Fatalf("DoGenerate: %v", err)
		}
		body := srv.bodies[0]
		if body["model"] != "claude-sonnet-5-5" || body["max_tokens"] != float64(256) || systemPrompt(body) != "Be brief." {
			t.Errorf("model = %v, max_tokens = %v, system = %q", body["model"], body["max_tokens"], systemPrompt(body))
		}
	})

	t.Run("request model wins", func(t *testing.T) {
		srv := &testServer{reply: messageReply("ok", "end_turn")}
		p := newTestProvider(t, srv)
		req := grail.Request{
			Inputs:          []grail.Input{grail.InputText("hi")},
			Output:          grail.OutputText(),
			Model:           "claude-haiku-5-5",
			ProviderOptions: []grail.ProviderOption{TextOptions{Model: "claude-sonnet-5-5", MaxTokens: grail.Pointer[int32](64)}},
		}
		if got := p.DescribeModels(req); got != "claude-haiku-5-5" {
			t.Errorf("DescribeModels = %q", got)
		}
		if _, err := p.DoGenerate(context.Background(), req); err != nil {
			t.Fatalf("DoGenerate: %v", err)
		}
		if body := srv.bodies[0]; body["model"] != "claude-haiku-5-5" || body["max_tokens"] != float64(64) {
			t.Errorf("model = %v, max_tokens = %v", body["model"], body["max_tokens"])
		}
	})
}

func TestAnthropic_GenerateText_StopReasons(t *testing.T) {
	t.Run("max_tokens warns", func(t *testing.T) {
		p := newTestProvider(t, &testServer{reply: messageReply("partial", "max_tokens")})
		res, err := p.DoGenerate(context.Background(), grail.Request{
			Inputs: []grail.Input{grail.InputText("hi")},
			Output: grail.OutputText(),
		})
		if err != nil {
			t.Fatalf("DoGenerate: %v", err)
		}
		if len(res.Warnings) != 1 || res.Warnings[0].Code != "max_tokens" {
			t.Errorf("warnings = %v", res.Warnings)
		}
	})

	t.Run("refusal errors", func(t *testing.T) {
		reply := messageReply("", "refusal")
		reply["stop_details"] = map[string]any{"type": "refusal", "category": "cyber", "explanation": "nope"}
		p := newTestProvider(t, &testServer{reply: reply})
		_, err := p.DoGenerate(context.Background(), grail.Request{
			Inputs: []grail.Input{grail.InputText("hi")},
			Output: grail.OutputText(),
		})
		if !grail.IsRefused(err) || !strings.Contains(err.Error(), "cyber") {
			t.Fatalf("err = %v, want refused with category", err)
		}
	})
}

func TestAnthropic_APIErrors(t *testing.T) {
	tests := []struct {
		status    int
		errType   string
		want      grail.ErrorCode
		retryable bool
	}{
		{http.StatusBadRequest, "invalid_request_error", grail.InvalidArgument, false},
		{http.StatusUnauthorized, "authentication_error", grail.Unauthorized, false},
		{http.StatusTooManyRequests, "rate_limit_error", grail.RateLimited, true},
		{529, "overloaded_error", grail.Unavailable, true},
	}
	for _, tt := range tests {
		t.Run(tt.errType, func(t *testing.T) {
			p := newTestProvider(t, &testServer{status: tt.status, reply: map[string]any{
				"type":  "error",
				"error": map[string]any{"type": tt.errType, "message": "boom"},
			}})
			_, err := p.DoGenerate(context.Background(), grail.Request{
				Inputs: []grail.Input{grail.InputText("hi")},
				Output: grail.OutputText(),
			})
			if code := grail.GetErrorCode(err); code != tt.want {
				t.Errorf("code = %q (%v), want %q", code, err, tt.want)
			}
			if got := grail.IsRetryable(err); got != tt.retryable {
				t.Errorf("IsRetryable = %v, want %v", got, tt.retryable)
			}
		})
	}
}

func TestAnthropic_GenerateJSON_SendsSchema(t *testing.T) {
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

	for name, output := range map[string]grail.Output{
		"map schema":         grail.OutputJSON(schema),
		"raw message schema": grail.OutputJSON(json.RawMessage(raw), grail.WithStrictJSON(false)),
	} {
		t.Run(name, func(t *testing.T) {
			srv := &testServer{reply: messageReply(`{"name":"grail"}`, "end_turn")}
			p := newTestProvider(t, srv)
			res, err := p.DoGenerate(context.Background(), grail.Request{
				Inputs: []grail.Input{grail.InputText("name?")},
				Output: output,
			})
			if err != nil {
				t.Fatalf("DoGenerate: %v", err)
			}
			cfg, ok := srv.bodies[0]["output_config"].(map[string]any)
			if !ok {
				t.Fatalf("request has no output_config: %v", srv.bodies[0])
			}
			format, _ := cfg["format"].(map[string]any)
			if format["type"] != "json_schema" {
				t.Errorf("format type = %v, want json_schema", format["type"])
			}
			if !reflect.DeepEqual(format["schema"], schema) {
				t.Errorf("format schema = %v, want %v", format["schema"], schema)
			}
			if _, ok := srv.bodies[0]["tools"]; ok {
				t.Errorf("structured output should not use tools: %v", srv.bodies[0]["tools"])
			}
			var got struct{ Name string }
			if err := res.DecodeJSON(&got); err != nil || got.Name != "grail" {
				t.Errorf("DecodeJSON = %+v, %v", got, err)
			}
		})
	}
}

func TestAnthropic_GenerateJSON_NilSchemaInstructs(t *testing.T) {
	srv := &testServer{reply: messageReply(`{"ok":true}`, "end_turn")}
	p := newTestProvider(t, srv)
	_, err := p.DoGenerate(context.Background(), grail.Request{
		Inputs:          []grail.Input{grail.InputText("hi")},
		Output:          grail.OutputJSON(nil),
		ProviderOptions: []grail.ProviderOption{TextOptions{SystemPrompt: "Be terse."}},
	})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}
	body := srv.bodies[0]
	if _, ok := body["output_config"]; ok {
		t.Errorf("nil schema should not send output_config: %v", body["output_config"])
	}
	if sys := systemPrompt(body); !strings.HasPrefix(sys, "Be terse.") || !strings.Contains(sys, jsonObjectInstruction) {
		t.Errorf("system = %q", sys)
	}
}

func TestAnthropic_GenerateJSON_MalformedOutput(t *testing.T) {
	p := newTestProvider(t, &testServer{reply: messageReply(`{"name":`, "max_tokens")})
	_, err := p.DoGenerate(context.Background(), grail.Request{
		Inputs: []grail.Input{grail.InputText("name?")},
		Output: grail.OutputJSON(map[string]any{"type": "object", "additionalProperties": false}),
	})
	if code := grail.GetErrorCode(err); code != grail.OutputInvalid {
		t.Fatalf("error code = %q (%v), want %q", code, err, grail.OutputInvalid)
	}
}

func TestAnthropic_GenerateJSON_NonObjectSchema(t *testing.T) {
	srv := &testServer{reply: messageReply(`{}`, "end_turn")}
	p := newTestProvider(t, srv)
	_, err := p.DoGenerate(context.Background(), grail.Request{
		Inputs: []grail.Input{grail.InputText("hi")},
		Output: grail.OutputJSON("not a schema"),
	})
	if code := grail.GetErrorCode(err); code != grail.InvalidArgument {
		t.Fatalf("error code = %q (%v), want %q", code, err, grail.InvalidArgument)
	}
	if len(srv.bodies) != 0 {
		t.Fatalf("expected no request, got %d", len(srv.bodies))
	}
}

var pngData = []byte("\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR")

func TestAnthropic_ImageInput(t *testing.T) {
	srv := &testServer{reply: messageReply("a cat", "end_turn")}
	p := newTestProvider(t, srv)
	_, err := p.DoGenerate(context.Background(), grail.Request{
		Inputs: []grail.Input{
			grail.InputText("What is this?"),
			grail.InputImage(pngData), // MIME is sniffed
		},
		Output: grail.OutputText(),
	})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}
	content := userContent(t, srv.bodies[0])
	if len(content) != 2 || content[0]["type"] != "text" || content[1]["type"] != "image" {
		t.Fatalf("content = %v", content)
	}
	src := content[1]["source"].(map[string]any)
	want := map[string]any{
		"type":       "base64",
		"media_type": "image/png",
		"data":       base64.StdEncoding.EncodeToString(pngData),
	}
	if !reflect.DeepEqual(src, want) {
		t.Errorf("image source = %v, want %v", src, want)
	}
}

func TestAnthropic_PDFInput(t *testing.T) {
	pdf := []byte("%PDF-1.7\n%test")
	srv := &testServer{reply: messageReply("a report", "end_turn")}
	p := newTestProvider(t, srv)
	_, err := p.DoGenerate(context.Background(), grail.Request{
		Inputs: []grail.Input{
			grail.InputPDF(pdf, grail.WithFileName("report.pdf")),
			grail.InputText("Summarize."),
		},
		Output: grail.OutputText(),
	})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}
	content := userContent(t, srv.bodies[0])
	if len(content) != 2 || content[0]["type"] != "document" || content[1]["type"] != "text" {
		t.Fatalf("content = %v", content)
	}
	if content[0]["title"] != "report.pdf" {
		t.Errorf("document title = %v", content[0]["title"])
	}
	src := content[0]["source"].(map[string]any)
	want := map[string]any{
		"type":       "base64",
		"media_type": "application/pdf",
		"data":       base64.StdEncoding.EncodeToString(pdf),
	}
	if !reflect.DeepEqual(src, want) {
		t.Errorf("document source = %v, want %v", src, want)
	}
}

func TestAnthropic_InvalidInputs(t *testing.T) {
	tests := map[string]grail.Input{
		"bad pdf header":    grail.InputPDF([]byte("not a pdf")),
		"unsupported image": grail.InputFile([]byte("BM...."), "image/bmp"),
		"unsupported file":  grail.InputFile([]byte{1, 2, 3}, "application/zip"),
		"empty file":        grail.InputFile(nil, "image/png"),
	}
	for name, input := range tests {
		t.Run(name, func(t *testing.T) {
			srv := &testServer{reply: messageReply("", "end_turn")}
			p := newTestProvider(t, srv)
			_, err := p.DoGenerate(context.Background(), grail.Request{
				Inputs: []grail.Input{input},
				Output: grail.OutputText(),
			})
			if code := grail.GetErrorCode(err); code != grail.InvalidArgument {
				t.Errorf("code = %q (%v), want %q", code, err, grail.InvalidArgument)
			}
			if len(srv.bodies) != 0 {
				t.Errorf("expected no request, got %d", len(srv.bodies))
			}
		})
	}
}

func TestAnthropic_ImageOutputUnsupported(t *testing.T) {
	srv := &testServer{reply: messageReply("", "end_turn")}
	p := newTestProvider(t, srv)
	req := grail.Request{
		Inputs: []grail.Input{grail.InputText("Draw a cat.")},
		Output: grail.OutputImage(grail.ImageSpec{Count: 1}),
	}

	_, err := p.DoGenerate(context.Background(), req)
	if code := grail.GetErrorCode(err); code != grail.Unsupported {
		t.Errorf("DoGenerate code = %q (%v), want %q", code, err, grail.Unsupported)
	}

	_, err = grail.NewClient(p).Generate(context.Background(), req)
	if code := grail.GetErrorCode(err); code != grail.Unsupported {
		t.Errorf("Client.Generate code = %q (%v), want %q", code, err, grail.Unsupported)
	}

	if len(srv.bodies) != 0 {
		t.Errorf("expected no request, got %d", len(srv.bodies))
	}
}
