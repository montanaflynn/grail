package gemini

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/montanaflynn/grail"

	"google.golang.org/genai"
)

var _ grail.StreamExecutor = (*Provider)(nil)

// streamServer replays SSE chunks to a streamGenerateContent request and
// records the request path and body. With hold set, it then keeps the
// response open until the client disconnects, and closes canceled.
type streamServer struct {
	chunks   []map[string]any
	hold     bool
	path     string
	body     map[string]any
	canceled chan struct{}
}

func newStreamProvider(t *testing.T, srv *streamServer) *Provider {
	t.Helper()
	srv.canceled = make(chan struct{})
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		srv.path = r.URL.Path
		if err := json.NewDecoder(r.Body).Decode(&srv.body); err != nil {
			t.Errorf("decode request body: %v", err)
		}
		w.Header().Set("Content-Type", "text/event-stream")
		for _, c := range srv.chunks {
			b, _ := json.Marshal(c)
			_, _ = fmt.Fprintf(w, "data: %s\n\n", b)
			w.(http.Flusher).Flush()
		}
		if srv.hold {
			select {
			case <-r.Context().Done():
				close(srv.canceled)
			case <-time.After(5 * time.Second):
			}
		}
	}))
	t.Cleanup(ts.Close)

	ctx := context.Background()
	p, err := New(ctx, WithAPIKey("dummy"))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	p.client, err = genai.NewClient(ctx, &genai.ClientConfig{
		APIKey:      "dummy",
		Backend:     genai.BackendGeminiAPI,
		HTTPOptions: genai.HTTPOptions{BaseURL: ts.URL},
	})
	if err != nil {
		t.Fatalf("genai.NewClient: %v", err)
	}
	return p
}

// textChunk is a streamed candidate with one text part. A non-nil usage is
// sent as the chunk's usage metadata.
func textChunk(text string, finish string, usage map[string]any) map[string]any {
	cand := map[string]any{"content": map[string]any{"role": "model", "parts": []any{map[string]any{"text": text}}}}
	if finish != "" {
		cand["finishReason"] = finish
	}
	chunk := map[string]any{"candidates": []any{cand}, "responseId": "gem_stream"}
	if usage != nil {
		chunk["usageMetadata"] = usage
	}
	return chunk
}

func TestGemini_DoStream(t *testing.T) {
	srv := &streamServer{chunks: []map[string]any{
		textChunk("Hel", "", map[string]any{"promptTokenCount": 6}),
		textChunk("lo", "", nil),
		textChunk("!", "STOP", map[string]any{"promptTokenCount": 6, "candidatesTokenCount": 3, "totalTokenCount": 9}),
	}}
	p := newStreamProvider(t, srv)

	var deltas []string
	var final grail.StreamEvent
	for ev, err := range p.DoStream(context.Background(), grail.Request{
		Inputs: []grail.Input{grail.InputText("Say hello.")},
		Output: grail.OutputText(),
		Model:  "gemini-test",
	}) {
		if err != nil {
			t.Fatalf("DoStream: %v", err)
		}
		if ev.Done {
			final = ev
			continue
		}
		deltas = append(deltas, ev.Text)
	}

	if strings.Join(deltas, "|") != "Hel|lo|!" {
		t.Errorf("deltas = %q", deltas)
	}
	if !final.Done || final.Usage != (grail.Usage{InputTokens: 6, OutputTokens: 3, TotalTokens: 9}) {
		t.Errorf("final = %+v", final)
	}
	if final.RequestID != "gem_stream" || final.Provider.Name != "gemini" || final.Provider.Models[0].Name != "gemini-test" {
		t.Errorf("final = %+v", final)
	}
	if !strings.HasSuffix(srv.path, "/models/gemini-test:streamGenerateContent") {
		t.Errorf("path = %q, want a streamGenerateContent call", srv.path)
	}
}

func TestGemini_DoStream_RejectsNonText(t *testing.T) {
	p, err := New(context.Background(), WithAPIKey("dummy"))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	for _, err := range p.DoStream(context.Background(), grail.Request{Inputs: []grail.Input{grail.InputText("hi")}, Output: grail.OutputImage(grail.ImageSpec{})}) {
		if code := grail.GetErrorCode(err); code != grail.Unsupported {
			t.Errorf("code = %q (%v), want %q", code, err, grail.Unsupported)
		}
	}
}

func TestGemini_DoStream_BreakCancels(t *testing.T) {
	srv := &streamServer{chunks: []map[string]any{textChunk("Hel", "", nil)}, hold: true}
	p := newStreamProvider(t, srv)

	for ev, err := range p.DoStream(context.Background(), grail.Request{Inputs: []grail.Input{grail.InputText("hi")}, Output: grail.OutputText()}) {
		if err != nil || ev.Text != "Hel" {
			t.Fatalf("first event = %+v, %v", ev, err)
		}
		break
	}

	select {
	case <-srv.canceled:
	case <-time.After(2 * time.Second):
		t.Fatal("server request not canceled after the consumer stopped")
	}
}
