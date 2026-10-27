package openai

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

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
)

var _ grail.StreamExecutor = (*Provider)(nil)

// streamServer replays SSE frames to a Responses API streaming request and
// records the request body. With hold set, it then keeps the response open
// until the client disconnects, and closes canceled.
type streamServer struct {
	frames   []string
	hold     bool
	body     map[string]any
	canceled chan struct{}
}

func sseFrame(event string, data map[string]any) string {
	data["type"] = event
	b, _ := json.Marshal(data)
	return fmt.Sprintf("event: %s\ndata: %s\n\n", event, b)
}

func newStreamProvider(t *testing.T, srv *streamServer) *Provider {
	t.Helper()
	srv.canceled = make(chan struct{})
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&srv.body); err != nil {
			t.Errorf("decode request body: %v", err)
		}
		w.Header().Set("Content-Type", "text/event-stream")
		for _, f := range srv.frames {
			_, _ = w.Write([]byte(f))
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

	p, err := New(WithAPIKey("dummy"))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	p.client = openai.NewClient(option.WithAPIKey("dummy"), option.WithBaseURL(ts.URL), option.WithMaxRetries(0))
	return p
}

func textDelta(seq int, delta string) string {
	return sseFrame("response.output_text.delta", map[string]any{
		"sequence_number": seq, "item_id": "msg_1", "output_index": 0, "content_index": 0, "delta": delta, "logprobs": []any{},
	})
}

func completed(seq int, status string, extra map[string]any) string {
	resp := map[string]any{
		"id": "resp_stream", "object": "response", "status": status, "output": []any{},
		"usage": map[string]any{
			"input_tokens": 9, "output_tokens": 3, "total_tokens": 12,
			"input_tokens_details": map[string]any{"cached_tokens": 0}, "output_tokens_details": map[string]any{"reasoning_tokens": 0},
		},
	}
	for k, v := range extra {
		resp[k] = v
	}
	return sseFrame("response."+status, map[string]any{"sequence_number": seq, "response": resp})
}

func TestOpenAI_DoStream(t *testing.T) {
	srv := &streamServer{frames: []string{
		sseFrame("response.created", map[string]any{"sequence_number": 0, "response": map[string]any{"id": "resp_stream", "object": "response", "status": "in_progress", "output": []any{}}}),
		textDelta(1, "Hel"),
		textDelta(2, "lo"),
		textDelta(3, "!"),
		completed(4, "completed", nil),
	}}
	p := newStreamProvider(t, srv)

	var deltas []string
	var final grail.StreamEvent
	for ev, err := range p.DoStream(context.Background(), grail.Request{
		Inputs: []grail.Input{grail.InputText("Say hello.")},
		Output: grail.OutputText(),
		Model:  "gpt-test",
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
	if !final.Done || final.Usage != (grail.Usage{InputTokens: 9, OutputTokens: 3, TotalTokens: 12}) {
		t.Errorf("final = %+v", final)
	}
	if final.RequestID != "resp_stream" || final.Provider.Name != "openai" || final.Provider.Models[0].Name != "gpt-test" {
		t.Errorf("final = %+v", final)
	}
	if srv.body["stream"] != true || srv.body["model"] != "gpt-test" {
		t.Errorf("request = %v, want stream: true and model gpt-test", srv.body)
	}
}

func TestOpenAI_DoStream_Incomplete(t *testing.T) {
	srv := &streamServer{frames: []string{
		textDelta(1, "Hel"),
		completed(2, "incomplete", map[string]any{"incomplete_details": map[string]any{"reason": "max_output_tokens"}}),
	}}
	p := newStreamProvider(t, srv)

	var final grail.StreamEvent
	for ev, err := range p.DoStream(context.Background(), grail.Request{Inputs: []grail.Input{grail.InputText("hi")}, Output: grail.OutputText()}) {
		if err != nil {
			t.Fatalf("DoStream: %v", err)
		}
		final = ev
	}
	if !final.Done || len(final.Warnings) != 1 || !strings.Contains(final.Warnings[0].Message, "max_output_tokens") {
		t.Errorf("final = %+v, want an incomplete warning", final)
	}
}

func TestOpenAI_DoStream_Failed(t *testing.T) {
	srv := &streamServer{frames: []string{
		textDelta(1, "Hel"),
		sseFrame("response.failed", map[string]any{"sequence_number": 2, "response": map[string]any{
			"id": "resp_stream", "object": "response", "status": "failed", "output": []any{},
			"error": map[string]any{"code": "server_error", "message": "boom"},
		}}),
	}}
	p := newStreamProvider(t, srv)

	var text string
	var gotErr error
	for ev, err := range p.DoStream(context.Background(), grail.Request{Inputs: []grail.Input{grail.InputText("hi")}, Output: grail.OutputText()}) {
		if err != nil {
			gotErr = err
			break
		}
		text += ev.Text
	}
	if text != "Hel" || gotErr == nil || !strings.Contains(gotErr.Error(), "boom") || !grail.IsRetryable(gotErr) {
		t.Errorf("text = %q, err = %v; want Hel then a retryable boom error", text, gotErr)
	}
}

func TestOpenAI_DoStream_RejectsNonText(t *testing.T) {
	p, err := New(WithAPIKey("dummy"))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	for _, err := range p.DoStream(context.Background(), grail.Request{Inputs: []grail.Input{grail.InputText("hi")}, Output: grail.OutputJSON(nil)}) {
		if code := grail.GetErrorCode(err); code != grail.Unsupported {
			t.Errorf("code = %q (%v), want %q", code, err, grail.Unsupported)
		}
	}
}

func TestOpenAI_DoStream_BreakCancels(t *testing.T) {
	srv := &streamServer{frames: []string{textDelta(1, "Hel")}, hold: true}
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
