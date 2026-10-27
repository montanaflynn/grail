package anthropic

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

	"github.com/anthropics/anthropic-sdk-go/v2"
	"github.com/anthropics/anthropic-sdk-go/v2/option"
)

var _ grail.StreamExecutor = (*Provider)(nil)

// streamServer replays SSE frames to a streaming Messages API request and
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
	p.client = anthropic.NewClient(
		option.WithoutEnvironmentDefaults(),
		option.WithAPIKey("dummy"),
		option.WithBaseURL(ts.URL),
		option.WithMaxRetries(0),
	)
	return p
}

func messageStart() string {
	return sseFrame("message_start", map[string]any{"message": map[string]any{
		"id": "msg_stream", "type": "message", "role": "assistant", "model": "claude-test",
		"content": []any{}, "stop_reason": nil, "stop_sequence": nil,
		"usage": map[string]any{"input_tokens": 11, "output_tokens": 1},
	}})
}

func blockStart() string {
	return sseFrame("content_block_start", map[string]any{"index": 0, "content_block": map[string]any{"type": "text", "text": ""}})
}

func textDelta(text string) string {
	return sseFrame("content_block_delta", map[string]any{"index": 0, "delta": map[string]any{"type": "text_delta", "text": text}})
}

func messageEnd(stopReason string, outputTokens int) []string {
	return []string{
		sseFrame("content_block_stop", map[string]any{"index": 0}),
		sseFrame("message_delta", map[string]any{
			"delta": map[string]any{"stop_reason": stopReason, "stop_sequence": nil},
			"usage": map[string]any{"output_tokens": outputTokens},
		}),
		sseFrame("message_stop", map[string]any{}),
	}
}

func streamText(t *testing.T, p *Provider) (string, grail.StreamEvent, error) {
	t.Helper()
	var text strings.Builder
	var final grail.StreamEvent
	for ev, err := range p.DoStream(context.Background(), grail.Request{
		Inputs: []grail.Input{grail.InputText("Say hello.")},
		Output: grail.OutputText(),
		Model:  "claude-test",
	}) {
		if err != nil {
			return text.String(), final, err
		}
		if ev.Done {
			final = ev
			continue
		}
		text.WriteString(ev.Text + "|")
	}
	return text.String(), final, nil
}

func TestAnthropic_DoStream(t *testing.T) {
	srv := &streamServer{frames: append([]string{
		messageStart(),
		sseFrame("ping", map[string]any{}),
		blockStart(),
		textDelta("Hel"),
		textDelta("lo"),
		textDelta("!"),
	}, messageEnd("end_turn", 4)...)}
	p := newStreamProvider(t, srv)

	deltas, final, err := streamText(t, p)
	if err != nil {
		t.Fatalf("DoStream: %v", err)
	}
	if deltas != "Hel|lo|!|" {
		t.Errorf("deltas = %q", deltas)
	}
	if !final.Done || final.Usage != (grail.Usage{InputTokens: 11, OutputTokens: 4, TotalTokens: 15}) {
		t.Errorf("final = %+v", final)
	}
	if final.RequestID != "msg_stream" || final.Provider.Name != "anthropic" || final.Provider.Models[0].Name != "claude-test" || len(final.Warnings) != 0 {
		t.Errorf("final = %+v", final)
	}
	if srv.body["stream"] != true || srv.body["model"] != "claude-test" {
		t.Errorf("request = %v, want stream: true and model claude-test", srv.body)
	}
}

func TestAnthropic_DoStream_MaxTokensWarning(t *testing.T) {
	srv := &streamServer{frames: append([]string{messageStart(), blockStart(), textDelta("Hel")}, messageEnd("max_tokens", 1)...)}
	p := newStreamProvider(t, srv)

	_, final, err := streamText(t, p)
	if err != nil {
		t.Fatalf("DoStream: %v", err)
	}
	if len(final.Warnings) != 1 || final.Warnings[0].Code != "max_tokens" {
		t.Errorf("warnings = %v, want max_tokens", final.Warnings)
	}
}

func TestAnthropic_DoStream_Errors(t *testing.T) {
	t.Run("refusal", func(t *testing.T) {
		srv := &streamServer{frames: append([]string{messageStart(), blockStart(), textDelta("I")}, messageEnd("refusal", 1)...)}
		_, _, err := streamText(t, newStreamProvider(t, srv))
		if code := grail.GetErrorCode(err); code != grail.Refused {
			t.Errorf("code = %q (%v), want %q", code, err, grail.Refused)
		}
	})

	t.Run("error event", func(t *testing.T) {
		srv := &streamServer{frames: []string{
			messageStart(), blockStart(), textDelta("Hel"),
			sseFrame("error", map[string]any{"error": map[string]any{"type": "overloaded_error", "message": "Overloaded"}}),
		}}
		deltas, _, err := streamText(t, newStreamProvider(t, srv))
		if deltas != "Hel|" || err == nil || !strings.Contains(err.Error(), "Overloaded") {
			t.Errorf("deltas = %q, err = %v; want Hel then an overloaded error", deltas, err)
		}
	})

	t.Run("truncated stream", func(t *testing.T) {
		srv := &streamServer{frames: []string{messageStart(), blockStart(), textDelta("Hel")}}
		_, _, err := streamText(t, newStreamProvider(t, srv))
		if err == nil || !grail.IsRetryable(err) {
			t.Errorf("err = %v, want a retryable error", err)
		}
	})

	t.Run("image output", func(t *testing.T) {
		p, err := New(WithAPIKey("dummy"))
		if err != nil {
			t.Fatalf("New: %v", err)
		}
		for _, err := range p.DoStream(context.Background(), grail.Request{Inputs: []grail.Input{grail.InputText("hi")}, Output: grail.OutputImage(grail.ImageSpec{})}) {
			if code := grail.GetErrorCode(err); code != grail.Unsupported {
				t.Errorf("code = %q (%v), want %q", code, err, grail.Unsupported)
			}
		}
	})
}

func TestAnthropic_DoStream_BreakCancels(t *testing.T) {
	srv := &streamServer{frames: []string{messageStart(), blockStart(), textDelta("Hel")}, hold: true}
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
