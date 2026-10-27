package grail_test

import (
	"context"
	"iter"
	"log/slog"
	"sync"
	"testing"

	"github.com/montanaflynn/grail"
	"github.com/montanaflynn/grail/providers/mock"
)

// recordingHandler captures log records at or above level.
type recordingHandler struct {
	level   slog.Level
	mu      sync.Mutex
	records []slog.Record
}

func (h *recordingHandler) Enabled(_ context.Context, l slog.Level) bool { return l >= h.level }

func (h *recordingHandler) Handle(_ context.Context, r slog.Record) error {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.records = append(h.records, r)
	return nil
}

func (h *recordingHandler) WithAttrs([]slog.Attr) slog.Handler { return h }
func (h *recordingHandler) WithGroup(string) slog.Handler      { return h }

func (h *recordingHandler) messages() []string {
	h.mu.Lock()
	defer h.mu.Unlock()
	var msgs []string
	for _, r := range h.records {
		msgs = append(msgs, r.Level.String()+" "+r.Message)
	}
	return msgs
}

func TestRequestLogsAreDebug(t *testing.T) {
	ctx := context.Background()
	textProvider := &mock.Provider{GenerateFn: func(context.Context, grail.Request) (grail.Response, error) {
		return grail.Response{Outputs: []grail.OutputPart{grail.NewTextOutputPart("hi")}}, nil
	}}
	streamProvider := &mock.StreamProvider{StreamFn: func(context.Context, grail.Request) iter.Seq2[grail.StreamEvent, error] {
		return mock.TextStream(grail.Usage{}, "hi")
	}}

	t.Run("info level is quiet", func(t *testing.T) {
		h := &recordingHandler{level: slog.LevelInfo}
		client := grail.NewClient(textProvider, grail.WithLogger(slog.New(h)))
		if _, err := client.Text(ctx, "hello"); err != nil {
			t.Fatalf("Text: %v", err)
		}
		streamClient := grail.NewClient(streamProvider, grail.WithLogger(slog.New(h)))
		for _, err := range streamClient.Stream(ctx, grail.Request{Inputs: []grail.Input{grail.InputText("hello")}, Output: grail.OutputText()}) {
			if err != nil {
				t.Fatalf("Stream: %v", err)
			}
		}
		if msgs := h.messages(); len(msgs) != 0 {
			t.Fatalf("expected no logs at Info, got %v", msgs)
		}
	})

	t.Run("debug level shows requests", func(t *testing.T) {
		h := &recordingHandler{level: slog.LevelDebug}
		client := grail.NewClient(textProvider, grail.WithLogger(slog.New(h)))
		if _, err := client.Text(ctx, "hello"); err != nil {
			t.Fatalf("Text: %v", err)
		}
		streamClient := grail.NewClient(streamProvider, grail.WithLogger(slog.New(h)))
		for _, err := range streamClient.Stream(ctx, grail.Request{Inputs: []grail.Input{grail.InputText("hello")}, Output: grail.OutputText()}) {
			if err != nil {
				t.Fatalf("Stream: %v", err)
			}
		}
		want := map[string]bool{"DEBUG generate request": false, "DEBUG stream request": false}
		for _, m := range h.messages() {
			if _, ok := want[m]; ok {
				want[m] = true
			}
		}
		for m, seen := range want {
			if !seen {
				t.Errorf("missing log %q in %v", m, h.messages())
			}
		}
	})
}
