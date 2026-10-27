package grail_test

import (
	"context"
	"iter"
	"strings"
	"testing"

	"github.com/montanaflynn/grail"
	"github.com/montanaflynn/grail/providers/mock"
)

// streamingProvider returns a mock that streams chunks and records each call
// in calls. A non-nil err is yielded after the chunks instead of the final event.
func streamingProvider(name string, calls *[]string, err error, chunks ...string) *mock.StreamProvider {
	return &mock.StreamProvider{
		Provider: mock.Provider{NameVal: name},
		StreamFn: func(ctx context.Context, req grail.Request) iter.Seq2[grail.StreamEvent, error] {
			*calls = append(*calls, name)
			if err == nil {
				return mock.TextStream(grail.Usage{InputTokens: 3, OutputTokens: 4, TotalTokens: 7}, chunks...)
			}
			return func(yield func(grail.StreamEvent, error) bool) {
				for _, c := range chunks {
					if !yield(grail.StreamEvent{Text: c}, nil) {
						return
					}
				}
				yield(grail.StreamEvent{}, err)
			}
		},
	}
}

func textRequest() grail.Request {
	return grail.Request{Inputs: routeText, Output: grail.OutputText()}
}

// collect drains a stream into its text, its final event and the first error.
func collect(seq iter.Seq2[grail.StreamEvent, error]) (string, grail.StreamEvent, error) {
	var text strings.Builder
	var final grail.StreamEvent
	for ev, err := range seq {
		if err != nil {
			return text.String(), final, err
		}
		text.WriteString(ev.Text)
		if ev.Done {
			final = ev
		}
	}
	return text.String(), final, nil
}

func TestStreamText(t *testing.T) {
	var calls []string
	client := routedClient(streamingProvider("streamer", &calls, nil, "Hel", "lo", "!"))

	var deltas []string
	var final grail.StreamEvent
	for ev, err := range client.Stream(context.Background(), textRequest()) {
		if err != nil {
			t.Fatalf("Stream: %v", err)
		}
		if final.Done {
			t.Fatal("event after the final event")
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
	if !final.Done || final.Usage != (grail.Usage{InputTokens: 3, OutputTokens: 4, TotalTokens: 7}) {
		t.Errorf("final event = %+v", final)
	}
	if final.Provider.Name != "streamer" {
		t.Errorf("final provider = %q, want streamer (filled in by the client)", final.Provider.Name)
	}
}

func TestStreamRejectsNonTextOutput(t *testing.T) {
	var calls []string
	client := routedClient(streamingProvider("streamer", &calls, nil, "x"))

	for name, output := range map[string]grail.Output{
		"json":  grail.OutputJSON(nil),
		"image": grail.OutputImage(grail.ImageSpec{}),
	} {
		_, _, err := collect(client.Stream(context.Background(), grail.Request{Inputs: routeText, Output: output}))
		if code := grail.GetErrorCode(err); code != grail.Unsupported {
			t.Errorf("%s output: code = %q (%v), want %q", name, code, err, grail.Unsupported)
		}
	}
	assertCalls(t, calls)
}

func TestStreamValidatesRequest(t *testing.T) {
	var calls []string
	client := routedClient(streamingProvider("streamer", &calls, nil, "x"))

	_, _, err := collect(client.Stream(context.Background(), grail.Request{Output: grail.OutputText()}))
	if code := grail.GetErrorCode(err); code != grail.InvalidArgument {
		t.Errorf("code = %q (%v), want %q", code, err, grail.InvalidArgument)
	}
	assertCalls(t, calls)
}

func TestStreamWithoutStreamingProvider(t *testing.T) {
	client := routedClient(&mock.Provider{NameVal: "plain"})

	_, _, err := collect(client.Stream(context.Background(), textRequest()))
	if code := grail.GetErrorCode(err); code != grail.Unsupported {
		t.Fatalf("code = %q (%v), want %q", code, err, grail.Unsupported)
	}
	if !strings.Contains(err.Error(), "plain") {
		t.Errorf("error should name the candidate providers: %v", err)
	}
}

func TestStreamPicksFirstStreamingCandidate(t *testing.T) {
	var calls []string
	client := routedClient(
		&mock.Provider{NameVal: "plain"},
		streamingProvider("first", &calls, nil, "a"),
		streamingProvider("second", &calls, nil, "b"),
	)

	text, final, err := collect(client.Stream(context.Background(), textRequest()))
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}
	if text != "a" || final.Provider.Name != "first" {
		t.Errorf("text = %q, provider = %q, want a from first", text, final.Provider.Name)
	}
	assertCalls(t, calls, "first")
}

func TestStreamNoFallbackMidStream(t *testing.T) {
	var calls []string
	retryable := grail.NewGrailError(grail.Unavailable, "overloaded").WithRetryable(true)
	client := routedClient(
		streamingProvider("primary", &calls, retryable, "partial"),
		streamingProvider("backup", &calls, nil, "full"),
	)

	text, _, err := collect(client.Stream(context.Background(), textRequest()))
	if code := grail.GetErrorCode(err); code != grail.Unavailable {
		t.Fatalf("code = %q (%v), want %q", code, err, grail.Unavailable)
	}
	if text != "partial" {
		t.Errorf("text before error = %q, want partial", text)
	}
	assertCalls(t, calls, "primary")
}

func TestStreamBreakCancelsProviderStream(t *testing.T) {
	var streamCtx context.Context
	var yielded int
	client := routedClient(&mock.StreamProvider{
		StreamFn: func(ctx context.Context, req grail.Request) iter.Seq2[grail.StreamEvent, error] {
			streamCtx = ctx
			return func(yield func(grail.StreamEvent, error) bool) {
				for ctx.Err() == nil {
					yielded++
					if !yield(grail.StreamEvent{Text: "tick"}, nil) {
						return
					}
				}
			}
		},
	})

	for ev, err := range client.Stream(context.Background(), textRequest()) {
		if err != nil {
			t.Fatalf("Stream: %v", err)
		}
		if ev.Text != "tick" {
			t.Fatalf("text = %q", ev.Text)
		}
		break
	}

	if yielded != 1 {
		t.Errorf("provider yielded %d events, want 1", yielded)
	}
	if streamCtx == nil || streamCtx.Err() == nil {
		t.Error("provider context not canceled after the consumer stopped")
	}
}
