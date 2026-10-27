package grail

import (
	"context"
	"fmt"
	"iter"
	"log/slog"
)

// StreamEvent is one event of a text stream. Every event but the last carries
// a Text delta. The last event has Done set and carries the stream's Usage,
// Provider, RequestID and Warnings; its Text may be empty.
type StreamEvent struct {
	Text      string // text delta
	Done      bool   // true on the final event
	Usage     Usage
	Provider  ProviderInfo
	RequestID string
	Warnings  []Warning
}

// StreamExecutor is an optional interface for providers that can stream text
// output. DoStream yields text deltas followed by a final event with Done set.
// It stops yielding after an error, and stops reading the underlying stream
// (releasing its connection) when the caller stops iterating.
type StreamExecutor interface {
	DoStream(ctx context.Context, req Request) iter.Seq2[StreamEvent, error]
}

// Stream sends req to a provider and yields the text output as it is
// generated. Only OutputText requests are supported; other outputs yield a
// single Unsupported error.
//
// The request goes to the first provider that can serve it (see
// WithProviders) and supports streaming. Unlike Generate, Stream does not
// fall back to another provider once the stream has started. Breaking out of
// the loop cancels the underlying stream.
//
//	for ev, err := range client.Stream(ctx, req) {
//		if err != nil {
//			return err
//		}
//		fmt.Print(ev.Text)
//	}
func (c *Client) Stream(ctx context.Context, req Request) iter.Seq2[StreamEvent, error] {
	return func(yield func(StreamEvent, error) bool) {
		if err := validateRequest(req); err != nil {
			yield(StreamEvent{}, err)
			return
		}
		if !IsTextOutput(req.Output) {
			yield(StreamEvent{}, NewGrailError(Unsupported, fmt.Sprintf("streaming supports text output only, got %s output", getOutputType(req.Output))))
			return
		}

		routes, err := c.selectProvider(ctx, req)
		if err != nil {
			yield(StreamEvent{}, err)
			return
		}

		var rt route
		var streamer StreamExecutor
		for _, r := range routes {
			if s, ok := r.provider.(StreamExecutor); ok {
				rt, streamer = r, s
				break
			}
		}
		if streamer == nil {
			yield(StreamEvent{}, NewGrailError(Unsupported, fmt.Sprintf("no provider supports streaming (candidates: %s)", routeNames(routes))))
			return
		}

		var warnings []Warning
		rt.req, warnings = c.checkDeprecation(ctx, rt.provider, rt.req)
		if c.log != nil {
			models := rt.req.Model
			if describer, ok := rt.provider.(ModelDescriber); ok {
				models = describer.DescribeModels(rt.req)
			}
			c.log.Debug("stream request",
				slog.String("provider", rt.provider.Name()),
				slog.Int("inputs", len(rt.req.Inputs)),
				slog.String("model", models),
			)
		}

		ctx, cancel := context.WithCancel(ctx)
		defer cancel()

		for ev, err := range streamer.DoStream(ctx, rt.req) {
			if err != nil {
				yield(StreamEvent{}, err)
				return
			}
			if ev.Done {
				if ev.Provider.Name == "" {
					ev.Provider.Name = rt.provider.Name()
				}
				ev.Warnings = append(warnings, ev.Warnings...)
			}
			if !yield(ev, nil) {
				return
			}
		}
	}
}

// routeNames lists the provider names of routes, comma separated.
func routeNames(routes []route) string {
	var s string
	for i, rt := range routes {
		if i > 0 {
			s += ", "
		}
		s += rt.provider.Name()
	}
	return s
}
