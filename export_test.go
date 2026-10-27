package grail

import "time"

// WithClock sets the clock the client uses to decide whether a model has shut
// down. It is exported for tests only.
func WithClock(now func() time.Time) ClientOption {
	return clientOptFunc(func(co *clientOpt) {
		co.now = now
	})
}
