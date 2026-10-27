package grail

import "time"

// Deprecation records a provider's deprecation notice for a model.
type Deprecation struct {
	Announced   time.Time // When the deprecation was announced (zero if unknown)
	Shutdown    time.Time // When the model stops serving requests (zero if no date announced)
	Replacement string    // Name of the model to migrate to
	Notes       string    // Free-form details, e.g. aliasing or routing behavior
}

// IsDeprecated reports whether the model has a deprecation record.
func (m Model) IsDeprecated() bool { return m.Deprecation != nil }

// IsShutDown reports whether the model's announced shutdown date is at or
// before now. A model with no announced shutdown date is never shut down.
func (m Model) IsShutDown(now time.Time) bool {
	if m.Deprecation == nil || m.Deprecation.Shutdown.IsZero() {
		return false
	}
	return !now.Before(m.Deprecation.Shutdown)
}

// KnownModelLister is an optional interface for providers to expose every model
// they know about, including deprecated and shut-down models. Unlike
// ModelLister, the result is not limited to the configured defaults.
type KnownModelLister interface {
	KnownModels() []Model
}
