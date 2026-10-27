package grail

import (
	"context"
	"fmt"
	"log/slog"
	"strings"
	"time"
)

// Warning codes added to a response when its model is deprecated.
const (
	// WarningModelDeprecated means the request's model is deprecated. The
	// message names its shutdown date and replacement when known.
	WarningModelDeprecated = "model_deprecated"
	// WarningModelMigrated means the request's model had shut down and the
	// client sent the request to its replacement instead (see WithAutoMigrate).
	WarningModelMigrated = "model_migrated"
)

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

// WithAutoMigrate replaces a requested model whose shutdown date has passed
// with its replacement before the request is sent, and adds a
// WarningModelMigrated warning to the response. Only one hop is taken: if the
// replacement is itself deprecated, the response also carries a
// WarningModelDeprecated warning for it. Without this option the request is
// sent unchanged and the provider's error surfaces.
//
// A request without a model is checked against the defaults the provider
// describes (see ModelDescriber). A shut-down default is migrated by setting
// Request.Model to its replacement, but only when the provider describes a
// single model: when it uses several (OpenAI image output uses a text and an
// image model), deprecated ones are warned about and not migrated.
func WithAutoMigrate() ClientOption {
	return clientOptFunc(func(co *clientOpt) {
		co.autoMigrate = true
	})
}

// checkDeprecation checks the models req will use against p's catalog and
// returns the request to send, migrated if WithAutoMigrate applies, with the
// warnings to add to the response. Without req.Model the provider uses its
// configured defaults, which callers may have pinned to an old model, so
// those are checked when p implements ModelDescriber.
func (c *Client) checkDeprecation(ctx context.Context, p ProviderExecutor, req Request) (Request, []Warning) {
	names := []string{req.Model}
	if req.Model == "" {
		names = describedModels(p, req)
	}

	var warnings []Warning
	for _, name := range names {
		// A single model can be swapped through req.Model; when the provider
		// uses several, req.Model can't replace just one of them.
		replacement, ws := c.checkModel(ctx, p, name, c.autoMigrate && len(names) == 1)
		if replacement != "" {
			req.Model = replacement
		}
		warnings = append(warnings, ws...)
	}
	return req, warnings
}

// describedModels returns the models p says it will use for req, e.g. a text
// and an image model for OpenAI image output.
func describedModels(p ProviderExecutor, req Request) []string {
	describer, ok := p.(ModelDescriber)
	if !ok {
		return nil
	}
	var names []string
	for name := range strings.SplitSeq(describer.DescribeModels(req), ",") {
		if name = strings.TrimSpace(name); name != "" {
			names = append(names, name)
		}
	}
	return names
}

// checkModel looks up name in p's catalog and returns the warnings for it,
// and its replacement when migrate is set and the model has shut down.
func (c *Client) checkModel(ctx context.Context, p ProviderExecutor, name string, migrate bool) (string, []Warning) {
	model, ok := catalogModel(ctx, p, name)
	if !ok || !model.IsDeprecated() {
		return "", nil
	}

	now := c.now()
	var replacement string
	var warnings []Warning
	if dep := model.Deprecation; migrate && model.IsShutDown(now) && dep.Replacement != "" && dep.Replacement != model.Name {
		w := Warning{
			Code:    WarningModelMigrated,
			Message: fmt.Sprintf("%s shut down %s; migrated to %s", model.Name, dep.Shutdown.Format(time.DateOnly), dep.Replacement),
		}
		c.warnOnce(p, model.Name, w)
		warnings = append(warnings, w)
		replacement = dep.Replacement

		// Take one hop only: a deprecated replacement is warned about, not
		// migrated again.
		if model, ok = catalogModel(ctx, p, replacement); !ok || !model.IsDeprecated() {
			return replacement, warnings
		}
	}

	w := Warning{Code: WarningModelDeprecated, Message: deprecationMessage(model, now)}
	c.warnOnce(p, model.Name, w)
	return replacement, append(warnings, w)
}

// deprecationMessage describes a deprecated model's shutdown date and
// replacement, e.g. "old is deprecated; shuts down 2026-10-23; use new".
func deprecationMessage(m Model, now time.Time) string {
	msg := m.Name + " is deprecated"
	if dep := m.Deprecation; !dep.Shutdown.IsZero() {
		verb := "shuts down"
		if m.IsShutDown(now) {
			verb = "shut down"
		}
		msg += fmt.Sprintf("; %s %s", verb, dep.Shutdown.Format(time.DateOnly))
	}
	if r := m.Deprecation.Replacement; r != "" {
		msg += "; use " + r
	}
	return msg
}

// warnOnce logs w at Warn level the first time this client sees its code for
// model. It is safe for concurrent use.
func (c *Client) warnOnce(p ProviderExecutor, model string, w Warning) {
	if c.log == nil {
		return
	}
	if _, seen := c.warned.LoadOrStore(w.Code+"\x00"+model, struct{}{}); seen {
		return
	}
	c.log.Warn(w.Message,
		slog.String("code", w.Code),
		slog.String("provider", p.Name()),
		slog.String("model", model),
	)
}

// catalogModel finds name in p's full catalog of known models, which carries
// deprecation records, or else in its model list.
func catalogModel(ctx context.Context, p ProviderExecutor, name string) (Model, bool) {
	if known, ok := p.(KnownModelLister); ok {
		for _, m := range known.KnownModels() {
			if m.Name == name {
				return m, true
			}
		}
	}
	return lookupModel(ctx, p, name)
}
