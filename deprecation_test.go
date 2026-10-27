package grail_test

import (
	"context"
	"fmt"
	"io"
	"iter"
	"log/slog"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/montanaflynn/grail"
	"github.com/montanaflynn/grail/providers/mock"
)

func TestModel_IsDeprecated(t *testing.T) {
	if (grail.Model{Name: "current"}).IsDeprecated() {
		t.Fatalf("model without a deprecation record should not be deprecated")
	}
	if !(grail.Model{Name: "old", Deprecation: &grail.Deprecation{Replacement: "new"}}).IsDeprecated() {
		t.Fatalf("model with a deprecation record should be deprecated")
	}
}

func TestModel_IsShutDown(t *testing.T) {
	shutdown := time.Date(2026, time.October, 23, 0, 0, 0, 0, time.UTC)
	dated := grail.Model{Name: "old", Deprecation: &grail.Deprecation{Shutdown: shutdown, Replacement: "new"}}
	undated := grail.Model{Name: "old", Deprecation: &grail.Deprecation{Replacement: "new"}}
	current := grail.Model{Name: "current"}

	tests := []struct {
		name  string
		model grail.Model
		now   time.Time
		want  bool
	}{
		{"not deprecated", current, shutdown.AddDate(10, 0, 0), false},
		{"no shutdown date", undated, shutdown.AddDate(10, 0, 0), false},
		{"before shutdown", dated, shutdown.Add(-time.Second), false},
		{"at shutdown", dated, shutdown, true},
		{"after shutdown", dated, shutdown.Add(time.Hour), true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.model.IsShutDown(tt.now); got != tt.want {
				t.Fatalf("IsShutDown(%v) = %v, want %v", tt.now, got, tt.want)
			}
		})
	}
}

var (
	// depNow is the test clock: after oldShutdown, before soonShutdown.
	depNow       = time.Date(2026, time.November, 1, 12, 0, 0, 0, time.UTC)
	oldShutdown  = time.Date(2026, time.October, 23, 0, 0, 0, 0, time.UTC)
	soonShutdown = time.Date(2027, time.January, 14, 0, 0, 0, 0, time.UTC)
)

// deprecatedModel returns a best-tier text model that shuts down at shutdown
// (zero for no date) and is replaced by replacement.
func deprecatedModel(name string, shutdown time.Time, replacement string) grail.Model {
	m := textModel(name, grail.ModelTierBest, grail.ModelCapabilities{})
	m.Deprecation = &grail.Deprecation{Shutdown: shutdown, Replacement: replacement}
	return m
}

// depClient returns a client over p with the test clock and a quiet logger.
func depClient(p grail.Provider, opts ...grail.ClientOption) *grail.Client {
	quiet := slog.New(slog.NewTextHandler(io.Discard, nil))
	opts = append([]grail.ClientOption{grail.WithClock(func() time.Time { return depNow }), grail.WithLogger(quiet)}, opts...)
	return grail.NewClient(p, opts...)
}

func assertWarnings(t *testing.T, got []grail.Warning, want ...grail.Warning) {
	t.Helper()
	if fmt.Sprint(got) != fmt.Sprint(want) {
		t.Fatalf("warnings = %v, want %v", got, want)
	}
}

func TestDeprecationWarnings(t *testing.T) {
	tests := []struct {
		name  string
		model string
		want  []grail.Warning
	}{
		{"current model", "current", nil},
		{"deprecated model", "soon", []grail.Warning{{
			Code: grail.WarningModelDeprecated, Message: "soon is deprecated; shuts down 2027-01-14; use current",
		}}},
		{"deprecated model without a shutdown date", "undated", []grail.Warning{{
			Code: grail.WarningModelDeprecated, Message: "undated is deprecated; use current",
		}}},
		{"shut-down model without auto-migrate", "old", []grail.Warning{{
			Code: grail.WarningModelDeprecated, Message: "old is deprecated; shut down 2026-10-23; use current",
		}}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var calls []string
			client := depClient(routedProvider("p", &calls, nil,
				textModel("current", grail.ModelTierFast, grail.ModelCapabilities{}),
				deprecatedModel("soon", soonShutdown, "current"),
				deprecatedModel("undated", time.Time{}, "current"),
				deprecatedModel("old", oldShutdown, "current"),
			))
			res, err := client.Generate(context.Background(), grail.Request{Inputs: routeText, Output: grail.OutputText(), Model: tt.model})
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			assertCalls(t, calls, "p:"+tt.model)
			assertWarnings(t, res.Warnings, tt.want...)
		})
	}
}

func TestDeprecationWarningForTierResolvedModel(t *testing.T) {
	var calls []string
	client := depClient(routedProvider("p", &calls, nil, deprecatedModel("soon", soonShutdown, "current")))
	res, err := client.Generate(context.Background(), grail.Request{Inputs: routeText, Output: grail.OutputText(), Tier: grail.ModelTierBest})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	assertCalls(t, calls, "p:soon")
	if len(res.Warnings) != 1 || res.Warnings[0].Code != grail.WarningModelDeprecated {
		t.Fatalf("expected one deprecation warning, got %v", res.Warnings)
	}
}

// describingProvider is a catalog provider whose ModelDescriber reports
// describe as the models it uses by default.
type describingProvider struct {
	*mock.CatalogProvider
	describe string
}

func (p describingProvider) DescribeModels(req grail.Request) string {
	if req.Model != "" {
		return req.Model
	}
	return p.describe
}

func TestDeprecationOfProviderDefault(t *testing.T) {
	ctx := context.Background()
	req := grail.Request{Inputs: routeText, Output: grail.OutputText()}
	models := []grail.Model{
		textModel("text", grail.ModelTierBest, grail.ModelCapabilities{}),
		deprecatedModel("soon", soonShutdown, "current"),
		deprecatedModel("old", oldShutdown, "current"),
	}
	describing := func(calls *[]string, describe string) describingProvider {
		return describingProvider{CatalogProvider: routedProvider("p", calls, nil, models...), describe: describe}
	}

	t.Run("warns for a deprecated default", func(t *testing.T) {
		var calls []string
		res, err := depClient(describing(&calls, "soon")).Generate(ctx, req)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		assertCalls(t, calls, "p:")
		assertWarnings(t, res.Warnings, grail.Warning{
			Code: grail.WarningModelDeprecated, Message: "soon is deprecated; shuts down 2027-01-14; use current",
		})
	})

	t.Run("warns only for the deprecated one of several defaults, without migrating", func(t *testing.T) {
		var calls []string
		res, err := depClient(describing(&calls, "text, old"), grail.WithAutoMigrate()).Generate(ctx, req)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		assertCalls(t, calls, "p:")
		assertWarnings(t, res.Warnings, grail.Warning{
			Code: grail.WarningModelDeprecated, Message: "old is deprecated; shut down 2026-10-23; use current",
		})
	})

	t.Run("migrates a single shut-down default", func(t *testing.T) {
		var calls []string
		res, err := depClient(describing(&calls, "old"), grail.WithAutoMigrate()).Generate(ctx, req)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		assertCalls(t, calls, "p:current")
		assertWarnings(t, res.Warnings, grail.Warning{
			Code: grail.WarningModelMigrated, Message: "old shut down 2026-10-23; migrated to current",
		})
	})

	t.Run("skips a provider without ModelDescriber", func(t *testing.T) {
		var calls []string
		// The provider's best model is deprecated, but nothing says it is the default.
		client := depClient(routedProvider("p", &calls, nil, deprecatedModel("old", oldShutdown, "current")), grail.WithAutoMigrate())
		res, err := client.Generate(ctx, req)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		assertCalls(t, calls, "p:")
		assertWarnings(t, res.Warnings)
	})
}

func TestAutoMigrate(t *testing.T) {
	ctx := context.Background()
	req := grail.Request{Inputs: routeText, Output: grail.OutputText(), Model: "old"}
	models := []grail.Model{deprecatedModel("old", oldShutdown, "current"), deprecatedModel("soon", soonShutdown, "current")}

	t.Run("replaces a shut-down model", func(t *testing.T) {
		var calls []string
		client := depClient(routedProvider("p", &calls, nil, models...), grail.WithAutoMigrate())
		res, err := client.Generate(ctx, req)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		assertCalls(t, calls, "p:current")
		assertWarnings(t, res.Warnings, grail.Warning{
			Code: grail.WarningModelMigrated, Message: "old shut down 2026-10-23; migrated to current",
		})
	})

	t.Run("keeps a model that has not shut down", func(t *testing.T) {
		var calls []string
		client := depClient(routedProvider("p", &calls, nil, models...), grail.WithAutoMigrate())
		res, err := client.Generate(ctx, grail.Request{Inputs: routeText, Output: grail.OutputText(), Model: "soon"})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		assertCalls(t, calls, "p:soon")
		if len(res.Warnings) != 1 || res.Warnings[0].Code != grail.WarningModelDeprecated {
			t.Fatalf("expected one deprecation warning, got %v", res.Warnings)
		}
	})

	t.Run("off sends the model unchanged and surfaces the provider error", func(t *testing.T) {
		var calls []string
		gone := grail.NewGrailError(grail.InvalidArgument, "model old does not exist")
		client := depClient(routedProvider("p", &calls, gone, models...))
		_, err := client.Generate(ctx, req)
		if err != gone {
			t.Fatalf("expected the provider's error, got %v", err)
		}
		assertCalls(t, calls, "p:old")
	})

	t.Run("follows a replacement chain one hop", func(t *testing.T) {
		var calls []string
		client := depClient(routedProvider("p", &calls, nil,
			deprecatedModel("old", oldShutdown, "older"),
			deprecatedModel("older", oldShutdown, "current"),
		), grail.WithAutoMigrate())
		res, err := client.Generate(ctx, req)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		assertCalls(t, calls, "p:older")
		assertWarnings(t, res.Warnings,
			grail.Warning{Code: grail.WarningModelMigrated, Message: "old shut down 2026-10-23; migrated to older"},
			grail.Warning{Code: grail.WarningModelDeprecated, Message: "older is deprecated; shut down 2026-10-23; use current"},
		)
	})
}

// knownProvider adds a full catalog of known models to a catalog provider.
type knownProvider struct {
	*mock.CatalogProvider
	known []grail.Model
}

func (p knownProvider) KnownModels() []grail.Model { return p.known }

func TestDeprecationUsesKnownModels(t *testing.T) {
	var calls []string
	// The listed model has no deprecation record; the known catalog does.
	p := knownProvider{
		CatalogProvider: routedProvider("p", &calls, nil, textModel("old", grail.ModelTierBest, grail.ModelCapabilities{})),
		known:           []grail.Model{deprecatedModel("old", oldShutdown, "current")},
	}
	res, err := depClient(p, grail.WithAutoMigrate()).Generate(context.Background(),
		grail.Request{Inputs: routeText, Output: grail.OutputText(), Model: "old"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	assertCalls(t, calls, "p:current")
	if len(res.Warnings) != 1 || res.Warnings[0].Code != grail.WarningModelMigrated {
		t.Fatalf("expected one migration warning, got %v", res.Warnings)
	}
}

// warnRecorder is a slog.Handler that records the messages of Warn records.
type warnRecorder struct {
	mu   sync.Mutex
	msgs []string
}

func (h *warnRecorder) Enabled(context.Context, slog.Level) bool { return true }
func (h *warnRecorder) WithAttrs([]slog.Attr) slog.Handler       { return h }
func (h *warnRecorder) WithGroup(string) slog.Handler            { return h }
func (h *warnRecorder) Handle(_ context.Context, r slog.Record) error {
	if r.Level == slog.LevelWarn {
		h.mu.Lock()
		h.msgs = append(h.msgs, r.Message)
		h.mu.Unlock()
	}
	return nil
}

func TestDeprecationLoggedOncePerModelPerClient(t *testing.T) {
	p := &mock.CatalogProvider{
		Provider: mock.Provider{GenerateFn: func(ctx context.Context, req grail.Request) (grail.Response, error) {
			return grail.Response{}, nil
		}},
		Models: []grail.Model{deprecatedModel("soon", soonShutdown, "current"), deprecatedModel("old", oldShutdown, "current")},
	}
	rec := &warnRecorder{}
	newClient := func() *grail.Client {
		return grail.NewClient(p, grail.WithLogger(slog.New(rec)), grail.WithClock(func() time.Time { return depNow }))
	}

	client := newClient()
	var wg sync.WaitGroup
	for i := range 20 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			model := []string{"soon", "old"}[i%2]
			res, err := client.Generate(context.Background(), grail.Request{Inputs: routeText, Output: grail.OutputText(), Model: model})
			if err != nil || len(res.Warnings) != 1 {
				t.Errorf("Generate(%s) = %v, %v; want one warning", model, res.Warnings, err)
			}
		}()
	}
	wg.Wait()
	if len(rec.msgs) != 2 {
		t.Fatalf("expected one warning log per model, got %q", rec.msgs)
	}

	// A new client logs again.
	if _, err := newClient().Generate(context.Background(), grail.Request{Inputs: routeText, Output: grail.OutputText(), Model: "soon"}); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(rec.msgs) != 3 || !strings.HasPrefix(rec.msgs[2], "soon is deprecated") {
		t.Fatalf("expected a second client to log again, got %q", rec.msgs)
	}
}

// streamingCatalog is a catalog provider that also streams, recording the
// model of each stream in calls.
type streamingCatalog struct {
	*mock.CatalogProvider
	calls *[]string
}

func (p streamingCatalog) DoStream(ctx context.Context, req grail.Request) iter.Seq2[grail.StreamEvent, error] {
	*p.calls = append(*p.calls, p.Name()+":"+req.Model)
	return mock.TextStream(grail.Usage{}, "hi")
}

func TestStreamDeprecation(t *testing.T) {
	ctx := context.Background()
	req := grail.Request{Inputs: routeText, Output: grail.OutputText(), Model: "old"}
	streaming := func(calls *[]string) streamingCatalog {
		return streamingCatalog{CatalogProvider: routedProvider("p", calls, nil, deprecatedModel("old", oldShutdown, "current")), calls: calls}
	}

	t.Run("warns on the final event", func(t *testing.T) {
		var calls []string
		_, final, err := collect(depClient(streaming(&calls)).Stream(ctx, req))
		if err != nil {
			t.Fatalf("Stream: %v", err)
		}
		assertCalls(t, calls, "p:old")
		assertWarnings(t, final.Warnings, grail.Warning{
			Code: grail.WarningModelDeprecated, Message: "old is deprecated; shut down 2026-10-23; use current",
		})
	})

	t.Run("streams the replacement with auto-migrate", func(t *testing.T) {
		var calls []string
		_, final, err := collect(depClient(streaming(&calls), grail.WithAutoMigrate()).Stream(ctx, req))
		if err != nil {
			t.Fatalf("Stream: %v", err)
		}
		assertCalls(t, calls, "p:current")
		assertWarnings(t, final.Warnings, grail.Warning{
			Code: grail.WarningModelMigrated, Message: "old shut down 2026-10-23; migrated to current",
		})
	})
}
