package grail_test

import (
	"context"
	"io"
	"log/slog"
	"strings"
	"testing"

	"github.com/montanaflynn/grail"
	"github.com/montanaflynn/grail/providers/mock"
)

var (
	routeText = []grail.Input{grail.InputText("test")}
	routePNG  = grail.InputImage([]byte("\x89PNG\r\n\x1a\n0000"))
	routePDF  = grail.InputPDF([]byte("%PDF-1.4"))
)

// routedProvider returns a catalog provider named name that records each
// model it is asked to serve in calls and responds with err (nil for success).
func routedProvider(name string, calls *[]string, err error, models ...grail.Model) *mock.CatalogProvider {
	return &mock.CatalogProvider{
		Provider: mock.Provider{
			NameVal: name,
			GenerateFn: func(ctx context.Context, req grail.Request) (grail.Response, error) {
				*calls = append(*calls, name+":"+req.Model)
				if err != nil {
					return grail.Response{}, err
				}
				return grail.Response{Outputs: []grail.OutputPart{grail.NewTextOutputPart("ok")}}, nil
			},
		},
		Models: models,
	}
}

func routedClient(p grail.Provider, ps ...grail.Provider) *grail.Client {
	quiet := slog.New(slog.NewTextHandler(io.Discard, nil))
	return grail.NewClient(p, grail.WithProviders(ps...), grail.WithLogger(quiet))
}

func textModel(name string, tier grail.ModelTier, caps grail.ModelCapabilities) grail.Model {
	caps.TextGeneration = true
	return grail.Model{Name: name, Role: grail.ModelRoleText, Tier: tier, Capabilities: caps}
}

func imageModel(name string, tier grail.ModelTier) grail.Model {
	return grail.Model{Name: name, Role: grail.ModelRoleImage, Tier: tier, Capabilities: grail.ModelCapabilities{ImageGeneration: true}}
}

func assertCalls(t *testing.T, calls []string, want ...string) {
	t.Helper()
	if strings.Join(calls, ",") != strings.Join(want, ",") {
		t.Fatalf("provider calls = %v, want %v", calls, want)
	}
}

func TestRouteByRole(t *testing.T) {
	ctx := context.Background()

	tests := []struct {
		name   string
		output grail.Output
		tier   grail.ModelTier
		want   string
	}{
		{"text goes to first text provider", grail.OutputText(), grail.ModelTierBest, "text:text-best"},
		{"image skips text-only provider", grail.OutputImage(grail.ImageSpec{}), grail.ModelTierBest, "image:image-best"},
		{"image without tier keeps provider default", grail.OutputImage(grail.ImageSpec{}), "", "image:"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var calls []string
			client := routedClient(
				routedProvider("text", &calls, nil, textModel("text-best", grail.ModelTierBest, grail.ModelCapabilities{})),
				routedProvider("image", &calls, nil, imageModel("image-best", grail.ModelTierBest)),
			)
			res, err := client.Generate(ctx, grail.Request{Inputs: routeText, Output: tt.output, Tier: tt.tier})
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			assertCalls(t, calls, tt.want)
			if want := strings.Split(tt.want, ":")[0]; res.Provider.Name != want {
				t.Fatalf("response provider = %q, want %q", res.Provider.Name, want)
			}
			if len(res.Warnings) != 0 {
				t.Fatalf("unexpected warnings: %v", res.Warnings)
			}
		})
	}
}

func TestRouteByInputCapability(t *testing.T) {
	ctx := context.Background()

	tests := []struct {
		name  string
		input grail.Input
		want  string
	}{
		{"plain text stays on primary", grail.InputText("more"), "basic:basic-fast"},
		{"image input needs image understanding", routePNG, "vision:vision-fast"},
		{"PDF input needs PDF understanding", routePDF, "docs:docs-fast"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var calls []string
			client := routedClient(
				routedProvider("basic", &calls, nil, textModel("basic-fast", grail.ModelTierFast, grail.ModelCapabilities{})),
				routedProvider("vision", &calls, nil, textModel("vision-fast", grail.ModelTierFast, grail.ModelCapabilities{ImageUnderstanding: true})),
				routedProvider("docs", &calls, nil, textModel("docs-fast", grail.ModelTierFast, grail.ModelCapabilities{PDFUnderstanding: true})),
			)
			_, err := client.Generate(ctx, grail.Request{
				Inputs: []grail.Input{grail.InputText("describe"), tt.input},
				Output: grail.OutputText(),
				Tier:   grail.ModelTierFast,
			})
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			assertCalls(t, calls, tt.want)
		})
	}
}

func TestRouteNoCapableProvider(t *testing.T) {
	var calls []string
	client := routedClient(
		routedProvider("basic", &calls, nil, textModel("basic-best", grail.ModelTierBest, grail.ModelCapabilities{})),
		routedProvider("vision", &calls, nil, textModel("vision-best", grail.ModelTierBest, grail.ModelCapabilities{ImageUnderstanding: true})),
	)
	_, err := client.Generate(context.Background(), grail.Request{
		Inputs: []grail.Input{routePDF},
		Output: grail.OutputText(),
		Tier:   grail.ModelTierBest,
	})
	if grail.GetErrorCode(err) != grail.Unsupported {
		t.Fatalf("expected unsupported, got %v", err)
	}
	if !strings.Contains(err.Error(), "PDF understanding") {
		t.Fatalf("error should name the missing capability, got %v", err)
	}
	assertCalls(t, calls)
}

func TestFallbackOnRetryableError(t *testing.T) {
	tests := []struct {
		name string
		err  error
	}{
		{"rate limited", grail.NewGrailError(grail.RateLimited, "slow down")},
		{"unavailable", grail.NewGrailError(grail.Unavailable, "down")},
		{"timeout", grail.NewGrailError(grail.Timeout, "too slow")},
		{"marked retryable", grail.NewGrailError(grail.Internal, "flaky").WithRetryable(true)},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var calls []string
			client := routedClient(
				routedProvider("first", &calls, tt.err, textModel("first-best", grail.ModelTierBest, grail.ModelCapabilities{})),
				routedProvider("second", &calls, nil, textModel("second-best", grail.ModelTierBest, grail.ModelCapabilities{})),
			)
			res, err := client.Generate(context.Background(), grail.Request{Inputs: routeText, Output: grail.OutputText(), Tier: grail.ModelTierBest})
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			assertCalls(t, calls, "first:first-best", "second:second-best")
			if res.Provider.Name != "second" {
				t.Fatalf("response provider = %q, want second", res.Provider.Name)
			}
			if len(res.Warnings) != 1 || res.Warnings[0].Code != grail.WarningFallback {
				t.Fatalf("expected one fallback warning, got %v", res.Warnings)
			}
			if msg := res.Warnings[0].Message; !strings.Contains(msg, "first") || !strings.Contains(msg, "second") {
				t.Fatalf("fallback warning should name both providers, got %q", msg)
			}
		})
	}

	t.Run("returns last error when every provider fails", func(t *testing.T) {
		var calls []string
		client := routedClient(
			routedProvider("first", &calls, grail.NewGrailError(grail.RateLimited, "slow down"), textModel("first-best", grail.ModelTierBest, grail.ModelCapabilities{})),
			routedProvider("second", &calls, grail.NewGrailError(grail.Unavailable, "down"), textModel("second-best", grail.ModelTierBest, grail.ModelCapabilities{})),
		)
		_, err := client.Generate(context.Background(), grail.Request{Inputs: routeText, Output: grail.OutputText(), Tier: grail.ModelTierBest})
		if grail.GetErrorCode(err) != grail.Unavailable {
			t.Fatalf("expected unavailable from last provider, got %v", err)
		}
		assertCalls(t, calls, "first:first-best", "second:second-best")
	})

	t.Run("stops when the context is done", func(t *testing.T) {
		ctx, cancel := context.WithCancel(context.Background())
		var calls []string
		first := routedProvider("first", &calls, nil, textModel("first-best", grail.ModelTierBest, grail.ModelCapabilities{}))
		first.GenerateFn = func(ctx context.Context, req grail.Request) (grail.Response, error) {
			calls = append(calls, "first")
			cancel()
			return grail.Response{}, grail.NewGrailError(grail.Timeout, "canceled")
		}
		client := routedClient(first,
			routedProvider("second", &calls, nil, textModel("second-best", grail.ModelTierBest, grail.ModelCapabilities{})),
		)
		if _, err := client.Generate(ctx, grail.Request{Inputs: routeText, Output: grail.OutputText(), Tier: grail.ModelTierBest}); err == nil {
			t.Fatalf("expected error")
		}
		assertCalls(t, calls, "first")
	})
}

func TestNoFallbackOnNonRetryableError(t *testing.T) {
	for _, code := range []grail.ErrorCode{grail.InvalidArgument, grail.Unauthorized, grail.Refused, grail.Internal} {
		t.Run(string(code), func(t *testing.T) {
			var calls []string
			client := routedClient(
				routedProvider("first", &calls, grail.NewGrailError(code, "nope"), textModel("first-best", grail.ModelTierBest, grail.ModelCapabilities{})),
				routedProvider("second", &calls, nil, textModel("second-best", grail.ModelTierBest, grail.ModelCapabilities{})),
			)
			_, err := client.Generate(context.Background(), grail.Request{Inputs: routeText, Output: grail.OutputText(), Tier: grail.ModelTierBest})
			if grail.GetErrorCode(err) != code {
				t.Fatalf("expected %s, got %v", code, err)
			}
			assertCalls(t, calls, "first:first-best")
		})
	}
}

func TestExplicitModelRouting(t *testing.T) {
	ctx := context.Background()
	newClient := func(calls *[]string, firstErr error) *grail.Client {
		return routedClient(
			routedProvider("first", calls, firstErr, textModel("first-best", grail.ModelTierBest, grail.ModelCapabilities{})),
			routedProvider("second", calls, nil, textModel("second-best", grail.ModelTierBest, grail.ModelCapabilities{})),
		)
	}

	t.Run("goes to the provider whose catalog lists the model", func(t *testing.T) {
		var calls []string
		if _, err := newClient(&calls, nil).Generate(ctx, grail.Request{Inputs: routeText, Output: grail.OutputText(), Model: "second-best"}); err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		assertCalls(t, calls, "second:second-best")
	})

	t.Run("unknown model goes to the primary provider", func(t *testing.T) {
		var calls []string
		if _, err := newClient(&calls, nil).Generate(ctx, grail.Request{Inputs: routeText, Output: grail.OutputText(), Model: "brand-new"}); err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		assertCalls(t, calls, "first:brand-new")
	})

	t.Run("disables fallback", func(t *testing.T) {
		var calls []string
		client := newClient(&calls, grail.NewGrailError(grail.RateLimited, "slow down"))
		_, err := client.Generate(ctx, grail.Request{Inputs: routeText, Output: grail.OutputText(), Model: "first-best"})
		if grail.GetErrorCode(err) != grail.RateLimited {
			t.Fatalf("expected rate_limited, got %v", err)
		}
		assertCalls(t, calls, "first:first-best")
	})
}

func TestWithProvidersSkipsNonExecutors(t *testing.T) {
	var calls []string
	client := routedClient(nameOnlyProvider{},
		routedProvider("real", &calls, nil, textModel("real-best", grail.ModelTierBest, grail.ModelCapabilities{})),
	)
	if _, err := client.Generate(context.Background(), grail.Request{Inputs: routeText, Output: grail.OutputText()}); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	assertCalls(t, calls, "real:")
}

func TestModelsAggregateAcrossProviders(t *testing.T) {
	ctx := context.Background()
	var calls []string
	client := routedClient(
		&mock.Provider{NameVal: "unlisted"},
		routedProvider("text", &calls, nil,
			textModel("text-best", grail.ModelTierBest, grail.ModelCapabilities{}),
			textModel("shared", grail.ModelTierFast, grail.ModelCapabilities{}),
		),
		routedProvider("image", &calls, nil,
			imageModel("image-best", grail.ModelTierBest),
			textModel("shared", grail.ModelTierBest, grail.ModelCapabilities{}),
		),
	)

	models, err := client.ListModels(ctx)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	var names []string
	for _, m := range models {
		names = append(names, m.Name)
	}
	if got := strings.Join(names, ","); got != "text-best,shared,image-best" {
		t.Fatalf("ListModels = %s, want text-best,shared,image-best", got)
	}

	m, err := client.GetModel(ctx, grail.ModelRoleImage, grail.ModelTierBest)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if m.Name != "image-best" {
		t.Fatalf("GetModel = %q, want image-best from the second provider", m.Name)
	}
}
