package grail_test

import (
	"context"
	"errors"
	"fmt"
	"testing"

	"github.com/montanaflynn/grail"
	"github.com/montanaflynn/grail/providers/mock"
)

func TestGenerateValidation(t *testing.T) {
	ctx := context.Background()
	prov := &mock.Provider{
		GenerateFn: func(ctx context.Context, req grail.Request) (grail.Response, error) {
			t.Fatalf("provider should not be called for invalid input")
			return grail.Response{}, nil
		},
	}

	client := grail.NewClient(prov)

	t.Run("empty inputs rejected", func(t *testing.T) {
		_, err := client.Generate(ctx, grail.Request{
			Output: grail.OutputText(),
		})
		if grail.GetErrorCode(err) != grail.InvalidArgument {
			t.Fatalf("expected invalid_argument, got %v", err)
		}
	})

	t.Run("missing output rejected", func(t *testing.T) {
		_, err := client.Generate(ctx, grail.Request{
			Inputs: []grail.Input{grail.InputText("test")},
		})
		if grail.GetErrorCode(err) != grail.InvalidArgument {
			t.Fatalf("expected invalid_argument, got %v", err)
		}
	})
}

func TestGenerateText(t *testing.T) {
	ctx := context.Background()
	prov := &mock.Provider{
		GenerateFn: func(ctx context.Context, req grail.Request) (grail.Response, error) {
			return grail.Response{
				Outputs: []grail.OutputPart{
					grail.NewTextOutputPart("test response"),
				},
			}, nil
		},
	}

	client := grail.NewClient(prov)

	res, err := client.Generate(ctx, grail.Request{
		Inputs: []grail.Input{grail.InputText("test")},
		Output: grail.OutputText(),
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	text, ok := res.Text()
	if !ok {
		t.Fatalf("expected text output")
	}
	if text != "test response" {
		t.Fatalf("expected 'test response', got %q", text)
	}
}

func TestGenerateImage(t *testing.T) {
	ctx := context.Background()
	prov := &mock.Provider{
		GenerateFn: func(ctx context.Context, req grail.Request) (grail.Response, error) {
			return grail.Response{
				Outputs: []grail.OutputPart{
					grail.NewImageOutputPart([]byte("fake image"), "image/png", ""),
				},
			}, nil
		},
	}

	client := grail.NewClient(prov)

	res, err := client.Generate(ctx, grail.Request{
		Inputs: []grail.Input{grail.InputText("generate an image")},
		Output: grail.OutputImage(grail.ImageSpec{Count: 1}),
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	images, ok := res.Images()
	if !ok {
		t.Fatalf("expected image output")
	}
	if len(images) != 1 {
		t.Fatalf("expected 1 image, got %d", len(images))
	}
}

func TestGrailError(t *testing.T) {
	root := errors.New("boom")

	err := grail.NewGrailError(grail.InvalidArgument, "empty input").WithCause(root)
	if err.Code() != grail.InvalidArgument {
		t.Fatalf("expected code %s, got %s", grail.InvalidArgument, err.Code())
	}
	if err.Error() == "" {
		t.Fatalf("error message should not be empty")
	}

	if got := grail.GetErrorCode(err); got != grail.InvalidArgument {
		t.Fatalf("GetErrorCode(err) mismatch: %s", got)
	}
	if got := grail.GetErrorCode(fmt.Errorf("wrap: %w", err)); got != grail.InvalidArgument {
		t.Fatalf("wrapped code mismatch: %s", got)
	}

	// RateLimited should be retryable (via default logic in IsRetryable)
	rateLimitedErr := grail.NewGrailError(grail.RateLimited, "rate limited")
	if !grail.IsRetryable(rateLimitedErr) {
		t.Fatalf("rate limited should be retryable")
	}
	// InvalidArgument should not be retryable
	invalidErr := grail.NewGrailError(grail.InvalidArgument, "invalid")
	if grail.IsRetryable(invalidErr) {
		t.Fatalf("invalid argument should not be retryable")
	}
}

func TestPDFInput(t *testing.T) {
	t.Run("PDF helper creates FileInput", func(t *testing.T) {
		data := []byte("fake pdf content")
		input := grail.InputPDF(data)
		data2, mime, _, ok := grail.AsFileInput(input)
		if !ok {
			t.Fatalf("expected FileInput type")
		}
		if len(data2) != len(data) {
			t.Fatalf("data mismatch")
		}
		if mime != "application/pdf" {
			t.Fatalf("expected application/pdf, got %s", mime)
		}
	})

	t.Run("PDF with filename", func(t *testing.T) {
		data := []byte("fake pdf")
		input := grail.InputPDF(data, grail.WithFileName("test.pdf"))
		_, _, name, ok := grail.AsFileInput(input)
		if !ok {
			t.Fatalf("expected FileInput type")
		}
		if name != "test.pdf" {
			t.Fatalf("expected filename 'test.pdf', got %q", name)
		}
	})
}

func TestPDFValidation(t *testing.T) {
	ctx := context.Background()
	prov := &mock.Provider{
		GenerateFn: func(ctx context.Context, req grail.Request) (grail.Response, error) {
			t.Fatalf("provider should not be called for invalid PDF input")
			return grail.Response{}, nil
		},
	}

	client := grail.NewClient(prov)

	t.Run("empty PDF data rejected", func(t *testing.T) {
		_, err := client.Generate(ctx, grail.Request{
			Inputs: []grail.Input{grail.InputPDF([]byte{})},
			Output: grail.OutputText(),
		})
		if grail.GetErrorCode(err) != grail.InvalidArgument {
			t.Fatalf("expected invalid_argument for empty PDF, got %v", err)
		}
	})

	t.Run("oversized PDF rejected", func(t *testing.T) {
		oversized := make([]byte, grail.MaxPDFSize+1)
		_, err := client.Generate(ctx, grail.Request{
			Inputs: []grail.Input{grail.InputPDF(oversized)},
			Output: grail.OutputText(),
		})
		if grail.GetErrorCode(err) != grail.InvalidArgument {
			t.Fatalf("expected invalid_argument for oversized PDF, got %v", err)
		}
	})

	t.Run("valid PDF accepted", func(t *testing.T) {
		data := []byte("fake pdf content")
		prov.GenerateFn = func(ctx context.Context, req grail.Request) (grail.Response, error) {
			if len(req.Inputs) != 1 {
				t.Fatalf("expected 1 input")
			}
			data2, mime, _, ok := grail.AsFileInput(req.Inputs[0])
			if !ok {
				t.Fatalf("expected FileInput")
			}
			if mime != "application/pdf" {
				t.Fatalf("expected PDF MIME")
			}
			if len(data2) != len(data) {
				t.Fatalf("data mismatch")
			}
			return grail.Response{
				Outputs: []grail.OutputPart{
					grail.NewTextOutputPart("ok"),
				},
			}, nil
		}
		res, err := client.Generate(ctx, grail.Request{
			Inputs: []grail.Input{grail.InputPDF(data)},
			Output: grail.OutputText(),
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		text, _ := res.Text()
		if text != "ok" {
			t.Fatalf("unexpected result")
		}
	})
}

func TestImageInput(t *testing.T) {
	t.Run("valid image data", func(t *testing.T) {
		// PNG magic bytes
		data := []byte{0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A}
		input := grail.InputImage(data)
		data2, mime, _, ok := grail.AsFileInput(input)
		if !ok {
			t.Fatalf("expected FileInput type")
		}
		if len(data2) != len(data) {
			t.Fatalf("data mismatch")
		}
		// MIME will be empty (sniffed at validation time)
		if mime != "" {
			t.Fatalf("expected empty MIME (will be sniffed), got %s", mime)
		}
	})

	t.Run("invalid image data - validation at Generate time", func(t *testing.T) {
		data := []byte("not an image")
		input := grail.InputImage(data)
		// ImageInput doesn't error - validation happens at Generate time
		prov := &mock.Provider{
			GenerateFn: func(ctx context.Context, req grail.Request) (grail.Response, error) {
				// Provider should not be called - validation should fail first
				t.Fatalf("provider should not be called")
				return grail.Response{}, nil
			},
		}
		client := grail.NewClient(prov)
		_, err := client.Generate(context.Background(), grail.Request{
			Inputs: []grail.Input{input},
			Output: grail.OutputText(),
		})
		if err == nil {
			t.Fatalf("expected error for invalid image data")
		}
		if grail.GetErrorCode(err) != grail.InvalidArgument {
			t.Fatalf("expected invalid_argument, got %v", err)
		}
	})
}

func TestResponseHelpers(t *testing.T) {
	t.Run("Text helper", func(t *testing.T) {
		res := grail.Response{
			Outputs: []grail.OutputPart{
				grail.NewTextOutputPart("hello"),
			},
		}
		text, ok := res.Text()
		if !ok {
			t.Fatalf("expected text")
		}
		if text != "hello" {
			t.Fatalf("expected 'hello', got %q", text)
		}
	})

	t.Run("Images helper", func(t *testing.T) {
		res := grail.Response{
			Outputs: []grail.OutputPart{
				grail.NewImageOutputPart([]byte("img1"), "image/png", ""),
				grail.NewImageOutputPart([]byte("img2"), "image/jpeg", ""),
			},
		}
		images, ok := res.Images()
		if !ok {
			t.Fatalf("expected images")
		}
		if len(images) != 2 {
			t.Fatalf("expected 2 images, got %d", len(images))
		}
	})

	t.Run("DecodeJSON helper", func(t *testing.T) {
		res := grail.Response{
			Outputs: []grail.OutputPart{
				grail.NewJSONOutputPart([]byte(`{"key":"value"}`)),
			},
		}
		var result map[string]string
		err := res.DecodeJSON(&result)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if result["key"] != "value" {
			t.Fatalf("expected value, got %q", result["key"])
		}
	})
}

// catalogProvider is a mock provider with a model catalog, tier resolution
// and model listing, used to exercise client-side provider selection.
type catalogProvider struct {
	mock.Provider
	models  []grail.Model
	listCtx context.Context
}

func (p *catalogProvider) ResolveModel(role grail.ModelRole, tier grail.ModelTier) (string, error) {
	for _, m := range p.models {
		if m.Role == role && m.Tier == tier {
			return m.Name, nil
		}
	}
	return "", fmt.Errorf("no %s model for tier %s", role, tier)
}

func (p *catalogProvider) ListModels(ctx context.Context) ([]grail.Model, error) {
	p.listCtx = ctx
	return p.models, nil
}

func newCatalogProvider(gotModel *string) *catalogProvider {
	return &catalogProvider{
		Provider: mock.Provider{
			GenerateFn: func(ctx context.Context, req grail.Request) (grail.Response, error) {
				*gotModel = req.Model
				return grail.Response{Outputs: []grail.OutputPart{grail.NewTextOutputPart("ok")}}, nil
			},
		},
		models: []grail.Model{
			{Name: "text-best", Role: grail.ModelRoleText, Tier: grail.ModelTierBest, Capabilities: grail.ModelCapabilities{TextGeneration: true, ImageUnderstanding: true}},
			{Name: "text-fast", Role: grail.ModelRoleText, Tier: grail.ModelTierFast, Capabilities: grail.ModelCapabilities{TextGeneration: true}},
			{Name: "image-best", Role: grail.ModelRoleImage, Tier: grail.ModelTierBest, Capabilities: grail.ModelCapabilities{ImageGeneration: true}},
		},
	}
}

// nameOnlyProvider implements Provider but not ProviderExecutor.
type nameOnlyProvider struct{}

func (nameOnlyProvider) Name() string { return "name-only" }

func TestNewClientConcrete(t *testing.T) {
	var client *grail.Client = grail.NewClient(&mock.Provider{})
	if client == nil {
		t.Fatalf("expected non-nil client")
	}

	t.Run("non-executor provider reports internal error", func(t *testing.T) {
		client := grail.NewClient(nameOnlyProvider{})
		_, err := client.Generate(context.Background(), grail.Request{
			Inputs: []grail.Input{grail.InputText("test")},
			Output: grail.OutputText(),
		})
		if grail.GetErrorCode(err) != grail.Internal {
			t.Fatalf("expected internal error from Generate, got %v", err)
		}
		if _, err := client.ListModels(context.Background()); grail.GetErrorCode(err) != grail.Internal {
			t.Fatalf("expected internal error from ListModels, got %v", err)
		}
	})
}

func TestGenerateModelSelection(t *testing.T) {
	ctx := context.Background()
	text := []grail.Input{grail.InputText("test")}

	tests := []struct {
		name string
		req  grail.Request
		want string
	}{
		{"tier best resolves text model", grail.Request{Inputs: text, Output: grail.OutputText(), Tier: grail.ModelTierBest}, "text-best"},
		{"tier fast resolves text model", grail.Request{Inputs: text, Output: grail.OutputText(), Tier: grail.ModelTierFast}, "text-fast"},
		{"tier resolves by output role", grail.Request{Inputs: text, Output: grail.OutputImage(grail.ImageSpec{}), Tier: grail.ModelTierBest}, "image-best"},
		{"explicit model wins over tier", grail.Request{Inputs: text, Output: grail.OutputText(), Model: "text-fast", Tier: grail.ModelTierBest}, "text-fast"},
		{"no model or tier leaves provider default", grail.Request{Inputs: text, Output: grail.OutputText()}, ""},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := "unset"
			client := grail.NewClient(newCatalogProvider(&got))
			if _, err := client.Generate(ctx, tt.req); err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if got != tt.want {
				t.Fatalf("provider got model %q, want %q", got, tt.want)
			}
		})
	}

	t.Run("unresolvable tier is unsupported", func(t *testing.T) {
		got := "unset"
		client := grail.NewClient(newCatalogProvider(&got))
		_, err := client.Generate(ctx, grail.Request{Inputs: text, Output: grail.OutputImage(grail.ImageSpec{}), Tier: grail.ModelTierFast})
		if grail.GetErrorCode(err) != grail.Unsupported {
			t.Fatalf("expected unsupported, got %v", err)
		}
		if got != "unset" {
			t.Fatalf("provider should not be called")
		}
	})
}

func TestGenerateValidatesCapabilities(t *testing.T) {
	png := []byte("\x89PNG\r\n\x1a\n0000")
	ctx := context.Background()

	t.Run("rejects image input for model without image understanding", func(t *testing.T) {
		got := "unset"
		client := grail.NewClient(newCatalogProvider(&got))
		_, err := client.Generate(ctx, grail.Request{
			Inputs: []grail.Input{grail.InputText("describe"), grail.InputImage(png)},
			Output: grail.OutputText(),
			Tier:   grail.ModelTierFast,
		})
		if grail.GetErrorCode(err) != grail.Unsupported {
			t.Fatalf("expected unsupported, got %v", err)
		}
		if got != "unset" {
			t.Fatalf("provider should not be called")
		}
	})

	t.Run("accepts image input for capable model", func(t *testing.T) {
		got := "unset"
		client := grail.NewClient(newCatalogProvider(&got))
		_, err := client.Generate(ctx, grail.Request{
			Inputs: []grail.Input{grail.InputText("describe"), grail.InputImage(png)},
			Output: grail.OutputText(),
			Tier:   grail.ModelTierBest,
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if got != "text-best" {
			t.Fatalf("provider got model %q, want text-best", got)
		}
	})

	t.Run("rejects text output for image-only model", func(t *testing.T) {
		got := "unset"
		client := grail.NewClient(newCatalogProvider(&got))
		_, err := client.Generate(ctx, grail.Request{
			Inputs: []grail.Input{grail.InputText("test")},
			Output: grail.OutputText(),
			Model:  "image-best",
		})
		if grail.GetErrorCode(err) != grail.InvalidArgument {
			t.Fatalf("expected invalid_argument, got %v", err)
		}
	})

	t.Run("uses caller context for model listing", func(t *testing.T) {
		type ctxKey struct{}
		callerCtx := context.WithValue(ctx, ctxKey{}, "caller")
		got := "unset"
		prov := newCatalogProvider(&got)
		client := grail.NewClient(prov)
		if _, err := client.Generate(callerCtx, grail.Request{
			Inputs: []grail.Input{grail.InputText("test")},
			Output: grail.OutputText(),
			Tier:   grail.ModelTierBest,
		}); err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if prov.listCtx == nil || prov.listCtx.Value(ctxKey{}) != "caller" {
			t.Fatalf("ListModels did not receive the caller's context")
		}
	})
}

func TestClientModels(t *testing.T) {
	ctx := context.Background()

	t.Run("ListModels and GetModel use the provider catalog", func(t *testing.T) {
		got := ""
		client := grail.NewClient(newCatalogProvider(&got))
		models, err := client.ListModels(ctx)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if len(models) != 3 {
			t.Fatalf("expected 3 models, got %d", len(models))
		}
		m, err := client.GetModel(ctx, grail.ModelRoleImage, grail.ModelTierBest)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if m.Name != "image-best" {
			t.Fatalf("expected image-best, got %q", m.Name)
		}
		if _, err := client.GetModel(ctx, grail.ModelRoleImage, grail.ModelTierFast); grail.GetErrorCode(err) != grail.Unsupported {
			t.Fatalf("expected unsupported for missing model, got %v", err)
		}
	})

	t.Run("ListModels is unsupported without a lister", func(t *testing.T) {
		client := grail.NewClient(&mock.Provider{})
		if _, err := client.ListModels(ctx); grail.GetErrorCode(err) != grail.Unsupported {
			t.Fatalf("expected unsupported, got %v", err)
		}
	})
}
