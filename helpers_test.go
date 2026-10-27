package grail_test

import (
	"bytes"
	"context"
	"errors"
	"testing"

	"github.com/montanaflynn/grail"
	"github.com/montanaflynn/grail/providers/mock"
)

// testProviderOption is a stand-in for a provider package's typed options.
type testProviderOption struct{ name string }

func (testProviderOption) ApplyProviderOption() {}

func TestTierAliases(t *testing.T) {
	if grail.Best != grail.ModelTierBest {
		t.Fatalf("Best = %q, want %q", grail.Best, grail.ModelTierBest)
	}
	if grail.Fast != grail.ModelTierFast {
		t.Fatalf("Fast = %q, want %q", grail.Fast, grail.ModelTierFast)
	}
}

func TestClientText(t *testing.T) {
	ctx := context.Background()
	var got grail.Request
	prov := &mock.Provider{
		GenerateFn: func(ctx context.Context, req grail.Request) (grail.Response, error) {
			got = req
			return grail.Response{
				Outputs: []grail.OutputPart{grail.NewTextOutputPart("hello")},
			}, nil
		},
	}
	client := grail.NewClient(prov)

	text, err := client.Text(ctx, "say hello")
	if err != nil {
		t.Fatalf("Text: %v", err)
	}
	if text != "hello" {
		t.Fatalf("text = %q, want %q", text, "hello")
	}
	if !grail.IsTextOutput(got.Output) {
		t.Fatalf("expected text output, got %T", got.Output)
	}
	if len(got.Inputs) != 1 {
		t.Fatalf("expected 1 input, got %d", len(got.Inputs))
	}
	if s, ok := grail.AsTextInput(got.Inputs[0]); !ok || s != "say hello" {
		t.Fatalf("first input = %q (text=%v), want prompt", s, ok)
	}
	if got.Model != "" || got.Tier != "" || len(got.ProviderOptions) != 0 {
		t.Fatalf("expected no model/tier/provider options by default, got %+v", got)
	}
}

func TestClientImage(t *testing.T) {
	ctx := context.Background()
	want := []byte("first-image")
	var got grail.Request
	prov := &mock.Provider{
		GenerateFn: func(ctx context.Context, req grail.Request) (grail.Response, error) {
			got = req
			return grail.Response{
				Outputs: []grail.OutputPart{
					grail.NewImageOutputPart(want, "image/png", "a.png"),
					grail.NewImageOutputPart([]byte("second-image"), "image/png", "b.png"),
				},
			}, nil
		},
	}
	client := grail.NewClient(prov)

	img, err := client.Image(ctx, "a red fox", grail.WithTier(grail.Fast))
	if err != nil {
		t.Fatalf("Image: %v", err)
	}
	if !bytes.Equal(img, want) {
		t.Fatalf("image = %q, want %q", img, want)
	}
	spec, ok := grail.GetImageSpec(got.Output)
	if !ok {
		t.Fatalf("expected image output, got %T", got.Output)
	}
	if spec.Count != 1 {
		t.Fatalf("image count = %d, want 1", spec.Count)
	}
	if got.Tier != grail.Fast {
		t.Fatalf("tier = %q, want %q", got.Tier, grail.Fast)
	}
}

func TestCallOptions(t *testing.T) {
	ctx := context.Background()
	var got grail.Request
	prov := &mock.Provider{
		GenerateFn: func(ctx context.Context, req grail.Request) (grail.Response, error) {
			got = req
			return grail.Response{
				Outputs: []grail.OutputPart{grail.NewTextOutputPart("ok")},
			}, nil
		},
	}
	client := grail.NewClient(prov)

	pdf := grail.InputPDF([]byte("%PDF-1.4 test"))
	img := grail.InputFile([]byte("png-bytes"), "image/png")
	optA := testProviderOption{name: "a"}
	optB := testProviderOption{name: "b"}

	_, err := client.Text(ctx, "describe these",
		grail.WithTier(grail.Best),
		grail.WithModel("explicit-model"),
		grail.WithInputs(pdf),
		grail.WithInputs(img),
		grail.WithProviderOptions(optA),
		grail.WithProviderOptions(optB),
		nil, // nil options are ignored
	)
	if err != nil {
		t.Fatalf("Text: %v", err)
	}

	if got.Model != "explicit-model" {
		t.Fatalf("model = %q, want %q", got.Model, "explicit-model")
	}
	if got.Tier != grail.Best {
		t.Fatalf("tier = %q, want %q", got.Tier, grail.Best)
	}

	if len(got.Inputs) != 3 {
		t.Fatalf("expected 3 inputs, got %d", len(got.Inputs))
	}
	if s, ok := grail.AsTextInput(got.Inputs[0]); !ok || s != "describe these" {
		t.Fatalf("first input should be the prompt, got %q (text=%v)", s, ok)
	}
	if _, mime, _, ok := grail.AsFileInput(got.Inputs[1]); !ok || mime != "application/pdf" {
		t.Fatalf("second input should be the PDF, got mime=%q ok=%v", mime, ok)
	}
	if _, mime, _, ok := grail.AsFileInput(got.Inputs[2]); !ok || mime != "image/png" {
		t.Fatalf("third input should be the image, got mime=%q ok=%v", mime, ok)
	}

	if len(got.ProviderOptions) != 2 {
		t.Fatalf("expected 2 provider options, got %d", len(got.ProviderOptions))
	}
	if got.ProviderOptions[0] != optA || got.ProviderOptions[1] != optB {
		t.Fatalf("provider options out of order: %+v", got.ProviderOptions)
	}
}

func TestHelpersMissingOutput(t *testing.T) {
	ctx := context.Background()
	// The provider returns the wrong kind of part for each helper.
	prov := &mock.Provider{
		GenerateFn: func(ctx context.Context, req grail.Request) (grail.Response, error) {
			if grail.IsTextOutput(req.Output) {
				return grail.Response{
					Outputs: []grail.OutputPart{grail.NewImageOutputPart([]byte("img"), "image/png", "")},
				}, nil
			}
			return grail.Response{
				Outputs: []grail.OutputPart{grail.NewTextOutputPart("not an image")},
			}, nil
		},
	}
	client := grail.NewClient(prov)

	t.Run("text", func(t *testing.T) {
		text, err := client.Text(ctx, "hi")
		if grail.GetErrorCode(err) != grail.OutputInvalid {
			t.Fatalf("expected output_invalid, got %v", err)
		}
		if text != "" {
			t.Fatalf("expected empty text, got %q", text)
		}
	})

	t.Run("image", func(t *testing.T) {
		img, err := client.Image(ctx, "hi")
		if grail.GetErrorCode(err) != grail.OutputInvalid {
			t.Fatalf("expected output_invalid, got %v", err)
		}
		if img != nil {
			t.Fatalf("expected nil image, got %q", img)
		}
	})
}

func TestHelpersPropagateErrors(t *testing.T) {
	ctx := context.Background()
	providerErr := grail.NewGrailError(grail.RateLimited, "slow down").WithProviderName("mock")
	prov := &mock.Provider{
		GenerateFn: func(ctx context.Context, req grail.Request) (grail.Response, error) {
			return grail.Response{}, providerErr
		},
	}
	client := grail.NewClient(prov)

	if _, err := client.Text(ctx, "hi"); !errors.Is(err, providerErr) {
		t.Fatalf("Text: expected provider error, got %v", err)
	}
	if _, err := client.Image(ctx, "hi"); !errors.Is(err, providerErr) {
		t.Fatalf("Image: expected provider error, got %v", err)
	}
}
