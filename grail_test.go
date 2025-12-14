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
