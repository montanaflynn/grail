package grail_test

import (
	"context"
	"errors"
	"fmt"
	"testing"

	"github.com/montanaflynn/grail"
	"github.com/montanaflynn/grail/providers/mock"
)

func TestGenerateTextValidation(t *testing.T) {
	ctx := context.Background()
	prov := &mock.Provider{
		TextFn: func(ctx context.Context, req grail.TextRequest) (grail.TextResult, error) {
			t.Fatalf("provider should not be called for invalid input")
			return grail.TextResult{}, nil
		},
		ImageFn: func(ctx context.Context, req grail.ImageRequest) (grail.ImageResult, error) {
			return grail.ImageResult{}, nil
		},
	}

	client := grail.NewClient(prov)

	_, err := client.GenerateText(ctx, grail.TextRequest{})
	if !grail.IsCode(err, grail.CodeInvalidInput) {
		t.Fatalf("expected invalid_input, got %v", err)
	}
}

func TestGenerateTextTemperatureTopPConflict(t *testing.T) {
	ctx := context.Background()

	prov := &mock.Provider{
		TextFn: func(ctx context.Context, req grail.TextRequest) (grail.TextResult, error) {
			t.Fatalf("provider should not be called for bad options")
			return grail.TextResult{}, nil
		},
		ImageFn: func(ctx context.Context, req grail.ImageRequest) (grail.ImageResult, error) {
			return grail.ImageResult{}, nil
		},
	}

	client := grail.NewClient(prov)
	temp := grail.Pointer[float32](0.5)
	topP := grail.Pointer[float32](0.9)

	_, err := client.GenerateText(ctx, grail.TextRequest{
		Input: []grail.Part{grail.Text("hi")},
		Options: grail.TextOptions{
			Temperature: temp,
			TopP:        topP,
		},
	})
	if !grail.IsCode(err, grail.CodeBadOptions) {
		t.Fatalf("expected bad_options, got %v", err)
	}
}

func TestGenerateImageValidation(t *testing.T) {
	ctx := context.Background()
	prov := &mock.Provider{
		TextFn: func(ctx context.Context, req grail.TextRequest) (grail.TextResult, error) {
			return grail.TextResult{}, nil
		},
		ImageFn: func(ctx context.Context, req grail.ImageRequest) (grail.ImageResult, error) {
			t.Fatalf("provider should not be called for invalid input")
			return grail.ImageResult{}, nil
		},
	}

	client := grail.NewClient(prov)

	_, err := client.GenerateImage(ctx, grail.ImageRequest{})
	if !grail.IsCode(err, grail.CodeInvalidInput) {
		t.Fatalf("expected invalid_input, got %v", err)
	}
}

func TestNewErrorAndAccessors(t *testing.T) {
	root := errors.New("boom")
	meta := map[string]any{"field": "input"}

	err := grail.InvalidInput("empty input", grail.WithCause(root), grail.WithMetadata(meta))
	if err.Code != grail.CodeInvalidInput {
		t.Fatalf("expected code %s, got %s", grail.CodeInvalidInput, err.Code)
	}
	if err.Message != "empty input" {
		t.Fatalf("unexpected message: %s", err.Message)
	}
	if err.Cause != root {
		t.Fatalf("cause not set")
	}
	if err.Metadata["field"] != "input" {
		t.Fatalf("metadata not copied")
	}

	if got := grail.GetErrorCode(err); got != grail.CodeInvalidInput {
		t.Fatalf("GetErrorCode(err) mismatch: %s", got)
	}
	if got := grail.GetErrorCode(fmt.Errorf("wrap: %w", err)); got != grail.CodeInvalidInput {
		t.Fatalf("wrapped code mismatch: %s", got)
	}

	if !grail.IsCode(err, grail.CodeInvalidInput) {
		t.Fatalf("IsCode should be true")
	}
	if grail.IsCode(err, grail.CodeUnsupported) {
		t.Fatalf("IsCode should be false for other code")
	}
}

func TestAsError(t *testing.T) {
	err := grail.BadOptions("bad", grail.WithCause(errors.New("root")))
	_, ok := grail.AsError(err)
	if !ok {
		t.Fatalf("expected AsError to succeed")
	}

	_, ok = grail.AsError(errors.New("plain"))
	if ok {
		t.Fatalf("expected AsError to fail for non-Error")
	}
}

func TestIsWithCodeMatching(t *testing.T) {
	target := &grail.Error{Code: grail.CodeUnsupported}
	wrapped := fmt.Errorf("outer: %w", grail.Unsupported("nope"))

	if !errors.Is(wrapped, target) {
		t.Fatalf("errors.Is should match on code")
	}
}

func TestPDFPart(t *testing.T) {
	t.Run("PDF helper creates PDFPart", func(t *testing.T) {
		data := []byte("fake pdf content")
		part := grail.PDF(data, "application/pdf")
		if part.Kind() != grail.PartPDF {
			t.Fatalf("expected PartPDF, got %v", part.Kind())
		}
		pdfPart, ok := part.(grail.PDFPart)
		if !ok {
			t.Fatalf("expected PDFPart type")
		}
		if len(pdfPart.Data) != len(data) {
			t.Fatalf("data mismatch")
		}
		if pdfPart.MIME != "application/pdf" {
			t.Fatalf("expected application/pdf, got %s", pdfPart.MIME)
		}
	})

	t.Run("PDF with empty MIME defaults", func(t *testing.T) {
		data := []byte("fake pdf")
		part := grail.PDF(data, "")
		pdfPart := part.(grail.PDFPart)
		if pdfPart.MIME != "" {
			t.Fatalf("expected empty MIME when not provided")
		}
	})
}

func TestPDFValidation(t *testing.T) {
	ctx := context.Background()
	prov := &mock.Provider{
		TextFn: func(ctx context.Context, req grail.TextRequest) (grail.TextResult, error) {
			t.Fatalf("provider should not be called for invalid PDF input")
			return grail.TextResult{}, nil
		},
		ImageFn: func(ctx context.Context, req grail.ImageRequest) (grail.ImageResult, error) {
			return grail.ImageResult{}, nil
		},
	}

	client := grail.NewClient(prov)

	t.Run("empty PDF data rejected", func(t *testing.T) {
		_, err := client.GenerateText(ctx, grail.TextRequest{
			Input: []grail.Part{grail.PDF([]byte{}, "application/pdf")},
		})
		if !grail.IsCode(err, grail.CodeInvalidInput) {
			t.Fatalf("expected invalid_input for empty PDF, got %v", err)
		}
	})

	t.Run("oversized PDF rejected", func(t *testing.T) {
		oversized := make([]byte, grail.MaxPDFSize+1)
		_, err := client.GenerateText(ctx, grail.TextRequest{
			Input: []grail.Part{grail.PDF(oversized, "application/pdf")},
		})
		if !grail.IsCode(err, grail.CodeInvalidInput) {
			t.Fatalf("expected invalid_input for oversized PDF, got %v", err)
		}
	})

	t.Run("invalid MIME type rejected", func(t *testing.T) {
		data := []byte("fake pdf")
		_, err := client.GenerateText(ctx, grail.TextRequest{
			Input: []grail.Part{grail.PDF(data, "text/plain")},
		})
		if !grail.IsCode(err, grail.CodeInvalidInput) {
			t.Fatalf("expected invalid_input for invalid MIME, got %v", err)
		}
	})

	t.Run("valid PDF accepted", func(t *testing.T) {
		data := []byte("fake pdf content")
		prov.TextFn = func(ctx context.Context, req grail.TextRequest) (grail.TextResult, error) {
			if len(req.Input) != 1 {
				t.Fatalf("expected 1 part")
			}
			if _, ok := req.Input[0].(grail.PDFPart); !ok {
				t.Fatalf("expected PDFPart")
			}
			return grail.TextResult{Text: "ok"}, nil
		}
		res, err := client.GenerateText(ctx, grail.TextRequest{
			Input: []grail.Part{grail.PDF(data, "application/pdf")},
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if res.Text != "ok" {
			t.Fatalf("unexpected result")
		}
	})

	t.Run("PDF with default MIME accepted", func(t *testing.T) {
		data := []byte("fake pdf")
		prov.TextFn = func(ctx context.Context, req grail.TextRequest) (grail.TextResult, error) {
			return grail.TextResult{Text: "ok"}, nil
		}
		_, err := client.GenerateText(ctx, grail.TextRequest{
			Input: []grail.Part{grail.PDF(data, "")},
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
	})
}
