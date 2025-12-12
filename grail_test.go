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
