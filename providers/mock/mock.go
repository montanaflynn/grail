package mock

import (
	"context"
	"errors"

	"github.com/montanaflynn/grail"
)

// Provider is a test double for grail.Provider. Configure the function fields to control behavior.
type Provider struct {
	TextFn  func(ctx context.Context, req grail.TextRequest) (grail.TextResult, error)
	ImageFn func(ctx context.Context, req grail.ImageRequest) (grail.ImageResult, error)
	// Optional configured models; used to satisfy grail.Provider metadata.
	TextModelVal  string
	ImageModelVal string
	// Optional defaults for pointer-based options.
	MaxTokens   *int32
	Temperature *float32
	TopP        *float32
}

func (m *Provider) GenerateText(ctx context.Context, req grail.TextRequest) (grail.TextResult, error) {
	if m.TextFn == nil {
		return grail.TextResult{}, errors.New("mock TextFn not set")
	}
	return m.TextFn(ctx, req)
}

func (m *Provider) GenerateImage(ctx context.Context, req grail.ImageRequest) (grail.ImageResult, error) {
	if m.ImageFn == nil {
		return grail.ImageResult{}, errors.New("mock ImageFn not set")
	}
	return m.ImageFn(ctx, req)
}

func (m *Provider) DefaultTextModel() string {
	if m.TextModelVal != "" {
		return m.TextModelVal
	}
	return "mock-text-model"
}

func (m *Provider) DefaultImageModel() string {
	if m.ImageModelVal != "" {
		return m.ImageModelVal
	}
	return "mock-image-model"
}
