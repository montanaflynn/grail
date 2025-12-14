// Package mock provides a test double implementation of the grail.Provider interface.
// It's designed for testing code that depends on grail.Provider without making
// actual API calls.
//
// Example usage:
//
//	provider := &mock.Provider{}
//	provider.GenerateFn = func(ctx context.Context, req grail.Request) (grail.Response, error) {
//		return grail.Response{
//			Outputs: []grail.OutputPart{
//				grail.NewTextOutputPart("mock response"),
//			},
//		}, nil
//	}
//	client := grail.NewClient(provider)
//	res, _ := client.Generate(ctx, grail.Request{
//		Inputs: []grail.Input{grail.InputText("test")},
//		Output: grail.OutputText(),
//	})
package mock

import (
	"context"

	"github.com/montanaflynn/grail"
)

// Provider is a test double for grail.Provider. Configure the function fields to control behavior.
type Provider struct {
	GenerateFn func(ctx context.Context, req grail.Request) (grail.Response, error)
	NameVal    string
}

// Name returns the provider name.
func (m *Provider) Name() string {
	if m.NameVal != "" {
		return m.NameVal
	}
	return "mock"
}

// DoGenerate implements the ProviderExecutor interface.
func (m *Provider) DoGenerate(ctx context.Context, req grail.Request) (grail.Response, error) {
	if m.GenerateFn == nil {
		return grail.Response{}, grail.NewGrailError(grail.Internal, "mock GenerateFn not set").WithProviderName("mock")
	}
	return m.GenerateFn(ctx, req)
}
