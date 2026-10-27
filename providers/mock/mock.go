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
	"fmt"

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

// CatalogProvider is a Provider with a model catalog. It resolves tiers and
// lists models from Models, so a client can route requests by role and
// capability. Return errors from GenerateFn to exercise fallback.
type CatalogProvider struct {
	Provider
	Models []grail.Model
}

// ResolveModel implements grail.ModelResolver. It returns the first model in
// Models with the given role and tier.
func (m *CatalogProvider) ResolveModel(role grail.ModelRole, tier grail.ModelTier) (string, error) {
	for _, model := range m.Models {
		if model.Role == role && model.Tier == tier {
			return model.Name, nil
		}
	}
	return "", grail.NewGrailError(grail.Unsupported, fmt.Sprintf("mock: no %s model with tier %s", role, tier)).WithProviderName(m.Name())
}

// ListModels implements grail.ModelLister.
func (m *CatalogProvider) ListModels(ctx context.Context) ([]grail.Model, error) {
	return m.Models, nil
}
