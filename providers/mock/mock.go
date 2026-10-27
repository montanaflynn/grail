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
	"iter"

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

// StreamProvider is a Provider that also streams text, for testing
// grail.Client.Stream. StreamFn supplies each stream; TextStream builds one
// from fixed chunks.
type StreamProvider struct {
	Provider
	StreamFn func(ctx context.Context, req grail.Request) iter.Seq2[grail.StreamEvent, error]
}

// DoStream implements grail.StreamExecutor.
func (m *StreamProvider) DoStream(ctx context.Context, req grail.Request) iter.Seq2[grail.StreamEvent, error] {
	if m.StreamFn == nil {
		return func(yield func(grail.StreamEvent, error) bool) {
			yield(grail.StreamEvent{}, grail.NewGrailError(grail.Internal, "mock StreamFn not set").WithProviderName(m.Name()))
		}
	}
	return m.StreamFn(ctx, req)
}

// TextStream returns a stream that yields each chunk as a text delta, then a
// final event carrying usage. It stops early if the consumer stops iterating.
func TextStream(usage grail.Usage, chunks ...string) iter.Seq2[grail.StreamEvent, error] {
	return func(yield func(grail.StreamEvent, error) bool) {
		for _, chunk := range chunks {
			if !yield(grail.StreamEvent{Text: chunk}, nil) {
				return
			}
		}
		yield(grail.StreamEvent{Done: true, Usage: usage}, nil)
	}
}
