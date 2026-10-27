// Package auto builds a grail client from whichever provider API keys are
// set in the environment.
//
// Example usage:
//
//	client, err := auto.FromEnv(ctx)
//	if err != nil {
//		log.Fatal(err)
//	}
//	text, err := client.Text(ctx, "Hello, world!")
//
// FromEnv checks these environment variables and adds a provider for each one
// that is set, in this default priority order:
//
//   - ANTHROPIC_API_KEY (anthropic)
//   - OPENAI_API_KEY (openai)
//   - GEMINI_API_KEY (gemini)
//
// The client routes each request to the first provider that can serve it, so
// priority only decides between providers with the needed capabilities.
// Anthropic comes first for text, JSON, image understanding and PDF
// understanding. It can't generate images, so image requests skip it and go
// to OpenAI, which comes next because it covers every capability. Gemini is
// last and serves as a fallback.
//
// Set GRAIL_PROVIDERS to a comma-separated list of provider names (for
// example "gemini,openai") to choose the providers and their order. Every
// listed provider must have its API key set.
//
// This lives in its own package because the provider packages import grail,
// so grail can't import them.
package auto

import (
	"context"
	"errors"
	"fmt"
	"os"
	"strings"

	"github.com/montanaflynn/grail"
	"github.com/montanaflynn/grail/providers/anthropic"
	"github.com/montanaflynn/grail/providers/gemini"
	"github.com/montanaflynn/grail/providers/openai"
)

// ProvidersEnv is the environment variable that overrides which providers
// FromEnv uses and their order.
const ProvidersEnv = "GRAIL_PROVIDERS"

// ErrNoProviders is returned by FromEnv when no provider API key is set.
var ErrNoProviders = errors.New("auto: no provider API key set")

// provider describes how to build one provider from its API key.
type provider struct {
	name string
	env  string
	new  func(ctx context.Context, apiKey string) (grail.Provider, error)
}

// defaultProviders lists the supported providers in default priority order.
var defaultProviders = []provider{
	{"anthropic", "ANTHROPIC_API_KEY", func(_ context.Context, key string) (grail.Provider, error) {
		return anthropic.New(anthropic.WithAPIKey(key))
	}},
	{"openai", "OPENAI_API_KEY", func(_ context.Context, key string) (grail.Provider, error) {
		return openai.New(openai.WithAPIKey(key))
	}},
	{"gemini", "GEMINI_API_KEY", func(ctx context.Context, key string) (grail.Provider, error) {
		return gemini.New(ctx, gemini.WithAPIKey(key))
	}},
}

// FromEnv returns a client with a provider for each API key set in the
// environment, in the default priority order or the order given by
// GRAIL_PROVIDERS. opts are passed to grail.NewClient.
//
// It returns an error wrapping ErrNoProviders when no key is set, and an
// error when GRAIL_PROVIDERS names an unknown provider or one whose key is
// not set. Constructing the providers makes no network calls.
func FromEnv(ctx context.Context, opts ...grail.ClientOption) (*grail.Client, error) {
	ps, err := providersFromEnv(ctx)
	if err != nil {
		return nil, err
	}
	opts = append([]grail.ClientOption{grail.WithProviders(ps[1:]...)}, opts...)
	return grail.NewClient(ps[0], opts...), nil
}

// providersFromEnv constructs the providers FromEnv uses, in priority order.
// It always returns at least one provider or an error.
func providersFromEnv(ctx context.Context) ([]grail.Provider, error) {
	candidates, explicit, err := candidatesFromEnv()
	if err != nil {
		return nil, err
	}

	var ps []grail.Provider
	for _, c := range candidates {
		key := strings.TrimSpace(os.Getenv(c.env))
		if key == "" {
			if explicit {
				return nil, fmt.Errorf("auto: %s lists %q but %s is not set", ProvidersEnv, c.name, c.env)
			}
			continue
		}
		p, err := c.new(ctx, key)
		if err != nil {
			return nil, fmt.Errorf("auto: new %s provider: %w", c.name, err)
		}
		ps = append(ps, p)
	}

	if len(ps) == 0 {
		envs := make([]string, len(candidates))
		for i, c := range candidates {
			envs[i] = c.env
		}
		return nil, fmt.Errorf("%w (checked %s)", ErrNoProviders, strings.Join(envs, ", "))
	}
	return ps, nil
}

// candidatesFromEnv returns the providers to try, in order. explicit reports
// whether they came from GRAIL_PROVIDERS rather than the default list.
func candidatesFromEnv() (candidates []provider, explicit bool, err error) {
	list := strings.TrimSpace(os.Getenv(ProvidersEnv))
	if list == "" {
		return defaultProviders, false, nil
	}

	seen := map[string]bool{}
	for _, name := range strings.Split(list, ",") {
		name = strings.ToLower(strings.TrimSpace(name))
		if name == "" || seen[name] {
			continue
		}
		seen[name] = true
		c, ok := lookupProvider(name)
		if !ok {
			return nil, false, fmt.Errorf("auto: unknown provider %q in %s (known: %s)", name, ProvidersEnv, knownNames())
		}
		candidates = append(candidates, c)
	}
	if len(candidates) == 0 {
		return defaultProviders, false, nil
	}
	return candidates, true, nil
}

func lookupProvider(name string) (provider, bool) {
	for _, c := range defaultProviders {
		if c.name == name {
			return c, true
		}
	}
	return provider{}, false
}

func knownNames() string {
	names := make([]string, len(defaultProviders))
	for i, c := range defaultProviders {
		names[i] = c.name
	}
	return strings.Join(names, ", ")
}
