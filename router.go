package grail

import (
	"context"
	"fmt"
	"log/slog"
	"slices"
	"strings"
)

// WarningFallback is the Warning.Code added to a response when a provider
// failed with a retryable error and a later provider served the request.
const WarningFallback = "fallback"

// WithProviders adds providers after the one passed to NewClient, in priority
// order. Each request goes to the first provider that can serve it. If that
// provider fails with a retryable error (see IsRetryable), the client falls
// back to the next capable provider. Providers that don't implement
// ProviderExecutor are ignored.
func WithProviders(ps ...Provider) ClientOption {
	return clientOptFunc(func(co *clientOpt) {
		co.providers = append(co.providers, ps...)
	})
}

// route is one provider that can serve a request, with the request's model
// resolved for that provider.
type route struct {
	provider ProviderExecutor
	req      Request
}

// capability is a model capability a request needs.
type capability struct {
	name string
	has  func(ModelCapabilities) bool
}

var (
	capTextGeneration     = capability{"text generation", func(c ModelCapabilities) bool { return c.TextGeneration }}
	capImageGeneration    = capability{"image generation", func(c ModelCapabilities) bool { return c.ImageGeneration }}
	capJSONOutput         = capability{"JSON output", func(c ModelCapabilities) bool { return c.JSONOutput }}
	capImageUnderstanding = capability{"image understanding", func(c ModelCapabilities) bool { return c.ImageUnderstanding }}
	capPDFUnderstanding   = capability{"PDF understanding", func(c ModelCapabilities) bool { return c.PDFUnderstanding }}
)

// routes returns the providers that can serve req, in the order to try them.
//
// An explicit req.Model yields a single route: the first provider whose
// catalog lists the model, or the primary provider if none does (it may be a
// custom or newly released model). Otherwise a provider is a candidate when
// it resolves a model for the request's role and tier, and that model (when
// it appears in the provider's catalog) has the capabilities the inputs and
// output need. Without a tier the provider keeps its default model, and the
// check uses its best (or else fast) model for the role. Providers that can't
// resolve models are assumed capable.
func (c *Client) routes(ctx context.Context, req Request) ([]route, error) {
	if len(c.providers) == 0 {
		return nil, NewGrailError(Internal, "provider executor not available")
	}

	if req.Model != "" {
		provider := c.providers[0]
		for _, p := range c.providers {
			if _, ok := lookupModel(ctx, p, req.Model); ok {
				provider = p
				break
			}
		}
		if err := validateModelCapabilities(ctx, provider, req); err != nil {
			return nil, err
		}
		return []route{{provider: provider, req: req}}, nil
	}

	role := roleFromOutput(req.Output)
	need := requiredCapabilities(req)

	var routes []route
	var missing, skipped []string
	for _, p := range c.providers {
		resolved, lacking, reason := resolveRoute(ctx, p, req, role, need)
		if len(lacking) == 0 {
			routes = append(routes, route{provider: p, req: resolved})
			continue
		}
		for _, name := range lacking {
			if !slices.Contains(missing, name) {
				missing = append(missing, name)
			}
		}
		skipped = append(skipped, p.Name()+": "+reason)
	}

	if len(routes) == 0 {
		return nil, NewGrailError(Unsupported, fmt.Sprintf("no provider supports %s (%s)",
			strings.Join(missing, ", "), strings.Join(skipped, "; ")))
	}
	return routes, nil
}

// resolveRoute resolves req's model for p and checks it against need. It
// returns the resolved request, or the capabilities p lacks and why.
func resolveRoute(ctx context.Context, p ProviderExecutor, req Request, role ModelRole, need []capability) (Request, []string, string) {
	resolver, ok := p.(ModelResolver)
	if !ok {
		return req, nil, ""
	}

	var name string
	var err error
	if req.Tier != "" {
		name, err = resolver.ResolveModel(role, req.Tier)
	} else {
		for _, tier := range []ModelTier{ModelTierBest, ModelTierFast} {
			if name, err = resolver.ResolveModel(role, tier); err == nil {
				break
			}
		}
	}
	if err != nil {
		roleCap := capTextGeneration
		if role == ModelRoleImage {
			roleCap = capImageGeneration
		}
		reason := fmt.Sprintf("no %s model", role)
		if req.Tier != "" {
			reason += fmt.Sprintf(" for tier %s", req.Tier)
		}
		return req, []string{roleCap.name}, reason
	}

	if model, ok := lookupModel(ctx, p, name); ok {
		var lacking []string
		for _, c := range need {
			if !c.has(model.Capabilities) {
				lacking = append(lacking, c.name)
			}
		}
		if len(lacking) > 0 {
			return req, lacking, fmt.Sprintf("model %s lacks %s", name, strings.Join(lacking, ", "))
		}
	}

	if req.Tier != "" {
		req.Model = name
	}
	return req, nil, ""
}

// requiredCapabilities lists the model capabilities req's output and inputs need.
func requiredCapabilities(req Request) []capability {
	var need []capability
	switch {
	case IsTextOutput(req.Output):
		need = append(need, capTextGeneration)
	case isImageOutput(req.Output):
		need = append(need, capImageGeneration)
	case isJSONOutput(req.Output):
		need = append(need, capJSONOutput)
	}

	var images, pdfs bool
	for _, input := range req.Inputs {
		var mime string
		if data, m, _, ok := AsFileInput(input); ok {
			mime = m
			if mime == "" {
				mime = SniffImageMIME(data)
			}
		} else if _, _, m, _, ok := AsFileReaderInput(input); ok {
			mime = m
		}
		images = images || strings.HasPrefix(mime, "image/")
		pdfs = pdfs || mime == "application/pdf"
	}
	if images {
		need = append(need, capImageUnderstanding)
	}
	if pdfs {
		need = append(need, capPDFUnderstanding)
	}
	return need
}

// lookupModel finds name in p's model list or, failing that, its full catalog
// of known models.
func lookupModel(ctx context.Context, p ProviderExecutor, name string) (Model, bool) {
	if lister, ok := p.(ModelLister); ok {
		if models, err := lister.ListModels(ctx); err == nil {
			for _, m := range models {
				if m.Name == name {
					return m, true
				}
			}
		}
	}
	if known, ok := p.(KnownModelLister); ok {
		for _, m := range known.KnownModels() {
			if m.Name == name {
				return m, true
			}
		}
	}
	return Model{}, false
}

// generate tries routes in order and returns the first success. It moves on
// to the next route only when the error is retryable.
func (c *Client) generate(ctx context.Context, routes []route) (Response, error) {
	var warnings []Warning
	for i, rt := range routes {
		res, err := c.attempt(ctx, rt)
		if err == nil {
			if res.Provider.Name == "" {
				res.Provider.Name = rt.provider.Name()
			}
			res.Warnings = append(warnings, res.Warnings...)
			return res, nil
		}

		if i == len(routes)-1 || !shouldFallback(err) || ctx.Err() != nil {
			return Response{}, err
		}
		next := routes[i+1].provider.Name()
		warnings = append(warnings, Warning{
			Code:    WarningFallback,
			Message: fmt.Sprintf("%s failed (%v); fell back to %s", rt.provider.Name(), err, next),
		})
		if c.log != nil {
			c.log.Warn("provider failed, falling back",
				slog.String("provider", rt.provider.Name()),
				slog.String("next", next),
				slog.String("error", err.Error()),
			)
		}
	}
	return Response{}, NewGrailError(Internal, "no provider to try")
}

// attempt sends the request along a single route.
func (c *Client) attempt(ctx context.Context, rt route) (Response, error) {
	if c.log != nil {
		// Get model description - provider can override for complex cases
		models := rt.req.Model
		if describer, ok := rt.provider.(ModelDescriber); ok {
			models = describer.DescribeModels(rt.req)
		}
		c.log.Info("generate request",
			slog.String("provider", rt.provider.Name()),
			slog.Int("inputs", len(rt.req.Inputs)),
			slog.String("output_type", getOutputType(rt.req.Output)),
			slog.String("model", models),
		)
	}
	return rt.provider.DoGenerate(ctx, rt.req)
}

// shouldFallback reports whether err warrants trying the next provider.
func shouldFallback(err error) bool {
	code := GetErrorCode(err)
	return IsRetryable(err) || code == RateLimited || code == Unavailable
}

// listModels returns the models of every provider that lists them, in
// priority order. A model name listed by several providers appears once.
func (c *Client) listModels(ctx context.Context) ([]Model, error) {
	if len(c.providers) == 0 {
		return nil, NewGrailError(Internal, "provider executor not available")
	}

	var models []Model
	var listed bool
	seen := map[string]bool{}
	for _, p := range c.providers {
		lister, ok := p.(ModelLister)
		if !ok {
			continue
		}
		listed = true
		ms, err := lister.ListModels(ctx)
		if err != nil {
			return nil, fmt.Errorf("%s: %w", p.Name(), err)
		}
		for _, m := range ms {
			if !seen[m.Name] {
				seen[m.Name] = true
				models = append(models, m)
			}
		}
	}

	if !listed {
		return nil, NewGrailError(Unsupported, "no provider supports model listing")
	}
	return models, nil
}

func isImageOutput(output Output) bool {
	_, ok := GetImageSpec(output)
	return ok
}

func isJSONOutput(output Output) bool {
	_, _, ok := GetJSONOutput(output)
	return ok
}
