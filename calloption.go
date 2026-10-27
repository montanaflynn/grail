package grail

// Tier aliases that make tier selection read naturally in call options:
//
//	client.Image(ctx, "a red fox", grail.WithTier(grail.Fast))
const (
	Best = ModelTierBest
	Fast = ModelTierFast
)

// CallOption configures a single convenience call such as Client.Text or
// Client.Image.
type CallOption interface{ applyCallOpt(*callOpt) }

type callOpt struct {
	model           string
	tier            ModelTier
	inputs          []Input
	providerOptions []ProviderOption
}

type callOptFunc func(*callOpt)

func (f callOptFunc) applyCallOpt(co *callOpt) {
	f(co)
}

// WithTier selects the model by tier (Best or Fast). It is ignored when
// WithModel is also set.
func WithTier(tier ModelTier) CallOption {
	return callOptFunc(func(co *callOpt) {
		co.tier = tier
	})
}

// WithModel selects an explicit provider-native model name. It takes priority
// over WithTier.
func WithModel(model string) CallOption {
	return callOptFunc(func(co *callOpt) {
		co.model = model
	})
}

// WithInputs appends extra inputs, such as images or PDFs, after the prompt.
// Repeated calls accumulate.
func WithInputs(inputs ...Input) CallOption {
	return callOptFunc(func(co *callOpt) {
		co.inputs = append(co.inputs, inputs...)
	})
}

// WithProviderOptions passes provider-specific options through to the
// request. Repeated calls accumulate.
func WithProviderOptions(opts ...ProviderOption) CallOption {
	return callOptFunc(func(co *callOpt) {
		co.providerOptions = append(co.providerOptions, opts...)
	})
}

// buildRequest turns a prompt, the desired output and call options into a
// Request. The prompt is always the first input.
func buildRequest(prompt string, output Output, opts []CallOption) Request {
	co := &callOpt{}
	for _, opt := range opts {
		if opt != nil {
			opt.applyCallOpt(co)
		}
	}

	inputs := make([]Input, 0, 1+len(co.inputs))
	inputs = append(inputs, InputText(prompt))
	inputs = append(inputs, co.inputs...)

	return Request{
		Inputs:          inputs,
		Output:          output,
		Model:           co.model,
		Tier:            co.tier,
		ProviderOptions: co.providerOptions,
	}
}
