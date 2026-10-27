package openai

import (
	"github.com/montanaflynn/grail"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/shared"
)

// Model constants for OpenAI models.
// Use these directly in requests: grail.Request{Model: openai.GPT6Astra.Name}
var (
	// GPT6Astra is the flagship GPT-6 text model for complex reasoning and
	// coding, with 1M-token context.
	GPT6Astra = grail.Model{
		Name: shared.ChatModelGPT6Astra,
		Role: grail.ModelRoleText,
		Tier: grail.ModelTierBest,
		Capabilities: grail.ModelCapabilities{
			TextGeneration:     true,
			ImageUnderstanding: true,
			PDFUnderstanding:   true,
			JSONOutput:         true,
		},
	}

	// GPT6_1Sol is the mid-tier GPT-6.1 text model, balancing capability and cost.
	GPT6_1Sol = grail.Model{
		Name: shared.ChatModelGPT6_1Sol,
		Role: grail.ModelRoleText,
		Capabilities: grail.ModelCapabilities{
			TextGeneration:     true,
			ImageUnderstanding: true,
			PDFUnderstanding:   true,
			JSONOutput:         true,
		},
	}

	// GPT6_1Terra is a lower-cost GPT-6.1 text model.
	GPT6_1Terra = grail.Model{
		Name: shared.ChatModelGPT6_1Terra,
		Role: grail.ModelRoleText,
		Capabilities: grail.ModelCapabilities{
			TextGeneration:     true,
			ImageUnderstanding: true,
			PDFUnderstanding:   true,
			JSONOutput:         true,
		},
	}

	// GPT6Luna is the most efficient GPT-6 text model, for focused,
	// high-volume tasks.
	GPT6Luna = grail.Model{
		Name: shared.ChatModelGPT6Luna,
		Role: grail.ModelRoleText,
		Tier: grail.ModelTierFast,
		Capabilities: grail.ModelCapabilities{
			TextGeneration:     true,
			ImageUnderstanding: true,
			PDFUnderstanding:   true,
			JSONOutput:         true,
		},
	}

	// GPT5_4 is the previous-generation GPT-5.4 text model, retained for
	// callers that want to pin to it explicitly.
	GPT5_4 = grail.Model{
		Name: shared.ChatModelGPT5_4,
		Role: grail.ModelRoleText,
		Capabilities: grail.ModelCapabilities{
			TextGeneration:     true,
			ImageUnderstanding: true,
			PDFUnderstanding:   true,
			JSONOutput:         true,
		},
	}

	// GPT5_4Mini is the cost-optimized GPT-5.4 text model, retained for
	// callers that want to pin to it explicitly.
	GPT5_4Mini = grail.Model{
		Name: shared.ChatModelGPT5_4Mini,
		Role: grail.ModelRoleText,
		Capabilities: grail.ModelCapabilities{
			TextGeneration:     true,
			ImageUnderstanding: true,
			PDFUnderstanding:   true,
			JSONOutput:         true,
		},
	}

	// GPT5_4Nano is the smallest GPT-5.4 text model. Deprecated with shutdown
	// on 2027-04-01; migrate to GPT6Luna.
	GPT5_4Nano = grail.Model{
		Name: shared.ChatModelGPT5_4Nano,
		Role: grail.ModelRoleText,
		Capabilities: grail.ModelCapabilities{
			TextGeneration:     true,
			ImageUnderstanding: true,
			PDFUnderstanding:   true,
			JSONOutput:         true,
		},
	}

	// GPT5_2 is the previous-generation GPT-5.2 text model, retained for
	// callers that want to pin to it explicitly.
	GPT5_2 = grail.Model{
		Name: shared.ChatModelGPT5_2,
		Role: grail.ModelRoleText,
		Capabilities: grail.ModelCapabilities{
			TextGeneration:     true,
			ImageUnderstanding: true,
			PDFUnderstanding:   true,
			JSONOutput:         true,
		},
	}

	// GPT4o is the GPT-4o model, retained for callers that want to pin to it.
	// Its snapshots are deprecated with shutdown on 2027-01-14; migrate to GPT6Luna.
	GPT4o = grail.Model{
		Name: shared.ChatModelGPT4o,
		Role: grail.ModelRoleText,
		Capabilities: grail.ModelCapabilities{
			TextGeneration:     true,
			ImageUnderstanding: true,
			PDFUnderstanding:   true,
			JSONOutput:         true,
		},
	}

	// GPTImage2_5Sunburst is the most capable image generation and editing model.
	GPTImage2_5Sunburst = grail.Model{
		Name: openai.ImageModelGPTImage2_5Sunburst,
		Role: grail.ModelRoleImage,
		Tier: grail.ModelTierBest,
		Capabilities: grail.ModelCapabilities{
			ImageGeneration:    true,
			ImageUnderstanding: true,
		},
	}

	// GPTImage2_5Flare is a fast, high-quality image generation model for
	// everyday use.
	GPTImage2_5Flare = grail.Model{
		Name: openai.ImageModelGPTImage2_5Flare,
		Role: grail.ModelRoleImage,
		Tier: grail.ModelTierFast,
		Capabilities: grail.ModelCapabilities{
			ImageGeneration:    true,
			ImageUnderstanding: true,
		},
	}

	// GPTImage2 is the previous-generation image model, with reasoning, 2K
	// output, and multi-image coherence. Retained for callers that want to pin
	// to it explicitly.
	GPTImage2 = grail.Model{
		Name: openai.ImageModelGPTImage2,
		Role: grail.ModelRoleImage,
		Capabilities: grail.ModelCapabilities{
			ImageGeneration:    true,
			ImageUnderstanding: true,
		},
	}

	// GPTImage1 is a legacy image model. Shut down on 2026-10-23 — kept as a
	// symbol for backwards-compat; migrate to GPTImage2_5Sunburst.
	GPTImage1 = grail.Model{
		Name: openai.ImageModelGPTImage1,
		Role: grail.ModelRoleImage,
		Capabilities: grail.ModelCapabilities{
			ImageGeneration:    true,
			ImageUnderstanding: true,
		},
	}

	// GPTImage1Mini is a legacy low-cost image model. Deprecated with shutdown
	// on 2026-12-01; migrate to GPTImage2_5Flare.
	GPTImage1Mini = grail.Model{
		Name: openai.ImageModelGPTImage1Mini,
		Role: grail.ModelRoleImage,
		Capabilities: grail.ModelCapabilities{
			ImageGeneration:    true,
			ImageUnderstanding: true,
		},
	}
)
