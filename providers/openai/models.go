package openai

import (
	"github.com/montanaflynn/grail"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/shared"
)

// Model constants for OpenAI models.
// Use these directly in requests: grail.Request{Model: openai.GPT5_4.Name}
var (
	// GPT5_4 is the frontier GPT-5.4 text model, with built-in computer use
	// and 1M-token context.
	GPT5_4 = grail.Model{
		Name: shared.ChatModelGPT5_4,
		Role: grail.ModelRoleText,
		Tier: grail.ModelTierBest,
		Capabilities: grail.ModelCapabilities{
			TextGeneration:     true,
			ImageUnderstanding: true,
			PDFUnderstanding:   true,
			JSONOutput:         true,
		},
	}

	// GPT5_4Mini is the cost-optimized GPT-5.4 text model.
	GPT5_4Mini = grail.Model{
		Name: shared.ChatModelGPT5_4Mini,
		Role: grail.ModelRoleText,
		Tier: grail.ModelTierFast,
		Capabilities: grail.ModelCapabilities{
			TextGeneration:     true,
			ImageUnderstanding: true,
			PDFUnderstanding:   true,
			JSONOutput:         true,
		},
	}

	// GPT5_4Nano is the smallest GPT-5.4 text model.
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

	// GPTImage2 is the best quality image generation model, with reasoning,
	// 2K output, multi-image coherence, and improved multilingual text rendering.
	GPTImage2 = grail.Model{
		Name: "gpt-image-2",
		Role: grail.ModelRoleImage,
		Tier: grail.ModelTierBest,
		Capabilities: grail.ModelCapabilities{
			ImageGeneration:    true,
			ImageUnderstanding: true,
		},
	}

	// GPTImage1 is the previous-generation image model, retained for callers
	// that want to pin to it explicitly.
	GPTImage1 = grail.Model{
		Name: openai.ImageModelGPTImage1,
		Role: grail.ModelRoleImage,
		Capabilities: grail.ModelCapabilities{
			ImageGeneration:    true,
			ImageUnderstanding: true,
		},
	}

	// GPTImage1Mini is a faster, lower-cost image generation model.
	GPTImage1Mini = grail.Model{
		Name: openai.ImageModelGPTImage1Mini,
		Role: grail.ModelRoleImage,
		Tier: grail.ModelTierFast,
		Capabilities: grail.ModelCapabilities{
			ImageGeneration:    true,
			ImageUnderstanding: true,
		},
	}
)
