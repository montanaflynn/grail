package openai

import (
	"github.com/montanaflynn/grail"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/shared"
)

// Model constants for OpenAI models.
// Use these directly in requests: grail.Request{Model: openai.GPT5_2.Name}
var (
	// GPT5_2 is the latest GPT-5 model for text generation.
	GPT5_2 = grail.Model{
		Name: shared.ChatModelGPT5_2,
		Role: grail.ModelRoleText,
		Tier: grail.ModelTierBest,
		Capabilities: grail.ModelCapabilities{
			TextGeneration:     true,
			ImageUnderstanding: true,
			PDFUnderstanding:   true,
			JSONOutput:         true,
		},
	}

	// GPT4o is the GPT-4o model, optimized for speed.
	GPT4o = grail.Model{
		Name: shared.ChatModelGPT4o,
		Role: grail.ModelRoleText,
		Tier: grail.ModelTierFast,
		Capabilities: grail.ModelCapabilities{
			TextGeneration:     true,
			ImageUnderstanding: true,
			PDFUnderstanding:   true,
			JSONOutput:         true,
		},
	}

	// GPTImage1 is the best quality image generation model.
	GPTImage1 = grail.Model{
		Name: openai.ImageModelGPTImage1,
		Role: grail.ModelRoleImage,
		Tier: grail.ModelTierBest,
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
