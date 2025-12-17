package gemini

import "github.com/montanaflynn/grail"

// Model constants for Gemini models.
// Use these directly in requests: grail.Request{Model: gemini.Gemini3Flash.Name}
var (
	// Gemini3Flash is the latest Gemini 3 Flash model for text generation.
	Gemini3Flash = grail.Model{
		Name: "gemini-3-flash-preview",
		Role: grail.ModelRoleText,
		Tier: grail.ModelTierBest,
		Capabilities: grail.ModelCapabilities{
			TextGeneration:     true,
			ImageUnderstanding: true,
			PDFUnderstanding:   true,
			JSONOutput:         true,
		},
	}

	// Gemini25Flash is the Gemini 2.5 Flash model, optimized for speed.
	Gemini25Flash = grail.Model{
		Name: "gemini-2.5-flash",
		Role: grail.ModelRoleText,
		Tier: grail.ModelTierFast,
		Capabilities: grail.ModelCapabilities{
			TextGeneration:     true,
			ImageUnderstanding: true,
			PDFUnderstanding:   true,
			JSONOutput:         true,
		},
	}

	// Gemini25FlashImage is the Gemini 2.5 Flash model for image generation.
	Gemini25FlashImage = grail.Model{
		Name: "gemini-2.5-flash-image",
		Role: grail.ModelRoleImage,
		Tier: grail.ModelTierBest,
		Capabilities: grail.ModelCapabilities{
			ImageGeneration:    true,
			ImageUnderstanding: true,
		},
	}
)
