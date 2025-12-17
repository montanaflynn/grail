package gemini

import "github.com/montanaflynn/grail"

// Model constants for Gemini models.
// Use these directly in requests: grail.Request{Model: gemini.Gemini3Pro.Name}

// Best models - highest quality
var (
	// Gemini3Pro is the best quality text generation model.
	Gemini3Pro = grail.Model{
		Name: "gemini-3-pro-preview",
		Role: grail.ModelRoleText,
		Tier: grail.ModelTierBest,
		Capabilities: grail.ModelCapabilities{
			TextGeneration:     true,
			ImageUnderstanding: true,
			PDFUnderstanding:   true,
			JSONOutput:         true,
		},
	}

	// Gemini3ProImage is the best quality image generation model.
	Gemini3ProImage = grail.Model{
		Name: "gemini-3-pro-image-preview",
		Role: grail.ModelRoleImage,
		Tier: grail.ModelTierBest,
		Capabilities: grail.ModelCapabilities{
			ImageGeneration:    true,
			ImageUnderstanding: true,
		},
	}
)

// Fast models - speed/cost optimized
var (
	// Gemini3Flash is a fast text generation model.
	Gemini3Flash = grail.Model{
		Name: "gemini-3-flash-preview",
		Role: grail.ModelRoleText,
		Tier: grail.ModelTierFast,
		Capabilities: grail.ModelCapabilities{
			TextGeneration:     true,
			ImageUnderstanding: true,
			PDFUnderstanding:   true,
			JSONOutput:         true,
		},
	}

	// Gemini25FlashImage is a fast image generation model.
	Gemini25FlashImage = grail.Model{
		Name: "gemini-2.5-flash-image",
		Role: grail.ModelRoleImage,
		Tier: grail.ModelTierFast,
		Capabilities: grail.ModelCapabilities{
			ImageGeneration:    true,
			ImageUnderstanding: true,
		},
	}
)

// Other models - available but not set as default best/fast
var (
	// Gemini25Flash is a balanced text generation model.
	Gemini25Flash = grail.Model{
		Name: "gemini-2.5-flash",
		Role: grail.ModelRoleText,
		Tier: "", // Not categorized as best or fast
		Capabilities: grail.ModelCapabilities{
			TextGeneration:     true,
			ImageUnderstanding: true,
			PDFUnderstanding:   true,
			JSONOutput:         true,
		},
	}

	// Gemini25FlashLite is a lightweight text generation model.
	Gemini25FlashLite = grail.Model{
		Name: "gemini-2.5-flash-lite",
		Role: grail.ModelRoleText,
		Tier: "", // Not categorized as best or fast
		Capabilities: grail.ModelCapabilities{
			TextGeneration:     true,
			ImageUnderstanding: true,
			PDFUnderstanding:   true,
			JSONOutput:         true,
		},
	}
)
