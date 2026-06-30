package gemini

import "github.com/montanaflynn/grail"

// Model constants for Gemini models.
// Use these directly in requests: grail.Request{Model: gemini.Gemini3_1Pro.Name}

// Best models - highest quality
var (
	// Gemini3_1Pro is the best quality text generation model, with advanced
	// reasoning and 1M-token context. Replaces the deprecated gemini-3-pro-preview.
	Gemini3_1Pro = grail.Model{
		Name: "gemini-3.1-pro-preview",
		Role: grail.ModelRoleText,
		Tier: grail.ModelTierBest,
		Capabilities: grail.ModelCapabilities{
			TextGeneration:     true,
			ImageUnderstanding: true,
			PDFUnderstanding:   true,
			JSONOutput:         true,
		},
	}

	// Gemini3ProImage (Nano Banana Pro) is the best quality image generation model.
	Gemini3ProImage = grail.Model{
		Name: "gemini-3-pro-image",
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
	// Gemini3_5Flash is the GA fast text model in the Gemini 3.5 series, with
	// frontier performance on agentic and coding tasks.
	Gemini3_5Flash = grail.Model{
		Name: "gemini-3.5-flash",
		Role: grail.ModelRoleText,
		Tier: grail.ModelTierFast,
		Capabilities: grail.ModelCapabilities{
			TextGeneration:     true,
			ImageUnderstanding: true,
			PDFUnderstanding:   true,
			JSONOutput:         true,
		},
	}

	// Gemini3_1FlashImage (Nano Banana 2) is a fast image generation model
	// with Pro-tier quality, 512px–4K output, and extended aspect ratios.
	Gemini3_1FlashImage = grail.Model{
		Name: "gemini-3.1-flash-image",
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
	// Gemini3_1FlashLite is the GA cost-efficient text model for high-volume,
	// low-latency workloads.
	Gemini3_1FlashLite = grail.Model{
		Name: "gemini-3.1-flash-lite",
		Role: grail.ModelRoleText,
		Capabilities: grail.ModelCapabilities{
			TextGeneration:     true,
			ImageUnderstanding: true,
			PDFUnderstanding:   true,
			JSONOutput:         true,
		},
	}

	// Gemini3Pro is the previous best text model. Deprecated and shut down
	// on 2026-03-09 — kept as a symbol for backwards-compat; callers should
	// migrate to Gemini3_1Pro.
	Gemini3Pro = grail.Model{
		Name: "gemini-3-pro-preview",
		Role: grail.ModelRoleText,
		Capabilities: grail.ModelCapabilities{
			TextGeneration:     true,
			ImageUnderstanding: true,
			PDFUnderstanding:   true,
			JSONOutput:         true,
		},
	}

	// Gemini3Flash is the previous fast text model in the Gemini 3 series.
	// Retained for callers that want to pin to it explicitly.
	Gemini3Flash = grail.Model{
		Name: "gemini-3-flash-preview",
		Role: grail.ModelRoleText,
		Capabilities: grail.ModelCapabilities{
			TextGeneration:     true,
			ImageUnderstanding: true,
			PDFUnderstanding:   true,
			JSONOutput:         true,
		},
	}

	// Gemini3ProImagePreview is the preview image model, shut down on
	// 2026-06-25 — kept for backwards-compat; migrate to Gemini3ProImage.
	Gemini3ProImagePreview = grail.Model{
		Name: "gemini-3-pro-image-preview",
		Role: grail.ModelRoleImage,
		Capabilities: grail.ModelCapabilities{
			ImageGeneration:    true,
			ImageUnderstanding: true,
		},
	}

	// Gemini3_1FlashImagePreview is the preview fast image model, shut down
	// on 2026-06-25 — kept for backwards-compat; migrate to Gemini3_1FlashImage.
	Gemini3_1FlashImagePreview = grail.Model{
		Name: "gemini-3.1-flash-image-preview",
		Role: grail.ModelRoleImage,
		Capabilities: grail.ModelCapabilities{
			ImageGeneration:    true,
			ImageUnderstanding: true,
		},
	}

	// Gemini25FlashImage (Nano Banana) is the previous-generation fast image
	// model, retained for callers that want to pin to it explicitly.
	Gemini25FlashImage = grail.Model{
		Name: "gemini-2.5-flash-image",
		Role: grail.ModelRoleImage,
		Capabilities: grail.ModelCapabilities{
			ImageGeneration:    true,
			ImageUnderstanding: true,
		},
	}

	// Gemini25Flash is a balanced text generation model from the 2.5 series.
	Gemini25Flash = grail.Model{
		Name: "gemini-2.5-flash",
		Role: grail.ModelRoleText,
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
		Capabilities: grail.ModelCapabilities{
			TextGeneration:     true,
			ImageUnderstanding: true,
			PDFUnderstanding:   true,
			JSONOutput:         true,
		},
	}
)
