package gemini

import (
	"time"

	"github.com/montanaflynn/grail"
)

// Model constants for Gemini models.
// Use these directly in requests: grail.Request{Model: gemini.Gemini4Argon.Name}

// Best models - highest quality
var (
	// Gemini4Argon is the flagship reasoning model, with a unified 1M-token
	// context window.
	Gemini4Argon = grail.Model{
		Name: "gemini-4-argon",
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
	// Gemini3_8Flash is the most intelligent Flash model, optimized for
	// speed and cost.
	Gemini3_8Flash = grail.Model{
		Name: "gemini-3.8-flash",
		Role: grail.ModelRoleText,
		Tier: grail.ModelTierFast,
		Capabilities: grail.ModelCapabilities{
			TextGeneration:     true,
			ImageUnderstanding: true,
			PDFUnderstanding:   true,
			JSONOutput:         true,
		},
	}

	// GeminiNanoBanana2_1 (Nano Banana 2.1) is a high-efficiency image
	// generation and editing model. Replaces the deprecated gemini-3.1-flash-image.
	GeminiNanoBanana2_1 = grail.Model{
		Name: "gemini-nano-banana-2.1",
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
	// Gemini3_1Pro is the previous best text model, with advanced reasoning
	// and 1M-token context. Retained for callers that want to pin to it.
	Gemini3_1Pro = grail.Model{
		Name: "gemini-3.1-pro-preview",
		Role: grail.ModelRoleText,
		Capabilities: grail.ModelCapabilities{
			TextGeneration:     true,
			ImageUnderstanding: true,
			PDFUnderstanding:   true,
			JSONOutput:         true,
		},
	}

	// Gemini3_6Flash is the previous-generation Flash model, balancing speed
	// and multimodal performance.
	Gemini3_6Flash = grail.Model{
		Name: "gemini-3.6-flash",
		Role: grail.ModelRoleText,
		Capabilities: grail.ModelCapabilities{
			TextGeneration:     true,
			ImageUnderstanding: true,
			PDFUnderstanding:   true,
			JSONOutput:         true,
		},
	}

	// Gemini3_5FlashLite is the fastest, most cost-effective 3.5 model for
	// high-throughput workloads.
	Gemini3_5FlashLite = grail.Model{
		Name: "gemini-3.5-flash-lite",
		Role: grail.ModelRoleText,
		Capabilities: grail.ModelCapabilities{
			TextGeneration:     true,
			ImageUnderstanding: true,
			PDFUnderstanding:   true,
			JSONOutput:         true,
		},
	}

	// Gemini3_5Flash is a previous fast text model. Deprecated on 2026-10-08
	// and routed to gemini-3.6-flash; migrate to Gemini3_8Flash.
	Gemini3_5Flash = grail.Model{
		Name: "gemini-3.5-flash",
		Role: grail.ModelRoleText,
		Capabilities: grail.ModelCapabilities{
			TextGeneration:     true,
			ImageUnderstanding: true,
			PDFUnderstanding:   true,
			JSONOutput:         true,
		},
		Deprecation: &grail.Deprecation{
			Announced:   date(2026, time.October, 8),
			Replacement: Gemini3_8Flash.Name,
			Notes:       "Requests are routed to gemini-3.6-flash.",
		},
	}

	// Gemini3_1FlashLiteImage (Nano Banana 2 Lite) is an efficiency-focused
	// image generation and editing model.
	Gemini3_1FlashLiteImage = grail.Model{
		Name: "gemini-3.1-flash-lite-image",
		Role: grail.ModelRoleImage,
		Capabilities: grail.ModelCapabilities{
			ImageGeneration:    true,
			ImageUnderstanding: true,
		},
	}

	// Gemini3_1FlashImage (Nano Banana 2) is the previous fast image model.
	// Deprecated on 2026-10-06; migrate to GeminiNanoBanana2_1.
	Gemini3_1FlashImage = grail.Model{
		Name: "gemini-3.1-flash-image",
		Role: grail.ModelRoleImage,
		Capabilities: grail.ModelCapabilities{
			ImageGeneration:    true,
			ImageUnderstanding: true,
		},
		Deprecation: &grail.Deprecation{
			Announced:   date(2026, time.October, 6),
			Replacement: GeminiNanoBanana2_1.Name,
		},
	}

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
	// migrate to Gemini4Argon.
	Gemini3Pro = grail.Model{
		Name: "gemini-3-pro-preview",
		Role: grail.ModelRoleText,
		Capabilities: grail.ModelCapabilities{
			TextGeneration:     true,
			ImageUnderstanding: true,
			PDFUnderstanding:   true,
			JSONOutput:         true,
		},
		Deprecation: &grail.Deprecation{
			Shutdown:    date(2026, time.March, 9),
			Replacement: Gemini4Argon.Name,
		},
	}

	// Gemini3Flash is a legacy preview fast text model in the Gemini 3 series.
	// Kept for backwards-compat; migrate to Gemini3_8Flash.
	Gemini3Flash = grail.Model{
		Name: "gemini-3-flash-preview",
		Role: grail.ModelRoleText,
		Capabilities: grail.ModelCapabilities{
			TextGeneration:     true,
			ImageUnderstanding: true,
			PDFUnderstanding:   true,
			JSONOutput:         true,
		},
		Deprecation: &grail.Deprecation{
			Replacement: Gemini3_8Flash.Name,
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
		Deprecation: &grail.Deprecation{
			Shutdown:    date(2026, time.June, 25),
			Replacement: Gemini3ProImage.Name,
		},
	}

	// Gemini3_1FlashImagePreview is the preview fast image model, shut down
	// on 2026-06-25 — kept for backwards-compat; migrate to GeminiNanoBanana2_1
	// (its GA equivalent, Gemini3_1FlashImage, is itself deprecated).
	Gemini3_1FlashImagePreview = grail.Model{
		Name: "gemini-3.1-flash-image-preview",
		Role: grail.ModelRoleImage,
		Capabilities: grail.ModelCapabilities{
			ImageGeneration:    true,
			ImageUnderstanding: true,
		},
		Deprecation: &grail.Deprecation{
			Shutdown:    date(2026, time.June, 25),
			Replacement: GeminiNanoBanana2_1.Name,
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

// KnownModels returns every model this package exports, including deprecated
// and shut-down models. It implements grail.KnownModelLister.
func (c *Provider) KnownModels() []grail.Model {
	return []grail.Model{
		Gemini4Argon,
		Gemini3ProImage,
		Gemini3_8Flash,
		GeminiNanoBanana2_1,
		Gemini3_1Pro,
		Gemini3_6Flash,
		Gemini3_5FlashLite,
		Gemini3_5Flash,
		Gemini3_1FlashLiteImage,
		Gemini3_1FlashImage,
		Gemini3_1FlashLite,
		Gemini3Pro,
		Gemini3Flash,
		Gemini3ProImagePreview,
		Gemini3_1FlashImagePreview,
		Gemini25FlashImage,
		Gemini25Flash,
		Gemini25FlashLite,
	}
}

// date returns midnight UTC on the given day.
func date(year int, month time.Month, day int) time.Time {
	return time.Date(year, month, day, 0, 0, 0, 0, time.UTC)
}
