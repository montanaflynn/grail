package modelslab

import "github.com/montanaflynn/grail"

// Model constants for ModelsLab models.
// Use these directly in requests: grail.Request{Model: modelslab.Flux.Name}
var (
	// Flux is ModelsLab's default high-quality text-to-image model.
	Flux = grail.Model{
		Name: "flux",
		Role: grail.ModelRoleImage,
		Tier: grail.ModelTierBest,
		Capabilities: grail.ModelCapabilities{
			ImageGeneration: true,
		},
	}

	// FluxDev is the Flux development model variant.
	FluxDev = grail.Model{
		Name: "flux-dev",
		Role: grail.ModelRoleImage,
		Tier: grail.ModelTierBest,
		Capabilities: grail.ModelCapabilities{
			ImageGeneration: true,
		},
	}

	// SDXL is Stable Diffusion XL, a fast and versatile image model.
	SDXL = grail.Model{
		Name: "sdxl",
		Role: grail.ModelRoleImage,
		Tier: grail.ModelTierFast,
		Capabilities: grail.ModelCapabilities{
			ImageGeneration: true,
		},
	}

	// RealisticVision is optimised for photorealistic images.
	RealisticVision = grail.Model{
		Name: "realistic-vision-v6",
		Role: grail.ModelRoleImage,
		Tier: grail.ModelTierFast,
		Capabilities: grail.ModelCapabilities{
			ImageGeneration: true,
		},
	}
)
