package anthropic

import (
	"time"

	"github.com/montanaflynn/grail"

	"github.com/anthropics/anthropic-sdk-go/v2"
)

// textCapabilities are shared by every Claude model: text generation, image
// and PDF understanding, and structured JSON output. Claude cannot generate
// images.
var textCapabilities = grail.ModelCapabilities{
	TextGeneration:     true,
	ImageUnderstanding: true,
	PDFUnderstanding:   true,
	JSONOutput:         true,
}

// Model constants for Anthropic Claude models.
// Use these directly in requests: grail.Request{Model: anthropic.ClaudeFable5_1.Name}
var (
	// ClaudeFable5_1 is Anthropic's most capable widely released model, for
	// demanding reasoning and long-horizon agentic work.
	ClaudeFable5_1 = grail.Model{
		Name:         anthropic.ModelClaudeFable5_1,
		Role:         grail.ModelRoleText,
		Tier:         grail.ModelTierBest,
		Capabilities: textCapabilities,
	}

	// ClaudeOpus5_5 is the current Opus model, for long-running agentic coding
	// and knowledge work.
	ClaudeOpus5_5 = grail.Model{
		Name:         anthropic.ModelClaudeOpus5_5,
		Role:         grail.ModelRoleText,
		Capabilities: textCapabilities,
	}

	// ClaudeSonnet5_5 is the current Sonnet model, balancing speed and
	// intelligence.
	ClaudeSonnet5_5 = grail.Model{
		Name:         anthropic.ModelClaudeSonnet5_5,
		Role:         grail.ModelRoleText,
		Capabilities: textCapabilities,
	}

	// ClaudeHaiku5_5 is the fastest and cheapest current model, for
	// high-volume, latency-sensitive tasks.
	ClaudeHaiku5_5 = grail.Model{
		Name:         anthropic.ModelClaudeHaiku5_5,
		Role:         grail.ModelRoleText,
		Tier:         grail.ModelTierFast,
		Capabilities: textCapabilities,
	}

	// ClaudeMythos5_1 has Fable 5.1's capabilities and is available only
	// through Anthropic's trusted access programs.
	ClaudeMythos5_1 = grail.Model{
		Name:         anthropic.ModelClaudeMythos5_1,
		Role:         grail.ModelRoleText,
		Capabilities: textCapabilities,
	}

	// ClaudeFable5 is the previous-generation Fable model (legacy).
	ClaudeFable5 = grail.Model{
		Name:         anthropic.ModelClaudeFable5,
		Role:         grail.ModelRoleText,
		Capabilities: textCapabilities,
	}

	// ClaudeMythos5 is the previous-generation Mythos model, available only
	// through trusted access programs.
	ClaudeMythos5 = grail.Model{
		Name:         anthropic.ModelClaudeMythos5,
		Role:         grail.ModelRoleText,
		Capabilities: textCapabilities,
	}

	// ClaudeMythosPreview is deprecated as of 2026-06-09 with no retirement
	// date announced; migrate to ClaudeMythos5.
	ClaudeMythosPreview = grail.Model{
		Name:         anthropic.ModelClaudeMythosPreview,
		Role:         grail.ModelRoleText,
		Capabilities: textCapabilities,
		Deprecation: &grail.Deprecation{
			Announced:   date(2026, time.June, 9),
			Replacement: ClaudeMythos5.Name,
			Notes:       "Retirement date to be announced.",
		},
	}

	// ClaudeOpus5 is the previous-generation Opus model (legacy).
	ClaudeOpus5 = grail.Model{
		Name:         anthropic.ModelClaudeOpus5,
		Role:         grail.ModelRoleText,
		Capabilities: textCapabilities,
	}

	// ClaudeOpus4_8 is a legacy Opus model.
	ClaudeOpus4_8 = grail.Model{
		Name:         anthropic.ModelClaudeOpus4_8,
		Role:         grail.ModelRoleText,
		Capabilities: textCapabilities,
	}

	// ClaudeOpus4_7 is a legacy Opus model.
	ClaudeOpus4_7 = grail.Model{
		Name:         anthropic.ModelClaudeOpus4_7,
		Role:         grail.ModelRoleText,
		Capabilities: textCapabilities,
	}

	// ClaudeOpus4_6 is a legacy Opus model.
	ClaudeOpus4_6 = grail.Model{
		Name:         anthropic.ModelClaudeOpus4_6,
		Role:         grail.ModelRoleText,
		Capabilities: textCapabilities,
	}

	// ClaudeOpus4_5 is a legacy Opus model (alias of ClaudeOpus4_5_20251101).
	ClaudeOpus4_5 = grail.Model{
		Name:         anthropic.ModelClaudeOpus4_5,
		Role:         grail.ModelRoleText,
		Capabilities: textCapabilities,
	}

	// ClaudeOpus4_5_20251101 is the dated snapshot of Claude Opus 4.5.
	ClaudeOpus4_5_20251101 = grail.Model{
		Name:         anthropic.ModelClaudeOpus4_5_20251101,
		Role:         grail.ModelRoleText,
		Capabilities: textCapabilities,
	}

	// ClaudeSonnet5 is the previous-generation Sonnet model (legacy).
	ClaudeSonnet5 = grail.Model{
		Name:         anthropic.ModelClaudeSonnet5,
		Role:         grail.ModelRoleText,
		Capabilities: textCapabilities,
	}

	// ClaudeSonnet4_6 is a legacy Sonnet model.
	ClaudeSonnet4_6 = grail.Model{
		Name:         anthropic.ModelClaudeSonnet4_6,
		Role:         grail.ModelRoleText,
		Capabilities: textCapabilities,
	}

	// ClaudeSonnet4_5 is deprecated with shutdown on 2026-11-30; migrate to
	// ClaudeSonnet5_5.
	ClaudeSonnet4_5 = grail.Model{
		Name:         anthropic.ModelClaudeSonnet4_5,
		Role:         grail.ModelRoleText,
		Capabilities: textCapabilities,
		Deprecation: &grail.Deprecation{
			Announced:   date(2026, time.September, 30),
			Shutdown:    date(2026, time.November, 30),
			Replacement: ClaudeSonnet5_5.Name,
		},
	}

	// ClaudeSonnet4_5_20250929 is the dated snapshot of Claude Sonnet 4.5.
	// Deprecated with shutdown on 2026-11-30; migrate to ClaudeSonnet5_5.
	ClaudeSonnet4_5_20250929 = grail.Model{
		Name:         anthropic.ModelClaudeSonnet4_5_20250929,
		Role:         grail.ModelRoleText,
		Capabilities: textCapabilities,
		Deprecation: &grail.Deprecation{
			Announced:   date(2026, time.September, 30),
			Shutdown:    date(2026, time.November, 30),
			Replacement: ClaudeSonnet5_5.Name,
		},
	}

	// ClaudeHaiku4_5 is the previous-generation Haiku model (alias of
	// ClaudeHaiku4_5_20251001).
	ClaudeHaiku4_5 = grail.Model{
		Name:         anthropic.ModelClaudeHaiku4_5,
		Role:         grail.ModelRoleText,
		Capabilities: textCapabilities,
	}

	// ClaudeHaiku4_5_20251001 is the dated snapshot of Claude Haiku 4.5.
	ClaudeHaiku4_5_20251001 = grail.Model{
		Name:         anthropic.ModelClaudeHaiku4_5_20251001,
		Role:         grail.ModelRoleText,
		Capabilities: textCapabilities,
	}
)

// KnownModels returns every model this package exports, including deprecated
// and shut-down models. It implements grail.KnownModelLister.
func (p *Provider) KnownModels() []grail.Model {
	return []grail.Model{
		ClaudeFable5_1,
		ClaudeOpus5_5,
		ClaudeSonnet5_5,
		ClaudeHaiku5_5,
		ClaudeMythos5_1,
		ClaudeFable5,
		ClaudeMythos5,
		ClaudeMythosPreview,
		ClaudeOpus5,
		ClaudeOpus4_8,
		ClaudeOpus4_7,
		ClaudeOpus4_6,
		ClaudeOpus4_5,
		ClaudeOpus4_5_20251101,
		ClaudeSonnet5,
		ClaudeSonnet4_6,
		ClaudeSonnet4_5,
		ClaudeSonnet4_5_20250929,
		ClaudeHaiku4_5,
		ClaudeHaiku4_5_20251001,
	}
}

// date returns midnight UTC on the given day.
func date(year int, month time.Month, day int) time.Time {
	return time.Date(year, month, day, 0, 0, 0, 0, time.UTC)
}
