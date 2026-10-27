// Anthropic-text demonstrates text and JSON generation with Claude.
// It uses the Anthropic provider with the API key from the ANTHROPIC_API_KEY
// environment variable, generates text with the fast tier, then requests
// structured JSON output matching a schema from the best tier.
//
// Usage:
//
//	go run examples/anthropic-text/main.go
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/montanaflynn/grail"
	"github.com/montanaflynn/grail/providers/anthropic"
)

func main() {
	ctx := context.Background()

	// Uses default API key from environment variable ANTHROPIC_API_KEY if set.
	provider, err := anthropic.New(
	// anthropic.WithAPIKey("sk-ant-*************"),
	// anthropic.WithAPIKeyFromEnv("ANTHROPIC_API_KEY"),
	// anthropic.WithTextModel("claude-opus-5-5"),
	)
	if err != nil {
		log.Fatalf("new provider: %v", err)
	}

	client := grail.NewClient(provider)

	// Text with the fast tier (claude-haiku-5-5).
	res, err := client.Generate(ctx, grail.Request{
		Inputs: []grail.Input{grail.InputText("Say hello in one short sentence.")},
		Output: grail.OutputText(),
		Tier:   grail.ModelTierFast,
		ProviderOptions: []grail.ProviderOption{
			anthropic.TextOptions{
				MaxTokens:    grail.Pointer[int32](256),
				SystemPrompt: "You are brief and friendly.",
			},
		},
	})
	if err != nil {
		log.Fatalf("generate text: %v", err)
	}
	text, _ := res.Text()
	fmt.Println(text)

	// JSON with the best tier (claude-fable-5-1). The schema is sent to the
	// Messages API's structured outputs, so the response matches it.
	// Claude requires additionalProperties: false on every object.
	schema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"name":    map[string]any{"type": "string"},
			"capital": map[string]any{"type": "string"},
		},
		"required":             []string{"name", "capital"},
		"additionalProperties": false,
	}
	res, err = client.Generate(ctx, grail.Request{
		Inputs: []grail.Input{grail.InputText("Name a country in South America and its capital.")},
		Output: grail.OutputJSON(schema),
		Tier:   grail.ModelTierBest,
	})
	if err != nil {
		log.Fatalf("generate JSON: %v", err)
	}
	var country struct {
		Name    string `json:"name"`
		Capital string `json:"capital"`
	}
	if err := res.DecodeJSON(&country); err != nil {
		log.Fatalf("decode JSON: %v", err)
	}
	fmt.Printf("%s: %s\n", country.Name, country.Capital)
}
