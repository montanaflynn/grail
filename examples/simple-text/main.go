// Simple-text demonstrates minimal text generation with default settings.
// It uses the OpenAI provider with defaults (API key from OPENAI_API_KEY environment variable,
// default model, default logger) to generate text from a simple prompt.
//
// Usage:
//
//	go run examples/simple-text/main.go
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/montanaflynn/grail"
	"github.com/montanaflynn/grail/providers/openai"
)

// Minimal text generation:
// - Defaults: uses provider default model (gpt-6-astra for OpenAI), default logger, API key from OPENAI_API_KEY.
// - Uncomment the logger options and call options blocks below to customize.
func main() {
	ctx := context.Background()

	// Uses default API key from environment variable OPENAI_API_KEY if set.
	provider, err := openai.New(
	// openai.WithAPIKey("sk-proj-*************"),
	// openai.WithAPIKeyFromEnv("OPENAI_API_KEY"),
	// openai.WithTextModel("gpt-6-astra"),
	// openai.WithImageModel("gpt-image-2.5-sunburst"),
	)
	if err != nil {
		log.Fatalf("new provider: %v", err)
	}

	client := grail.NewClient(
		provider,
		// Logger options (uncomment to customize):
		// grail.WithLogger(yourLogger),
		// grail.WithLoggerFormat("text", grail.LoggerLevels["info"]),
		// grail.WithLoggerFormat("json", grail.LoggerLevels["debug"]),
	)

	text, err := client.Text(
		ctx,
		"Say hello in one short sentence.",
		// Call options (uncomment to customize):
		// grail.WithTier(grail.Fast),
		// grail.WithProviderOptions(openai.TextOptions{
		// 	MaxTokens:    grail.Pointer[int32](32),
		// 	Temperature:  grail.Pointer[float32](0.4),
		// 	TopP:         grail.Pointer[float32](0.9),
		// 	SystemPrompt: "You are brief and friendly.",
		// }),
	)
	if err != nil {
		if grail.GetErrorCode(err) == grail.InvalidArgument {
			log.Fatalf("fix request: %v", err)
		}
		log.Fatalf("generate text: %v", err)
	}

	fmt.Println(text)
}
