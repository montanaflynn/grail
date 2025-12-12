package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/montanaflynn/grail"
	"github.com/montanaflynn/grail/providers/openai"
)

// Minimal text generation:
// - Defaults: uses provider default model (gpt-5.1 for OpenAI), default logger, API key from OPENAI_API_KEY.
// - Uncomment the logger options and TextOptions blocks below to customize.
// - For images, ImageRequest supports ProviderOptions (e.g., openai.WithImageFormat/WithImageBackground).
func main() {
	ctx := context.Background()

	// Uses default API key from environment variable OPENAI_API_KEY if set.
	provider, err := openai.New(
		openai.WithAPIKey(os.Getenv("UNSET_API_KEY")),
	// openai.WithAPIKeyFromEnv("UNSET_API_KEY"),
	// openai.WithTextModel("gpt-5.2"),
	// openai.WithImageModel("gpt-image-1"),
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

	res, err := client.GenerateText(ctx, grail.TextRequest{
		Input: []grail.Part{
			grail.Text("Say hello in one short sentence."),
		},
		// Options: grail.TextOptions{
		// 	Model:        "gpt-5.1",
		// 	MaxTokens:    grail.Pointer[int32](32),
		// 	Temperature:  grail.Pointer[float32](0.4),
		// 	TopP:         grail.Pointer[float32](0.9),
		// 	SystemPrompt: "You are brief and friendly.",
		// },
		// ProviderOptions: []grail.ProviderOption{
		// 	// Provider-specific knobs (e.g., OpenAI image options):
		// 	// openai.WithImageFormat(openai.ImageFormatPNG),
		// 	// openai.WithImageBackground(openai.ImageBackgroundTransparent),
		// },
	})
	if err != nil {
		if grail.IsCode(err, grail.CodeInvalidInput) {
			log.Fatalf("fix request: %v", err)
		}
		log.Fatalf("generate text: %v", err)
	}

	fmt.Println(res.Text)
}
