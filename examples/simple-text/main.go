package main

import (
	"context"
	"fmt"
	"log"

	"github.com/montanaflynn/grail"
	"github.com/montanaflynn/grail/providers/openai"
)

// Minimal text generation:
// - Defaults: uses provider default model (gpt-5.1 for OpenAI), default logger, API key from OPENAI_API_KEY.
// - Uncomment the logger options and provider options blocks below to customize.
func main() {
	ctx := context.Background()

	// Uses default API key from environment variable OPENAI_API_KEY if set.
	provider, err := openai.New(
	// openai.WithAPIKey("sk-proj-*************"),
	// openai.WithAPIKeyFromEnv("OPENAI_API_KEY"),
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

	res, err := client.Generate(ctx, grail.Request{
		Inputs: []grail.Input{
			grail.InputText("Say hello in one short sentence."),
		},
		Output: grail.OutputText(),
		// ProviderOptions: []grail.ProviderOption{
		// 	openai.TextOptions{
		// 		Model:        "gpt-5.1",
		// 		MaxTokens:    grail.Pointer[int32](32),
		// 		Temperature:  grail.Pointer[float32](0.4),
		// 		TopP:         grail.Pointer[float32](0.9),
		// 		SystemPrompt: "You are brief and friendly.",
		// 	},
		// },
	})
	if err != nil {
		if grail.GetErrorCode(err) == grail.InvalidArgument {
			log.Fatalf("fix request: %v", err)
		}
		log.Fatalf("generate text: %v", err)
	}

	text, _ := res.Text()
	fmt.Println(text)
}
