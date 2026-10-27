// Streaming demonstrates printing text as it is generated.
// It uses the OpenAI provider with the API key from the OPENAI_API_KEY
// environment variable, streams a short story from the fast tier, and prints
// the token usage from the final event.
//
// Usage:
//
//	go run examples/streaming/main.go
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/montanaflynn/grail"
	"github.com/montanaflynn/grail/providers/openai"
)

func main() {
	ctx := context.Background()

	// Uses default API key from environment variable OPENAI_API_KEY if set.
	// gemini.New(ctx) and anthropic.New() stream the same way.
	provider, err := openai.New()
	if err != nil {
		log.Fatalf("new provider: %v", err)
	}

	client := grail.NewClient(provider)

	// Stream yields text deltas, then a final event (Done) with usage.
	// Breaking out of the loop cancels the request.
	for ev, err := range client.Stream(ctx, grail.Request{
		Inputs: []grail.Input{grail.InputText("Tell a three-sentence story about a lighthouse.")},
		Output: grail.OutputText(),
		Tier:   grail.Fast,
	}) {
		if err != nil {
			log.Fatalf("stream: %v", err)
		}
		fmt.Print(ev.Text)
		if ev.Done {
			fmt.Printf("\n\n%s: %d tokens\n", ev.Provider.Name, ev.Usage.TotalTokens)
		}
	}
}
