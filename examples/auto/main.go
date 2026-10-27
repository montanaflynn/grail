// Auto demonstrates zero-config setup with auto.FromEnv.
// It builds a client from whichever of ANTHROPIC_API_KEY, OPENAI_API_KEY and
// GEMINI_API_KEY are set, then generates text. Set GRAIL_PROVIDERS (for
// example "gemini,openai") to choose the providers and their order.
//
// Usage:
//
//	go run examples/auto/main.go
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/montanaflynn/grail/auto"
)

func main() {
	ctx := context.Background()

	client, err := auto.FromEnv(ctx)
	if err != nil {
		log.Fatalf("new client: %v", err)
	}

	text, err := client.Text(ctx, "Say hello in one short sentence.")
	if err != nil {
		log.Fatalf("generate text: %v", err)
	}

	fmt.Println(text)
}
