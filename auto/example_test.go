package auto_test

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/montanaflynn/grail"
	"github.com/montanaflynn/grail/auto"
)

// The README quick start. It needs at least one of ANTHROPIC_API_KEY,
// OPENAI_API_KEY or GEMINI_API_KEY and calls the providers, so it only
// compiles.
func ExampleFromEnv() {
	ctx := context.Background()

	// Uses whichever of ANTHROPIC_API_KEY, OPENAI_API_KEY and GEMINI_API_KEY are set.
	client, err := auto.FromEnv(ctx)
	if err != nil {
		log.Fatal(err)
	}

	text, err := client.Text(ctx, "Write a haiku about Go.")
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(text)

	png, err := client.Image(ctx, "A gopher surfing at sunset", grail.WithTier(grail.Fast))
	if err != nil {
		log.Fatal(err)
	}
	if err := os.WriteFile("gopher.png", png, 0o644); err != nil {
		log.Fatal(err)
	}

	type City struct {
		Name    string `json:"name"`
		Country string `json:"country"`
	}
	city, err := grail.JSON[City](ctx, client, "The largest city in Japan")
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(city.Name, city.Country)
}
