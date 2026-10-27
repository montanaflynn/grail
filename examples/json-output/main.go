// Json-output demonstrates typed structured output with grail.JSON.
// The JSON Schema is derived from a Go struct, sent to the provider's native
// structured-output API, and the response is decoded into the struct.
//
// Usage:
//
//	go run examples/json-output/main.go
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/montanaflynn/grail"
	"github.com/montanaflynn/grail/providers/openai"
)

// Recipe is the shape of the output. Fields tagged omitempty (and pointers)
// may come back as null, which decodes to the zero value.
type Recipe struct {
	Name        string       `json:"name" jsonschema:"description=Name of the dish"`
	Difficulty  string       `json:"difficulty" jsonschema:"enum=easy|medium|hard"`
	Servings    int          `json:"servings"`
	Ingredients []Ingredient `json:"ingredients"`
	Steps       []string     `json:"steps"`
	Tip         string       `json:"tip,omitempty" jsonschema:"description=Optional serving tip"`
}

type Ingredient struct {
	Item     string `json:"item"`
	Quantity string `json:"quantity" jsonschema:"description=Amount with units, e.g. 200 g"`
}

func main() {
	ctx := context.Background()

	// Uses default API key from environment variable OPENAI_API_KEY if set.
	provider, err := openai.New()
	if err != nil {
		log.Fatalf("new provider: %v", err)
	}
	client := grail.NewClient(provider)

	recipe, err := grail.JSON[Recipe](ctx, client, "A simple pancake recipe.", grail.WithTier(grail.Fast))
	if err != nil {
		log.Fatalf("generate JSON: %v", err)
	}

	fmt.Printf("%s (%s, serves %d)\n", recipe.Name, recipe.Difficulty, recipe.Servings)
	for _, ing := range recipe.Ingredients {
		fmt.Printf("- %s %s\n", ing.Quantity, ing.Item)
	}
	for i, step := range recipe.Steps {
		fmt.Printf("%d. %s\n", i+1, step)
	}
	if recipe.Tip != "" {
		fmt.Println("Tip:", recipe.Tip)
	}
}
