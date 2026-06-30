// Command listmodels prints model IDs from provider APIs for auditing grail defaults.
package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"sort"
	"strings"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	"google.golang.org/genai"
)

func main() {
	provider := flag.String("provider", "all", "Provider to list: openai, gemini, or all")
	flag.Parse()

	ctx := context.Background()
	var failed bool

	switch strings.ToLower(*provider) {
	case "openai":
		failed = listOpenAI(ctx) || failed
	case "gemini":
		failed = listGemini(ctx) || failed
	case "all":
		failed = listOpenAI(ctx) || listGemini(ctx) || failed
	default:
		fmt.Fprintf(os.Stderr, "Error: unknown provider %q\n", *provider)
		fmt.Fprintf(os.Stderr, "Examples:\n")
		fmt.Fprintf(os.Stderr, "  listmodels -provider openai\n")
		fmt.Fprintf(os.Stderr, "  listmodels -provider gemini\n")
		fmt.Fprintf(os.Stderr, "  listmodels -provider all\n")
		os.Exit(2)
	}

	if failed {
		os.Exit(1)
	}
}

func listOpenAI(ctx context.Context) bool {
	key := strings.TrimSpace(os.Getenv("OPENAI_API_KEY"))
	if key == "" {
		fmt.Fprintln(os.Stderr, "Error: OPENAI_API_KEY is not set")
		fmt.Fprintln(os.Stderr, "Example: OPENAI_API_KEY=sk-... listmodels -provider openai")
		return true
	}

	client := openai.NewClient(option.WithAPIKey(key))
	page, err := client.Models.List(ctx)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error listing OpenAI models: %v\n", err)
		return true
	}

	ids := make([]string, 0, len(page.Data))
	for _, m := range page.Data {
		if m.ID != "" {
			ids = append(ids, m.ID)
		}
	}
	sort.Strings(ids)

	fmt.Println("# openai")
	for _, id := range ids {
		fmt.Println(id)
	}
	return false
}

func listGemini(ctx context.Context) bool {
	key := strings.TrimSpace(os.Getenv("GEMINI_API_KEY"))
	if key == "" {
		fmt.Fprintln(os.Stderr, "Error: GEMINI_API_KEY is not set")
		fmt.Fprintln(os.Stderr, "Example: GEMINI_API_KEY=... listmodels -provider gemini")
		return true
	}

	client, err := genai.NewClient(ctx, &genai.ClientConfig{
		Backend: genai.BackendGeminiAPI,
		APIKey:  key,
	})
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error creating Gemini client: %v\n", err)
		return true
	}

	var names []string
	for model, err := range client.Models.All(ctx) {
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error listing Gemini models: %v\n", err)
			return true
		}
		if model.Name != "" {
			names = append(names, model.Name)
		}
	}
	sort.Strings(names)

	fmt.Println("# gemini")
	for _, name := range names {
		fmt.Println(name)
	}
	return false
}
