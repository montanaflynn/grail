package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"sync"

	"log/slog"

	"github.com/montanaflynn/grail"
	"github.com/montanaflynn/grail/providers/gemini"
	"github.com/montanaflynn/grail/providers/openai"
)

// Demonstrates text generation with provider selection.
func main() {
	ctx := context.Background()

	openaiFlag := flag.Bool("openai", false, "use OpenAI provider")
	geminiFlag := flag.Bool("gemini", false, "use Gemini provider")
	debugFlag := flag.Bool("debug", false, "enable debug logging")
	flag.Parse()

	level := slog.LevelInfo
	if *debugFlag {
		level = slog.LevelDebug
	}
	logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{
		Level: level,
	}))

	runOpenAI := *openaiFlag
	runGemini := *geminiFlag || (!*openaiFlag && !*geminiFlag)

	type result struct {
		provider string
		text     string
		err      error
	}

	var wg sync.WaitGroup
	resultsCh := make(chan result, 2)

	if runGemini {
		wg.Add(1)
		go func() {
			defer wg.Done()
			text, err := generateWithProvider(ctx, logger, "gemini", "GEMINI_API_KEY")
			resultsCh <- result{provider: "gemini", text: text, err: err}
		}()
	}

	if runOpenAI {
		wg.Add(1)
		go func() {
			defer wg.Done()
			text, err := generateWithProvider(ctx, logger, "openai", "OPENAI_API_KEY")
			resultsCh <- result{provider: "openai", text: text, err: err}
		}()
	}

	go func() {
		wg.Wait()
		close(resultsCh)
	}()

	for res := range resultsCh {
		if res.err != nil {
			log.Printf("%s: generate text error: %v", res.provider, res.err)
			continue
		}
		if res.text == "" {
			log.Printf("%s: empty text response", res.provider)
			continue
		}
		fmt.Printf("[%s] %s\n", res.provider, res.text)
	}
}

func generateWithProvider(ctx context.Context, logger *slog.Logger, providerName, envKey string) (string, error) {
	key := os.Getenv(envKey)

	var (
		provider grail.Provider
		err      error
	)

	switch providerName {
	case "gemini":
		provider, err = gemini.New(
			ctx,
			gemini.WithAPIKey(key),
		)
	case "openai":
		provider, err = openai.New(
			openai.WithAPIKey(key),
		)
	default:
		return "", fmt.Errorf("unknown provider %q", providerName)
	}
	if err != nil {
		return "", fmt.Errorf("new %s provider: %w", providerName, err)
	}

	client := grail.NewClient(provider, grail.WithLogger(logger))
	return generateText(ctx, client)
}

func generateText(ctx context.Context, client grail.Client) (string, error) {
	res, err := client.Generate(ctx, grail.Request{
		Inputs: []grail.Input{
			grail.InputText("Explain how AI works in a few words"),
		},
		Output: grail.OutputText(),
	})
	if err != nil {
		return "", err
	}
	text, _ := res.Text()
	return text, nil
}
