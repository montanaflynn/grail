// Image-understanding demonstrates text generation from image inputs.
// It fetches an image from a URL and uses it along with text prompts to generate
// descriptive text. Supports OpenAI and Gemini providers.
//
// Usage:
//
//	go run examples/image-understanding/main.go
//	go run examples/image-understanding/main.go -openai
//	go run examples/image-understanding/main.go -gemini -debug
package main

import (
	"context"
	"flag"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"sync"

	"log/slog"

	"github.com/montanaflynn/grail"
	"github.com/montanaflynn/grail/providers/gemini"
	"github.com/montanaflynn/grail/providers/openai"
)

// Demonstrates text generation from mixed text+image input (ordered parts).
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

	img, err := fetchImage(ctx, "https://picsum.photos/seed/grail-demo/256")
	if err != nil {
		log.Fatalf("fetch image: %v", err)
	}

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
			text, err := generateWithProvider(ctx, logger, "gemini", "GEMINI_API_KEY", img)
			resultsCh <- result{provider: "gemini", text: text, err: err}
		}()
	}

	if runOpenAI {
		wg.Add(1)
		go func() {
			defer wg.Done()
			text, err := generateWithProvider(ctx, logger, "openai", "OPENAI_API_KEY", img)
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

func generateWithProvider(ctx context.Context, logger *slog.Logger, providerName, envKey string, img []byte) (string, error) {
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
	return generateText(ctx, client, img)
}

func generateText(ctx context.Context, client grail.Client, img []byte) (string, error) {
	res, err := client.Generate(ctx, grail.Request{
		Inputs: []grail.Input{
			grail.InputText("Describe the style of this image."),
			grail.InputImage(img),
			grail.InputText("Keep it short."),
		},
		Output: grail.OutputText(),
	})
	if err != nil {
		return "", err
	}
	text, _ := res.Text()
	return text, nil
}

func fetchImage(ctx context.Context, url string) ([]byte, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, err
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, fmt.Errorf("unexpected status %s", resp.Status)
	}
	return io.ReadAll(resp.Body)
}
