package main

import (
	"context"
	"flag"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"strings"
	"sync"

	"log/slog"

	"github.com/montanaflynn/grail"
	"github.com/montanaflynn/grail/providers/gemini"
	"github.com/montanaflynn/grail/providers/openai"
)

// Demonstrates text generation from PDF input.
func main() {
	ctx := context.Background()

	openaiFlag := flag.Bool("openai", false, "use OpenAI provider")
	geminiFlag := flag.Bool("gemini", false, "use Gemini provider")
	debugFlag := flag.Bool("debug", false, "enable debug logging")
	urlFlag := flag.String("url", "", "PDF URL to fetch (optional)")
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

	var pdfData []byte
	var filename string
	var err error

	if *urlFlag != "" {
		pdfData, err = fetchPDF(ctx, *urlFlag)
		if err != nil {
			log.Fatalf("fetch PDF: %v", err)
		}
		filename = extractFilenameFromURL(*urlFlag)
	} else {
		// Use a sample PDF URL
		pdfData, err = fetchPDF(ctx, "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf")
		if err != nil {
			log.Fatalf("fetch PDF: %v", err)
		}
		filename = "dummy.pdf"
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
			text, err := generateWithProvider(ctx, logger, "gemini", "GEMINI_API_KEY", pdfData, filename)
			resultsCh <- result{provider: "gemini", text: text, err: err}
		}()
	}

	if runOpenAI {
		wg.Add(1)
		go func() {
			defer wg.Done()
			text, err := generateWithProvider(ctx, logger, "openai", "OPENAI_API_KEY", pdfData, filename)
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

func generateWithProvider(ctx context.Context, logger *slog.Logger, providerName, envKey string, pdfData []byte, filename string) (string, error) {
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
	return generateText(ctx, client, pdfData, filename)
}

func generateText(ctx context.Context, client grail.Client, pdfData []byte, filename string) (string, error) {
	pdfInput := grail.InputPDF(pdfData, grail.WithFileName(filename))

	res, err := client.Generate(ctx, grail.Request{
		Inputs: []grail.Input{
			grail.InputText("Summarize the key points from this document."),
			pdfInput,
		},
		Output: grail.OutputText(),
	})
	if err != nil {
		return "", err
	}
	text, _ := res.Text()
	return text, nil
}

func fetchPDF(ctx context.Context, urlStr string) ([]byte, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, urlStr, nil)
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

func extractFilenameFromURL(urlStr string) string {
	// Simple extraction - in practice you might want more robust parsing
	parts := strings.Split(urlStr, "/")
	if len(parts) > 0 {
		last := parts[len(parts)-1]
		if strings.Contains(last, ".") {
			return last
		}
	}
	return "document.pdf"
}
