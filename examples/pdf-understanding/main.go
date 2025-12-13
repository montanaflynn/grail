package main

import (
	"context"
	"flag"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"sync"

	"log/slog"

	"github.com/montanaflynn/grail"
	"github.com/montanaflynn/grail/providers/gemini"
	"github.com/montanaflynn/grail/providers/openai"
)

const defaultPDFURL = "https://bitcoin.org/bitcoin.pdf"

// Demonstrates text generation from PDF input.
func main() {
	ctx := context.Background()

	openaiFlag := flag.Bool("openai", false, "use OpenAI provider")
	geminiFlag := flag.Bool("gemini", false, "use Gemini provider")
	debugFlag := flag.Bool("debug", false, "enable debug logging")
	pdfPath := flag.String("path", "", "path to PDF file (mutually exclusive with -url)")
	pdfURL := flag.String("url", "", "URL to PDF file (mutually exclusive with -path)")
	flag.Parse()

	// Validate mutually exclusive flags
	if *pdfPath != "" && *pdfURL != "" {
		log.Fatal("cannot specify both -path and -url flags")
	}

	level := slog.LevelInfo
	if *debugFlag {
		level = slog.LevelDebug
	}
	logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{
		Level: level,
	}))

	var pdfData []byte
	var filename string
	var err error

	if *pdfPath != "" {
		// Read from file path
		pdfData, err = os.ReadFile(*pdfPath)
		if err != nil {
			log.Fatalf("read PDF file: %v", err)
		}
		// Extract filename from path
		filename = filepath.Base(*pdfPath)
	} else {
		// Use URL (default or provided)
		url := *pdfURL
		if url == "" {
			url = defaultPDFURL
		}
		pdfData, err = fetchPDF(ctx, url)
		if err != nil {
			log.Fatalf("fetch PDF from URL: %v", err)
		}
		// Extract filename from URL
		filename = extractFilenameFromURL(url)
	}

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

func generateText(ctx context.Context, client *grail.Client, pdfData []byte, filename string) (string, error) {
	res, err := client.GenerateText(ctx, grail.TextRequest{
		Input: []grail.Part{
			grail.Text("Summarize the key points from this document."),
			grail.PDFWithFilename(pdfData, "application/pdf", filename),
		},
	})
	if err != nil {
		return "", err
	}
	return res.Text, nil
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
	parsedURL, err := url.Parse(urlStr)
	if err != nil {
		return "document.pdf"
	}

	// Get filename from path
	path := parsedURL.Path
	if path == "" || path == "/" {
		return "document.pdf"
	}

	// Extract the last component of the path
	filename := filepath.Base(path)

	// If it doesn't end with .pdf, add it
	if !strings.HasSuffix(strings.ToLower(filename), ".pdf") {
		if filename == "" || filename == "." {
			return "document.pdf"
		}
		return filename + ".pdf"
	}

	if filename == "" {
		return "document.pdf"
	}

	return filename
}
