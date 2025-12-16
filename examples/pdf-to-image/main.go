// Pdf-to-image demonstrates image generation from PDF documents.
// It reads a PDF from a file path or URL and generates images (e.g., infographics)
// based on the PDF content. Supports OpenAI and Gemini providers.
//
// Usage:
//
//	go run examples/pdf-to-image/main.go
//	go run examples/pdf-to-image/main.go -path document.pdf
//	go run examples/pdf-to-image/main.go -url https://example.com/document.pdf -openai
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

// Demonstrates image generation from PDF input.
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
		urlStr := *pdfURL
		if urlStr == "" {
			urlStr = defaultPDFURL
		}
		pdfData, err = fetchPDF(ctx, urlStr)
		if err != nil {
			log.Fatalf("fetch PDF from URL: %v", err)
		}
		// Extract filename from URL
		filename = extractFilenameFromURL(urlStr)
	}

	runOpenAI := *openaiFlag
	runGemini := *geminiFlag || (!*openaiFlag && !*geminiFlag)

	type result struct {
		provider string
		images   []grail.ImageOutputInfo
		err      error
	}

	var wg sync.WaitGroup
	resultsCh := make(chan result, 2)

	if runGemini {
		wg.Add(1)
		go func() {
			defer wg.Done()
			images, err := generateWithProvider(ctx, logger, "gemini", "GEMINI_API_KEY", pdfData, filename)
			resultsCh <- result{provider: "gemini", images: images, err: err}
		}()
	}

	if runOpenAI {
		wg.Add(1)
		go func() {
			defer wg.Done()
			images, err := generateWithProvider(ctx, logger, "openai", "OPENAI_API_KEY", pdfData, filename)
			resultsCh <- result{provider: "openai", images: images, err: err}
		}()
	}

	go func() {
		wg.Wait()
		close(resultsCh)
	}()

	for res := range resultsCh {
		if res.err != nil {
			log.Printf("%s: generate image error: %v", res.provider, res.err)
			continue
		}
		if len(res.images) == 0 {
			log.Printf("%s: no image returned", res.provider)
			continue
		}
		if err := saveImages("examples-output", fmt.Sprintf("pdf-to-image-%s", res.provider), res.images); err != nil {
			log.Printf("%s: save images: %v", res.provider, err)
		}
	}
}

func generateWithProvider(ctx context.Context, logger *slog.Logger, providerName, envKey string, pdfData []byte, filename string) ([]grail.ImageOutputInfo, error) {
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
		return nil, fmt.Errorf("unknown provider %q", providerName)
	}
	if err != nil {
		return nil, fmt.Errorf("new %s provider: %w", providerName, err)
	}

	client := grail.NewClient(provider, grail.WithLogger(logger))
	return generateImage(ctx, client, pdfData, filename)
}

func generateImage(ctx context.Context, client grail.Client, pdfData []byte, filename string) ([]grail.ImageOutputInfo, error) {
	pdfInput := grail.InputPDF(pdfData, grail.WithFileName(filename))

	res, err := client.Generate(ctx, grail.Request{
		Inputs: []grail.Input{
			grail.InputText("Create an infographic that summarizes the key concepts from this Bitcoin whitepaper. Make it visually appealing with clear sections, icons, and a modern design."),
			pdfInput,
		},
		Output: grail.OutputImage(grail.ImageSpec{Count: 1}),
	})
	if err != nil {
		return nil, err
	}
	return res.ImageOutputs(), nil
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

// saveImages writes all returned images to disk with numbered filenames.
func saveImages(dir, base string, imgs []grail.ImageOutputInfo) error {
	extFromMIME := func(mime string) string {
		switch mime {
		case "image/jpeg", "image/jpg":
			return ".jpg"
		case "image/png":
			return ".png"
		case "image/webp":
			return ".webp"
		default:
			return ".bin"
		}
	}

	if err := os.MkdirAll(dir, 0o755); err != nil {
		return fmt.Errorf("make output dir: %w", err)
	}
	for i, img := range imgs {
		ext := extFromMIME(img.MIME)
		outPath := filepath.Join(dir, fmt.Sprintf("%s-%02d%s", base, i+1, ext))
		if err := os.WriteFile(outPath, img.Data, 0o644); err != nil {
			return fmt.Errorf("write image %d: %w", i, err)
		}
		fmt.Printf("saved image %d to %s (mime=%s, bytes=%d)\n", i+1, outPath, img.MIME, len(img.Data))
	}
	return nil
}
