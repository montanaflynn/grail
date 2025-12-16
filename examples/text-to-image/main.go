// Text-to-image demonstrates image generation from text prompts.
// It can run with OpenAI, Gemini, or both providers in parallel, generating images
// from text descriptions and saving them to the examples-output directory.
//
// Usage:
//
//	go run examples/text-to-image/main.go
//	go run examples/text-to-image/main.go -openai
//	go run examples/text-to-image/main.go -gemini -debug
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"sync"

	"log/slog"

	"github.com/montanaflynn/grail"
	"github.com/montanaflynn/grail/providers/gemini"
	"github.com/montanaflynn/grail/providers/openai"
)

// Demonstrates text -> image generation.
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

	// Determine which providers to run.
	runOpenAI := *openaiFlag
	runGemini := *geminiFlag || (!*openaiFlag && !*geminiFlag) // default gemini if none set

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
			images, err := generateWithProvider(ctx, logger, "gemini", "GEMINI_API_KEY")
			resultsCh <- result{provider: "gemini", images: images, err: err}
		}()
	}

	if runOpenAI {
		wg.Add(1)
		go func() {
			defer wg.Done()
			images, err := generateWithProvider(ctx, logger, "openai", "OPENAI_API_KEY")
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
		if err := saveImages("examples-output", fmt.Sprintf("text-to-image-%s", res.provider), res.images); err != nil {
			log.Printf("%s: save images: %v", res.provider, err)
		}
	}
}

func generateWithProvider(ctx context.Context, logger *slog.Logger, providerName, envKey string) ([]grail.ImageOutputInfo, error) {
	var (
		provider grail.Provider
		err      error
	)

	switch providerName {
	case "gemini":
		provider, err = gemini.New(
			ctx,
			gemini.WithAPIKey(os.Getenv(envKey)),
		)
	case "openai":
		provider, err = openai.New(
			openai.WithAPIKey(os.Getenv(envKey)),
		)
	default:
		return nil, fmt.Errorf("unknown provider %q", providerName)
	}
	if err != nil {
		return nil, fmt.Errorf("new %s provider: %w", providerName, err)
	}

	client := grail.NewClient(provider, grail.WithLogger(logger))
	res, err := client.Generate(ctx, grail.Request{
		Inputs: []grail.Input{
			grail.InputText("An image of a cozy cabin in the woods at dusk, in watercolor style"),
			grail.InputText("With the words Merry Christmas written in the top right corner"),
		},
		Output: grail.OutputImage(grail.ImageSpec{Count: 1}),
	})
	if err != nil {
		return nil, err
	}
	return res.ImageOutputs(), nil
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
