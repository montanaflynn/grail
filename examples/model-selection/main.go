// Model-selection demonstrates choosing between best and fast model tiers.
// It shows how to use model constants, tier-based selection, and capability checking.
//
// Usage:
//
//	go run examples/model-selection/main.go
//	go run examples/model-selection/main.go --openai
//	go run examples/model-selection/main.go --gemini
//	go run examples/model-selection/main.go --openai --gemini
//	go run examples/model-selection/main.go --best
//	go run examples/model-selection/main.go --fast
//	go run examples/model-selection/main.go --text
//	go run examples/model-selection/main.go --image
//	go run examples/model-selection/main.go --gemini --image --best --fast
//	go run examples/model-selection/main.go --debug
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"log/slog"

	"github.com/montanaflynn/grail"
	"github.com/montanaflynn/grail/providers/gemini"
	"github.com/montanaflynn/grail/providers/openai"
)

type result struct {
	provider string
	tier     string
	model    string
	text     string
	images   [][]byte
	duration time.Duration
	err      error
}

func main() {
	ctx := context.Background()

	openaiFlag := flag.Bool("openai", false, "use OpenAI provider")
	geminiFlag := flag.Bool("gemini", false, "use Gemini provider")
	bestFlag := flag.Bool("best", false, "only run best tier")
	fastFlag := flag.Bool("fast", false, "only run fast tier")
	textFlag := flag.Bool("text", false, "generate text (default)")
	imageFlag := flag.Bool("image", false, "generate images")
	debugFlag := flag.Bool("debug", false, "enable debug logging")
	flag.Parse()

	// Validate exclusive flags
	if *textFlag && *imageFlag {
		log.Fatal("--text and --image are mutually exclusive")
	}

	level := slog.LevelInfo
	if *debugFlag {
		level = slog.LevelDebug
	}
	logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{
		Level: level,
	}))

	// Default to gemini if no provider specified
	runOpenAI := *openaiFlag
	runGemini := *geminiFlag || (!*openaiFlag && !*geminiFlag)

	// Default to both tiers if neither specified
	runBest := *bestFlag || (!*bestFlag && !*fastFlag)
	runFast := *fastFlag || (!*bestFlag && !*fastFlag)

	// Default to text if neither specified
	generateText := *textFlag || (!*textFlag && !*imageFlag)
	generateImage := *imageFlag

	var wg sync.WaitGroup
	resultsCh := make(chan result, 4)

	textPrompt := "Explain quantum computing in exactly one sentence."
	imagePrompt := "A serene mountain lake at sunset with snow-capped peaks reflected in the water"

	// Run generations
	if runGemini {
		if runBest {
			wg.Add(1)
			go func() {
				defer wg.Done()
				if generateImage {
					resultsCh <- generateImageWithTier(ctx, logger, "gemini", grail.ModelTierBest, imagePrompt)
				} else {
					resultsCh <- generateTextWithTier(ctx, logger, "gemini", grail.ModelTierBest, textPrompt)
				}
			}()
		}
		if runFast {
			wg.Add(1)
			go func() {
				defer wg.Done()
				if generateImage {
					resultsCh <- generateImageWithTier(ctx, logger, "gemini", grail.ModelTierFast, imagePrompt)
				} else {
					resultsCh <- generateTextWithTier(ctx, logger, "gemini", grail.ModelTierFast, textPrompt)
				}
			}()
		}
	}

	if runOpenAI {
		if runBest {
			wg.Add(1)
			go func() {
				defer wg.Done()
				if generateImage {
					resultsCh <- generateImageWithTier(ctx, logger, "openai", grail.ModelTierBest, imagePrompt)
				} else {
					resultsCh <- generateTextWithTier(ctx, logger, "openai", grail.ModelTierBest, textPrompt)
				}
			}()
		}
		if runFast {
			wg.Add(1)
			go func() {
				defer wg.Done()
				if generateImage {
					resultsCh <- generateImageWithTier(ctx, logger, "openai", grail.ModelTierFast, imagePrompt)
				} else {
					resultsCh <- generateTextWithTier(ctx, logger, "openai", grail.ModelTierFast, textPrompt)
				}
			}()
		}
	}

	go func() {
		wg.Wait()
		close(resultsCh)
	}()

	// Collect and display results
	if generateText {
		fmt.Println("\n=== Text Generation Results ===")
	} else {
		fmt.Println("\n=== Image Generation Results ===")
	}
	fmt.Println()

	for res := range resultsCh {
		if res.err != nil {
			log.Printf("[%s/%s] error: %v\n", res.provider, res.tier, res.err)
			continue
		}

		fmt.Printf("[%s/%s] model: %s (%.2fs)\n", res.provider, res.tier, res.model, res.duration.Seconds())

		if res.text != "" {
			fmt.Printf("  → %s\n\n", res.text)
		}

		if len(res.images) > 0 {
			// Save images to files
			outputDir := "examples-output"
			if err := os.MkdirAll(outputDir, 0755); err != nil {
				log.Printf("  → failed to create output dir: %v\n", err)
				continue
			}

			for i, img := range res.images {
				filename := fmt.Sprintf("%s-%s-%s-%d.png", res.provider, res.tier, time.Now().Format("20060102-150405"), i+1)
				path := filepath.Join(outputDir, filename)
				if err := os.WriteFile(path, img, 0644); err != nil {
					log.Printf("  → failed to save image: %v\n", err)
					continue
				}
				fmt.Printf("  → saved: %s\n", path)
			}
			fmt.Println()
		}
	}
}

func generateTextWithTier(ctx context.Context, logger *slog.Logger, providerName string, tier grail.ModelTier, prompt string) result {
	start := time.Now()

	var (
		provider grail.Provider
		model    string
	)

	switch providerName {
	case "gemini":
		p, err := gemini.New(ctx, gemini.WithAPIKey(os.Getenv("GEMINI_API_KEY")))
		if err != nil {
			return result{provider: providerName, tier: string(tier), err: err}
		}
		provider = p
		if tier == grail.ModelTierBest {
			model = p.BestTextModel().Name
		} else {
			model = p.FastTextModel().Name
		}

	case "openai":
		p, err := openai.New(openai.WithAPIKey(os.Getenv("OPENAI_API_KEY")))
		if err != nil {
			return result{provider: providerName, tier: string(tier), err: err}
		}
		provider = p
		if tier == grail.ModelTierBest {
			model = p.BestTextModel().Name
		} else {
			model = p.FastTextModel().Name
		}

	default:
		return result{provider: providerName, tier: string(tier), err: fmt.Errorf("unknown provider")}
	}

	client := grail.NewClient(provider, grail.WithLogger(logger))

	res, err := client.Generate(ctx, grail.Request{
		Inputs: []grail.Input{grail.InputText(prompt)},
		Output: grail.OutputText(),
		Tier:   tier,
	})
	if err != nil {
		return result{provider: providerName, tier: string(tier), model: model, err: err}
	}

	text, _ := res.Text()
	return result{
		provider: providerName,
		tier:     string(tier),
		model:    model,
		text:     text,
		duration: time.Since(start),
	}
}

func generateImageWithTier(ctx context.Context, logger *slog.Logger, providerName string, tier grail.ModelTier, prompt string) result {
	start := time.Now()

	var (
		provider        grail.Provider
		model           string
		reqModel        string
		providerOptions []grail.ProviderOption
	)

	switch providerName {
	case "gemini":
		p, err := gemini.New(ctx, gemini.WithAPIKey(os.Getenv("GEMINI_API_KEY")))
		if err != nil {
			return result{provider: providerName, tier: string(tier), err: err}
		}
		provider = p
		if tier == grail.ModelTierBest {
			model = p.BestImageModel().Name
		} else {
			model = p.FastImageModel().Name
		}
		reqModel = model

	case "openai":
		p, err := openai.New(openai.WithAPIKey(os.Getenv("OPENAI_API_KEY")))
		if err != nil {
			return result{provider: providerName, tier: string(tier), err: err}
		}
		provider = p
		// For OpenAI, the image model is specified via ImageOptions
		// Don't set req.Model - let the provider use its default text model
		// The response will include the actual models used
		var imageModel string
		if tier == grail.ModelTierBest {
			imageModel = p.BestImageModel().Name
		} else {
			imageModel = p.FastImageModel().Name
		}
		model = imageModel // Will be updated from response
		providerOptions = []grail.ProviderOption{
			openai.ImageOptions{Model: imageModel},
		}
		// reqModel stays empty - provider uses its default

	default:
		return result{provider: providerName, tier: string(tier), err: fmt.Errorf("unknown provider")}
	}

	client := grail.NewClient(provider, grail.WithLogger(logger))

	req := grail.Request{
		Inputs:          []grail.Input{grail.InputText(prompt)},
		Output:          grail.OutputImage(grail.ImageSpec{Count: 1}),
		ProviderOptions: providerOptions,
		Model:           reqModel,
	}

	res, err := client.Generate(ctx, req)
	if err != nil {
		return result{provider: providerName, tier: string(tier), model: model, err: err}
	}

	// Build model string from response (provider reports actual models used)
	if len(res.Provider.Models) > 0 {
		var modelNames []string
		for _, m := range res.Provider.Models {
			modelNames = append(modelNames, m.Name)
		}
		model = strings.Join(modelNames, ",")
	}

	images, _ := res.Images()
	return result{
		provider: providerName,
		tier:     string(tier),
		model:    model,
		images:   images,
		duration: time.Since(start),
	}
}
