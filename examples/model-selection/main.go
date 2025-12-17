// Model-selection demonstrates choosing between best and fast model tiers.
// It shows how to use model constants, tier-based selection, and capability checking.
//
// Usage:
//
//	go run examples/model-selection/main.go
//	go run examples/model-selection/main.go -openai
//	go run examples/model-selection/main.go -gemini
//	go run examples/model-selection/main.go -openai -gemini
//	go run examples/model-selection/main.go -best      # only best tier
//	go run examples/model-selection/main.go -fast      # only fast tier
//	go run examples/model-selection/main.go -debug
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
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
	duration time.Duration
	err      error
}

func main() {
	ctx := context.Background()

	openaiFlag := flag.Bool("openai", false, "use OpenAI provider")
	geminiFlag := flag.Bool("gemini", false, "use Gemini provider")
	bestFlag := flag.Bool("best", false, "only run best tier")
	fastFlag := flag.Bool("fast", false, "only run fast tier")
	debugFlag := flag.Bool("debug", false, "enable debug logging")
	flag.Parse()

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

	var wg sync.WaitGroup
	resultsCh := make(chan result, 4)

	prompt := "Explain quantum computing in exactly one sentence."

	// Run generations
	if runGemini {
		if runBest {
			wg.Add(1)
			go func() {
				defer wg.Done()
				r := generateWithTier(ctx, logger, "gemini", grail.ModelTierBest, prompt)
				resultsCh <- r
			}()
		}
		if runFast {
			wg.Add(1)
			go func() {
				defer wg.Done()
				r := generateWithTier(ctx, logger, "gemini", grail.ModelTierFast, prompt)
				resultsCh <- r
			}()
		}
	}

	if runOpenAI {
		if runBest {
			wg.Add(1)
			go func() {
				defer wg.Done()
				r := generateWithTier(ctx, logger, "openai", grail.ModelTierBest, prompt)
				resultsCh <- r
			}()
		}
		if runFast {
			wg.Add(1)
			go func() {
				defer wg.Done()
				r := generateWithTier(ctx, logger, "openai", grail.ModelTierFast, prompt)
				resultsCh <- r
			}()
		}
	}

	go func() {
		wg.Wait()
		close(resultsCh)
	}()

	// Collect and display results
	fmt.Println("\n=== Model Selection Results ===\n")

	for res := range resultsCh {
		if res.err != nil {
			log.Printf("[%s/%s] error: %v\n", res.provider, res.tier, res.err)
			continue
		}
		fmt.Printf("[%s/%s] model: %s (%.2fs)\n", res.provider, res.tier, res.model, res.duration.Seconds())
		fmt.Printf("  → %s\n\n", res.text)
	}

	// Show available models
	fmt.Println("=== Available Models ===\n")
	showAvailableModels()
}

func generateWithTier(ctx context.Context, logger *slog.Logger, providerName string, tier grail.ModelTier, prompt string) result {
	start := time.Now()

	var (
		provider grail.Provider
		err      error
		model    string
	)

	switch providerName {
	case "gemini":
		p, err := gemini.New(ctx, gemini.WithAPIKey(os.Getenv("GEMINI_API_KEY")))
		if err != nil {
			return result{provider: providerName, tier: string(tier), err: err}
		}
		provider = p
		// Get the model that will be used for this tier
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
		// Get the model that will be used for this tier
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
		Tier:   tier, // Let the provider resolve the model based on tier
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

func showAvailableModels() {
	// OpenAI models
	fmt.Println("OpenAI:")
	fmt.Printf("  Best Text:  %s\n", openai.GPT5_2.Name)
	fmt.Printf("    Capabilities: Text=%v, ImageUnderstanding=%v, PDF=%v, JSON=%v\n",
		openai.GPT5_2.Capabilities.TextGeneration,
		openai.GPT5_2.Capabilities.ImageUnderstanding,
		openai.GPT5_2.Capabilities.PDFUnderstanding,
		openai.GPT5_2.Capabilities.JSONOutput)

	fmt.Printf("  Fast Text:  %s\n", openai.GPT4o.Name)
	fmt.Printf("  Best Image: %s\n", openai.GPTImage1.Name)
	fmt.Printf("    Capabilities: ImageGeneration=%v, ImageUnderstanding=%v\n",
		openai.GPTImage1.Capabilities.ImageGeneration,
		openai.GPTImage1.Capabilities.ImageUnderstanding)
	fmt.Printf("  Fast Image: %s\n\n", openai.GPTImage1Mini.Name)

	// Gemini models
	fmt.Println("Gemini:")
	fmt.Printf("  Best Text:  %s\n", gemini.Gemini3Pro.Name)
	fmt.Printf("    Capabilities: Text=%v, ImageUnderstanding=%v, PDF=%v, JSON=%v\n",
		gemini.Gemini3Pro.Capabilities.TextGeneration,
		gemini.Gemini3Pro.Capabilities.ImageUnderstanding,
		gemini.Gemini3Pro.Capabilities.PDFUnderstanding,
		gemini.Gemini3Pro.Capabilities.JSONOutput)

	fmt.Printf("  Fast Text:  %s\n", gemini.Gemini3Flash.Name)
	fmt.Printf("  Best Image: %s\n", gemini.Gemini3ProImage.Name)
	fmt.Printf("    Capabilities: ImageGeneration=%v, ImageUnderstanding=%v\n",
		gemini.Gemini3ProImage.Capabilities.ImageGeneration,
		gemini.Gemini3ProImage.Capabilities.ImageUnderstanding)
	fmt.Printf("  Fast Image: %s\n\n", gemini.Gemini25FlashImage.Name)

	// Other Gemini models
	fmt.Println("  Other Gemini models (not best/fast):")
	fmt.Printf("    %s\n", gemini.Gemini25Flash.Name)
	fmt.Printf("    %s\n", gemini.Gemini25FlashLite.Name)
}
