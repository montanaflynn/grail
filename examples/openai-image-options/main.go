package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"

	"log/slog"

	"github.com/montanaflynn/grail"
	"github.com/montanaflynn/grail/providers/openai"
)

// Demonstrates OpenAI image options (output format, background) using ImageOptions.
func main() {
	ctx := context.Background()

	formatFlag := flag.String("format", "png", "openai output format: png|jpeg|jpg|webp")
	backgroundFlag := flag.String("background", "auto", "openai background: auto|transparent|opaque")
	debugFlag := flag.Bool("debug", false, "enable debug logging")
	flag.Parse()

	level := slog.LevelInfo
	if *debugFlag {
		level = slog.LevelDebug
	}
	logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{
		Level: level,
	}))

	provider, err := openai.New(
		openai.WithAPIKey(os.Getenv("OPENAI_API_KEY")),
	)
	if err != nil {
		log.Fatalf("new openai provider: %v", err)
	}

	client := grail.NewClient(provider, grail.WithLogger(logger))

	res, err := client.GenerateImage(ctx, grail.ImageRequest{
		Input: []grail.Part{
			grail.Text("An owl logo icon for a childrens clothing brand"),
		},
		Options: grail.ImageOptions{
			SystemPrompt: "You're an experienced logo illustrator.",
		},
		ProviderOptions: []grail.ProviderOption{
			openai.WithImageFormat(openai.ImageFormats[strings.ToLower(*formatFlag)]),
			openai.WithImageBackground(openai.ImageBackgrounds[strings.ToLower(*backgroundFlag)]),
		},
	})
	if err != nil {
		log.Fatalf("generate image: %v", err)
	}

	if len(res.Images) == 0 {
		fmt.Println("no image returned")
		return
	}

	if err := saveImages("examples-output", "openai-image-options", res.Images); err != nil {
		log.Fatalf("save images: %v", err)
	}
}

func saveImages(dir, base string, imgs []grail.ImageOutput) error {
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

func extFromMIME(mime string) string {
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
