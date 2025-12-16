// Openai-image-options demonstrates OpenAI-specific image generation options.
// It shows how to configure image model (gpt-image-1, gpt-image-1-mini),
// output format (PNG, JPEG, WebP), background (auto, transparent, opaque),
// image size, moderation level, and compression settings.
//
// Usage:
//
//	go run examples/openai-image-options/main.go
//	go run examples/openai-image-options/main.go -model gpt-image-1 -format jpeg -size 1024x1024
//	go run examples/openai-image-options/main.go -model gpt-image-1-mini -background transparent -compression 80
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

// Demonstrates OpenAI image options (output format, background, size, moderation, compression) using ImageOptions.
func main() {
	ctx := context.Background()

	modelFlag := flag.String("model", "gpt-image-1", "openai image model: gpt-image-1|gpt-image-1-mini")
	formatFlag := flag.String("format", "png", "openai output format: png|jpeg|jpg|webp")
	backgroundFlag := flag.String("background", "auto", "openai background: auto|transparent|opaque")
	sizeFlag := flag.String("size", "auto", "openai image size: auto|1024x1024|1536x1024|1024x1536|256x256|512x512|1792x1024|1024x1792")
	moderationFlag := flag.String("moderation", "auto", "openai moderation: auto|low")
	compressionFlag := flag.Int("compression", 100, "openai output compression: 0-100")
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
		openai.WithImageModel(*modelFlag),
	)
	if err != nil {
		log.Fatalf("new openai provider: %v", err)
	}

	client := grail.NewClient(provider, grail.WithLogger(logger))

	res, err := client.Generate(ctx, grail.Request{
		Inputs: []grail.Input{
			grail.InputText("An owl logo icon for a childrens clothing brand"),
		},
		Output: grail.OutputImage(grail.ImageSpec{Count: 1}),
		ProviderOptions: []grail.ProviderOption{
			openai.ImageOptions{
				SystemPrompt: "You're an experienced logo illustrator.",
			},
			openai.WithImageFormat(openai.ImageFormats[strings.ToLower(*formatFlag)]),
			openai.WithImageBackground(openai.ImageBackgrounds[strings.ToLower(*backgroundFlag)]),
			openai.WithImageSize(openai.ImageSizes[strings.ToLower(*sizeFlag)]),
			openai.WithImageModeration(openai.ImageModerations[strings.ToLower(*moderationFlag)]),
			openai.WithImageOutputCompression(*compressionFlag),
		},
	})
	if err != nil {
		log.Fatalf("generate image: %v", err)
	}

	imgInfos := res.ImageOutputs()
	if len(imgInfos) == 0 {
		fmt.Println("no image returned")
		return
	}

	// Convert to imageOutput format for saveImages function
	imgOutputs := make([]imageOutput, len(imgInfos))
	for i, info := range imgInfos {
		imgOutputs[i] = imageOutput{
			Data: info.Data,
			MIME: info.MIME,
		}
	}

	if err := saveImages("examples-output", "openai-image-options", imgOutputs); err != nil {
		log.Fatalf("save images: %v", err)
	}
}

type imageOutput struct {
	Data []byte
	MIME string
}

func saveImages(dir, base string, imgs []imageOutput) error {
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
