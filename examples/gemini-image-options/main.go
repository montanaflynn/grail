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
	"github.com/montanaflynn/grail/providers/gemini"
)

// Demonstrates Gemini image options (aspect ratio, size) using ImageOptions.
func main() {
	ctx := context.Background()

	aspectRatioFlag := flag.String("aspect-ratio", "16:9", "gemini aspect ratio: 1:1|2:3|3:2|3:4|4:3|4:5|5:4|9:16|16:9|21:9")
	sizeFlag := flag.String("size", "2K", "gemini image size: 1K|2K|4K")
	debugFlag := flag.Bool("debug", false, "enable debug logging")
	flag.Parse()

	level := slog.LevelInfo
	if *debugFlag {
		level = slog.LevelDebug
	}
	logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{
		Level: level,
	}))

	provider, err := gemini.New(
		ctx,
		gemini.WithAPIKey(os.Getenv("GEMINI_API_KEY")),
	)
	if err != nil {
		log.Fatalf("new gemini provider: %v", err)
	}

	client := grail.NewClient(provider, grail.WithLogger(logger))

	res, err := client.Generate(ctx, grail.Request{
		Inputs: []grail.Input{
			grail.InputText("An owl logo icon for a childrens clothing brand"),
		},
		Output: grail.OutputImage(grail.ImageSpec{Count: 1}),
		ProviderOptions: []grail.ProviderOption{
			gemini.ImageOptions{
				SystemPrompt: "You're an experienced logo illustrator.",
			},
			gemini.WithImageAspectRatio(gemini.ImageAspectRatios[strings.ToLower(*aspectRatioFlag)]),
			gemini.WithImageSize(gemini.ImageSizes[strings.ToUpper(*sizeFlag)]),
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

	if err := saveImages("examples-output", "gemini-image-options", imgOutputs); err != nil {
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
