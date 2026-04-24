# grail

![CI](https://img.shields.io/github/actions/workflow/status/montanaflynn/grail/ci.yml)
[![Go Reference](https://img.shields.io/badge/go.dev-reference-00ADD8)](https://pkg.go.dev/github.com/montanaflynn/grail)

A lightweight Go SDK that unifies multiple AI providers behind a consistent interface for text and image generation.

## Design Goals

- **One client** for text & image generation across providers
- **Provider-agnostic** by default, extensible when needed
- **Multimodal-first**: ordered text + image inputs
- **Flexible Configuration**: Client, provider and per-request options
- **Type-Safe Errors**: Typed error codes for predictable error handling

## Installation

```bash
go get github.com/montanaflynn/grail
```

## Quick Start

```go
package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/montanaflynn/grail"
	"github.com/montanaflynn/grail/providers/openai"
	// Swap providers by importing another one instead:
	// "github.com/montanaflynn/grail/providers/gemini"
)

func main() {
	ctx := context.Background()

	// Uses OPENAI_API_KEY from the environment.
	// For Gemini: provider, err := gemini.New(ctx)    (uses GEMINI_API_KEY)
	provider, err := openai.New()
	if err != nil {
		log.Fatal(err)
	}
	client := grail.NewClient(provider)

	// Generate text.
	textRes, err := client.Generate(ctx, grail.Request{
		Inputs: []grail.Input{grail.InputText("Write a haiku about Go.")},
		Output: grail.OutputText(),
	})
	if err != nil {
		log.Fatal(err)
	}
	text, _ := textRes.Text()
	fmt.Println(text)

	// Generate an image.
	imgRes, err := client.Generate(ctx, grail.Request{
		Inputs: []grail.Input{grail.InputText("A beautiful sunset over the ocean.")},
		Output: grail.OutputImage(grail.ImageSpec{Count: 1}),
	})
	if err != nil {
		log.Fatal(err)
	}
	imgs, _ := imgRes.Images()
	if err := os.WriteFile("sunset.png", imgs[0], 0644); err != nil {
		log.Fatal(err)
	}
}
```

## Examples

See the [`examples/`](examples/) directory for complete, runnable examples:

- **[Simple Text](examples/simple-text/main.go)**: Minimal text generation
- **[Text Generation](examples/text-generation/main.go)**: Text generation with provider selection
- **[Text to Image](examples/text-to-image/main.go)**: Image generation from text prompts
- **[Image Understanding](examples/image-understanding/main.go)**: Text generation from images
- **[PDF Understanding](examples/pdf-understanding/main.go)**: Text generation from PDF documents
- **[PDF to Image](examples/pdf-to-image/main.go)**: Image generation from PDF documents (e.g., infographics)
- **[OpenAI Image Options](examples/openai-image-options/main.go)**: Provider-specific image options (format, background, size, moderation, compression)
- **[Gemini Image Options](examples/gemini-image-options/main.go)**: Provider-specific image options (aspect ratio, size)

## Providers

### OpenAI

```go
import "github.com/montanaflynn/grail/providers/openai"

// Basic usage (uses OPENAI_API_KEY env var)
provider, err := openai.New()

// With options
provider, err := openai.New(
    openai.WithAPIKey("sk-..."),
    openai.WithTextModel("gpt-5.4"),
    openai.WithImageModel("gpt-image-2"),
    openai.WithLogger(logger),
)
```

**Options:**
- `WithAPIKey(key string)` - Set API key explicitly
- `WithAPIKeyFromEnv(env string)` - Read API key from environment variable
- `WithTextModel(model string)` - Override default text model (default: `gpt-5.4`)
- `WithImageModel(model string)` - Override default image model (default: `gpt-image-2`)
- `WithLogger(logger *slog.Logger)` - Set custom logger

**Image Options:**
- `WithImageFormat(format ImageFormat)` - Set output format (`png`, `jpeg`, `webp`)
- `WithImageBackground(bg ImageBackground)` - Set background (`auto`, `transparent`, `opaque`)
- `WithImageSize(size ImageSize)` - Set image size (`auto`, `1024x1024`, `1536x1024`, `1024x1536`, `256x256`, `512x512`, `1792x1024`, `1024x1792`)
- `WithImageModeration(moderation ImageModeration)` - Set moderation level (`auto`, `low`)
- `WithImageOutputCompression(compression int)` - Set output compression quality (0-100)

**Text Options:**
- `TextOptions{Model, MaxTokens, Temperature, TopP, SystemPrompt}` - Provider-specific text generation options

### Gemini

```go
import "github.com/montanaflynn/grail/providers/gemini"

// Basic usage (uses GEMINI_API_KEY env var)
provider, err := gemini.New(ctx)

// With options
provider, err := gemini.New(ctx,
    gemini.WithAPIKey("..."),
    gemini.WithTextModel("gemini-3.1-pro-preview"),
    gemini.WithImageModel("gemini-3.1-flash-image-preview"),
    gemini.WithLogger(logger),
)
```

**Options:**
- `WithAPIKey(key string)` - Set API key explicitly
- `WithAPIKeyFromEnv(env string)` - Read API key from environment variable
- `WithTextModel(model string)` - Override default text model (default: `gemini-3.1-pro-preview`)
- `WithImageModel(model string)` - Override default image model (default: `gemini-3-pro-image-preview`)
- `WithLogger(logger *slog.Logger)` - Set custom logger

**Image Options:**
- `WithImageAspectRatio(ratio ImageAspectRatio)` - Set aspect ratio (`1:1`, `16:9`, etc.)
- `WithImageSize(size ImageSize)` - Set image size (`1K`, `2K`, `4K`)

**Text Options:**
- `TextOptions{Model, MaxTokens, Temperature, TopP, SystemPrompt}` - Provider-specific text generation options

## Development

```bash
# Run tests
go test ./...

# Format code
go fmt ./...

# Run linter
go vet ./...

# Or use make
make format
make lint
make test
make # runs all
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines.

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for a detailed list of changes.

## License

MIT License - see [LICENSE](LICENSE) for details.
