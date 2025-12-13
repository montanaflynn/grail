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
// Create a provider (automatically uses OPENAI_API_KEY if not provided)
provider, _ := openai.New()

// Create a client
client := grail.NewClient(provider)

// Generate text
res, _ := client.GenerateText(ctx, grail.TextRequest{
	Input: []grail.Part{grail.Text("Create a haiku")},
})
fmt.Println(res.Text)

// Generate image
imgRes, _ := client.GenerateImage(ctx, grail.ImageRequest{
	Input: []grail.Part{grail.Text("A beautiful sunset")},
})
os.WriteFile("sunset.png", imgRes.Images[0].Data, 0644)

// Generate image with provider-specific options
import "github.com/montanaflynn/grail/providers/gemini"
imgRes2, _ := client.GenerateImage(ctx, grail.ImageRequest{
	Input: []grail.Part{grail.Text("A landscape photo")},
	ProviderOptions: []grail.ProviderOption{
		gemini.WithImageAspectRatio(gemini.ImageAspectRatio16_9),
		gemini.WithImageSize(gemini.ImageSize2K),
	},
})

// Image understanding (text from image)
imgData, _ := os.ReadFile("photo.jpg")
textRes, _ := client.GenerateText(ctx, grail.TextRequest{
	Input: []grail.Part{
		grail.Text("Describe this image"),
		grail.Image(imgData, "image/jpeg"),
	},
})
fmt.Println(textRes.Text)

// Multimodal image generation (image from text + image)
imgRes2, _ := client.GenerateImage(ctx, grail.ImageRequest{
	Input: []grail.Part{
		grail.Text("Create a variation of this image"),
		grail.Image(imgData, "image/jpeg"),
		grail.Text("but make it more colorful"),
	},
})
os.WriteFile("variation.png", imgRes2.Images[0].Data, 0644)

// PDF understanding (text from PDF)
pdfData, _ := os.ReadFile("document.pdf")
pdfRes, _ := client.GenerateText(ctx, grail.TextRequest{
	Input: []grail.Part{
		grail.Text("Summarize this document"),
		grail.PDF(pdfData, "application/pdf"),
	},
})
fmt.Println(pdfRes.Text)
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
    openai.WithTextModel("gpt-4"),
    openai.WithImageModel("dall-e-3"),
    openai.WithLogger(logger),
)
```

**Options:**
- `WithAPIKey(key string)` - Set API key explicitly
- `WithAPIKeyFromEnv(env string)` - Read API key from environment variable
- `WithTextModel(model string)` - Override default text model (default: `gpt-5.1`)
- `WithImageModel(model string)` - Override default image model (default: `gpt-image-1`)
- `WithLogger(logger *slog.Logger)` - Set custom logger

**Image Options:**
- `WithImageFormat(format ImageFormat)` - Set output format (`png`, `jpeg`, `webp`)
- `WithImageBackground(bg ImageBackground)` - Set background (`auto`, `transparent`, `opaque`)
- `WithImageSize(size ImageSize)` - Set image size (`auto`, `1024x1024`, `1536x1024`, `1024x1536`, `256x256`, `512x512`, `1792x1024`, `1024x1792`)
- `WithImageModeration(moderation ImageModeration)` - Set moderation level (`auto`, `low`)
- `WithImageOutputCompression(compression int64)` - Set output compression quality (0-100)

### Gemini

```go
import "github.com/montanaflynn/grail/providers/gemini"

// Basic usage (uses GEMINI_API_KEY env var)
provider, err := gemini.New(ctx)

// With options
provider, err := gemini.New(ctx,
    gemini.WithAPIKey("..."),
    gemini.WithTextModel("gemini-2.5-flash"),
    gemini.WithImageModel("gemini-2.5-flash-image"),
    gemini.WithLogger(logger),
)
```

**Options:**
- `WithAPIKey(key string)` - Set API key explicitly
- `WithAPIKeyFromEnv(env string)` - Read API key from environment variable
- `WithTextModel(model string)` - Override default text model (default: `gemini-2.5-flash`)
- `WithImageModel(model string)` - Override default image model (default: `gemini-2.5-flash-image`)
- `WithLogger(logger *slog.Logger)` - Set custom logger

## Links

- [API Reference](https://pkg.go.dev/github.com/montanaflynn/grail)
- [Providers](https://pkg.go.dev/github.com/montanaflynn/grail/providers)
- [Examples](https://github.com/montanaflynn/grail/tree/main/examples)
- [Changelog](https://github.com/montanaflynn/grail/blob/main/CHANGELOG.md)

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
