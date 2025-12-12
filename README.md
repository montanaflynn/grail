# grail

![CI](https://img.shields.io/github/actions/workflow/status/montanaflynn/grail/ci.yml)
![Version](https://img.shields.io/github/v/tag/montanaflynn/grail?label=version)
![License](https://img.shields.io/github/license/montanaflynn/grail)
[![Go Reference](https://img.shields.io/badge/go.dev-reference-00ADD8)](https://pkg.go.dev/github.com/montanaflynn/grail)

A lightweight Go SDK that unifies multiple AI providers (OpenAI, Gemini) behind a consistent interface for text and image generation. Swap providers without changing your code.

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

> [!NOTE]
> Providers automatically use environment variables for API keys (`OPENAI_API_KEY`, `GEMINI_API_KEY`) if no key is explicitly provided.

## Quick Start

```go
// Create a provider
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
```

## Examples

See the [`examples/`](examples/) directory for complete, runnable examples:

- **[Simple Text](examples/simple-text/main.go)**: Minimal text generation
- **[Text Generation](examples/text-generation/main.go)**: Text generation with provider selection
- **[Text to Image](examples/text-to-image/main.go)**: Image generation from text prompts
- **[Image Understanding](examples/image-understanding/main.go)**: Text generation from images
- **[OpenAI Image Options](examples/openai-image-options/main.go)**: Provider-specific image options

## Documentation

- **API Reference**: [pkg.go.dev/github.com/montanaflynn/grail](https://pkg.go.dev/github.com/montanaflynn/grail)
- **Providers**: See `providers/openai` and `providers/gemini` for provider-specific documentation

## Development

```bash
# Run tests
go test ./...

# Format code
go fmt ./...

# Run linter
go vet ./...

# Or use make
make          # Run fmt, lint, and test
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines.

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for a detailed list of changes.

## License

MIT License - see [LICENSE](LICENSE) for details.
