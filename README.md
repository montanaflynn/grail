# grail

![CI](https://github.com/montanaflynn/grail/actions/workflows/ci.yml/badge.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A lightweight Go SDK that unifies multiple AI providers (OpenAI, Gemini) behind a consistent interface for text and image generation. Swap providers without changing your code.

## Features

- **Unified API**: Single `Client` interface for text and image generation across providers
- **Provider Agnostic**: Swap between OpenAI, Gemini, or custom providers seamlessly
- **Multimodal Support**: Ordered sequences of text and image inputs
- **Flexible Configuration**: Per-request model overrides, system prompts, and provider-specific options
- **Structured Logging**: Built-in `log/slog` integration with configurable levels and formats
- **Type-Safe Errors**: Typed error codes for predictable error handling

## Installation

```bash
go get github.com/montanaflynn/grail
```

> [!NOTE]
> Providers automatically use environment variables for API keys (`OPENAI_API_KEY`, `GEMINI_API_KEY`) if no key is explicitly provided. See the [Providers](#providers) section for details.

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

<details>
<summary><strong>Complete Text Generation Example</strong></summary>

```go
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/montanaflynn/grail"
	"github.com/montanaflynn/grail/providers/openai"
)

func main() {
	ctx := context.Background()

	// Uses default API key from environment variable OPENAI_API_KEY if not explicitly set
	provider, err := openai.New(
		// openai.WithAPIKey("sk-proj-*************"),
		// openai.WithAPIKeyFromEnv("OPENAI_API_KEY"),
		// openai.WithTextModel("gpt-5.1"),
		// openai.WithImageModel("gpt-image-1"),
	)
	if err != nil {
		log.Fatalf("new provider: %v", err)
	}

	client := grail.NewClient(
		provider,
		// Logger options (uncomment to customize):
		// grail.WithLogger(yourLogger),
		// grail.WithLoggerFormat("text", grail.LoggerLevels["info"]),
		// grail.WithLoggerFormat("json", grail.LoggerLevels["debug"]),
	)

	res, err := client.GenerateText(ctx, grail.TextRequest{
		Input: []grail.Part{
			grail.Text("Write a haiku about Go."),
		},
		// Options: grail.TextOptions{
		// 	Model:        "gpt-5.1",
		// 	MaxTokens:    grail.Pointer[int32](100),
		// 	Temperature:  grail.Pointer[float32](0.7),
		// 	TopP:         grail.Pointer[float32](0.9),
		// 	SystemPrompt: "You are a concise assistant.",
		// },
	})
	if err != nil {
		if grail.IsCode(err, grail.CodeInvalidInput) {
			log.Fatalf("fix request: %v", err)
		}
		log.Fatalf("generate text: %v", err)
	}

	fmt.Println(res.Text)
}
```

</details>

<details>
<summary><strong>Complete Image Generation Example</strong></summary>

```go
package main

import (
	"context"
	"log"
	"os"

	"github.com/montanaflynn/grail"
	"github.com/montanaflynn/grail/providers/openai"
)

func main() {
	ctx := context.Background()

	// Uses default API key from environment variable OPENAI_API_KEY if set
	provider, err := openai.New(
		// openai.WithAPIKey("sk-proj-*************"),
		// openai.WithAPIKeyFromEnv("OPENAI_API_KEY"),
		// openai.WithTextModel("gpt-5.1"),
		// openai.WithImageModel("gpt-image-1"),
	)
	if err != nil {
		log.Fatalf("new provider: %v", err)
	}

	client := grail.NewClient(
		provider,
		// Logger options (uncomment to customize):
		// grail.WithLogger(yourLogger),
		// grail.WithLoggerFormat("text", grail.LoggerLevels["info"]),
		// grail.WithLoggerFormat("json", grail.LoggerLevels["debug"]),
	)

	res, err := client.GenerateImage(ctx, grail.ImageRequest{
		Input: []grail.Part{
			grail.Text("A serene mountain landscape at sunset"),
		},
		// Options: grail.ImageOptions{
		// 	Model:        "gpt-image-1",
		// 	SystemPrompt: "Generate high-quality, photorealistic images.",
		// },
		ProviderOptions: []grail.ProviderOption{
			openai.WithImageFormat(openai.ImageFormatPNG),
			// openai.WithImageBackground(openai.ImageBackgroundTransparent),
		},
	})
	if err != nil {
		if grail.IsCode(err, grail.CodeInvalidInput) {
			log.Fatalf("fix request: %v", err)
		}
		log.Fatalf("generate image: %v", err)
	}

	// Save the first generated image
	os.WriteFile("sunset.png", res.Images[0].Data, 0644)
}
```

</details>

<details>
<summary><strong>Complete Image Understanding Example</strong></summary>

```go
package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/montanaflynn/grail"
	"github.com/montanaflynn/grail/providers/openai"
)

func main() {
	ctx := context.Background()

	provider, err := openai.New(
		// openai.WithAPIKey("sk-proj-*************"),
		// openai.WithAPIKeyFromEnv("OPENAI_API_KEY"),
		// openai.WithTextModel("gpt-5.1"),
	)
	if err != nil {
		log.Fatalf("new provider: %v", err)
	}

	client := grail.NewClient(
		provider,
		// grail.WithLoggerFormat("text", grail.LoggerLevels["debug"]),
	)

	// Read image from file
	imgData, err := os.ReadFile("photo.jpg")
	if err != nil {
		log.Fatalf("read image: %v", err)
	}

	res, err := client.GenerateText(ctx, grail.TextRequest{
		Input: []grail.Part{
			grail.Text("Describe the style and content of this image."),
			grail.Image(imgData, "image/jpeg"),
			grail.Text("Keep it concise."),
		},
		// Options: grail.TextOptions{
		// 	Model:        "gpt-5.1",
		// 	MaxTokens:    grail.Pointer[int32](200),
		// 	SystemPrompt: "You are an expert art critic.",
		// },
	})
	if err != nil {
		if grail.IsCode(err, grail.CodeInvalidInput) {
			log.Fatalf("fix request: %v", err)
		}
		log.Fatalf("generate text: %v", err)
	}

	fmt.Println(res.Text)
}
```

</details>

<details>
<summary><strong>Complete Multimodal Image Generation Example</strong></summary>

```go
package main

import (
	"context"
	"log"
	"os"

	"github.com/montanaflynn/grail"
	"github.com/montanaflynn/grail/providers/openai"
)

func main() {
	ctx := context.Background()

	provider, err := openai.New(
		// openai.WithAPIKey("sk-proj-*************"),
		// openai.WithAPIKeyFromEnv("OPENAI_API_KEY"),
		// openai.WithImageModel("gpt-image-1"),
	)
	if err != nil {
		log.Fatalf("new provider: %v", err)
	}

	client := grail.NewClient(
		provider,
		// grail.WithLoggerFormat("text", grail.LoggerLevels["debug"]),
	)

	// Read reference image
	imgData, err := os.ReadFile("reference.jpg")
	if err != nil {
		log.Fatalf("read image: %v", err)
	}

	res, err := client.GenerateImage(ctx, grail.ImageRequest{
		Input: []grail.Part{
			grail.Text("Create a variation of this image"),
			grail.Image(imgData, "image/jpeg"),
			grail.Text("but make it more vibrant and add a sunset sky"),
		},
		// Options: grail.ImageOptions{
		// 	Model:        "gpt-image-1",
		// 	SystemPrompt: "Generate high-quality, artistic images.",
		// },
		ProviderOptions: []grail.ProviderOption{
			openai.WithImageFormat(openai.ImageFormatPNG),
			// openai.WithImageBackground(openai.ImageBackgroundOpaque),
		},
	})
	if err != nil {
		if grail.IsCode(err, grail.CodeInvalidInput) {
			log.Fatalf("fix request: %v", err)
		}
		log.Fatalf("generate image: %v", err)
	}

	// Save the generated image
	os.WriteFile("variation.png", res.Images[0].Data, 0644)
}
```

</details>

## Core API

<details>
<summary><strong>Table of Contents</strong></summary>

- [Client](#client)
- [Provider Interface](#provider-interface)
- [Requests](#requests)
- [Options](#options)
- [Parts (Multimodal Input)](#parts-multimodal-input)
- [Results](#results)
- [Client Options](#client-options)
- [Helper Functions](#helper-functions)

</details>

### Client

The `Client` is the main entry point for all operations. It wraps a `Provider` and adds logging and validation.

```go
type Client struct {
	// ... unexported fields
}

// NewClient creates a new client with the given provider and options
func NewClient(p Provider, opts ...ClientOption) *Client

// GenerateText generates text from multimodal input
func (c *Client) GenerateText(ctx context.Context, req TextRequest) (TextResult, error)

// GenerateImage generates images from multimodal input
func (c *Client) GenerateImage(ctx context.Context, req ImageRequest) (ImageResult, error)
```

### Provider Interface

Providers implement the `Provider` interface to support different AI backends:

```go
type Provider interface {
	GenerateText(ctx context.Context, req TextRequest) (TextResult, error)
	GenerateImage(ctx context.Context, req ImageRequest) (ImageResult, error)
	DefaultTextModel() string
	DefaultImageModel() string
}
```

### Requests

#### TextRequest

```go
type TextRequest struct {
	Input   []Part      // Ordered sequence of text and/or image parts
	Options TextOptions // Generation parameters
}
```

#### ImageRequest

```go
type ImageRequest struct {
	Input           []Part           // Ordered sequence of text and/or image parts
	Options         ImageOptions     // Generation parameters
	ProviderOptions []ProviderOption // Provider-specific options (e.g., image format)
}
```

### Options

#### TextOptions

```go
type TextOptions struct {
	Model        string   // Model override (empty = provider default)
	MaxTokens    *int32   // Maximum output tokens
	Temperature  *float32 // Randomness (0.0-2.0, use with TopP, not both)
	TopP         *float32 // Nucleus sampling (0.0-1.0, use with Temperature, not both)
	SystemPrompt string   // System instruction applied before user content
}
```

> [!WARNING]
> `Temperature` and `TopP` are mutually exclusive. Setting both will result in a `CodeBadOptions` error.

#### ImageOptions

```go
type ImageOptions struct {
	Model        string // Model override (empty = provider default)
	SystemPrompt string // System instruction for image generation
}
```

### Parts (Multimodal Input)

Parts represent ordered multimodal input. The order matters and is preserved.

```go
// Part is the interface for multimodal input
type Part interface {
	Kind() PartKind
}

// Text creates a text part
func Text(s string) Part

// Image creates an image part from raw bytes
func Image(data []byte, mime string) Part
```

> [!TIP]
> The order of parts in your input array is preserved and sent to the provider in that exact sequence. This is important for multimodal conversations where context matters.

Example with ordered input:

```go
input := []grail.Part{
	grail.Text("What's in this image?"),
	grail.Image(imgData, "image/png"),
	grail.Text("Now describe the second image:"),
	grail.Image(imgData2, "image/jpeg"),
}
```

### Results

#### TextResult

```go
type TextResult struct {
	Text string // Generated text content
	Raw  any    // Raw provider response for advanced use cases
}
```

#### ImageResult

```go
type ImageResult struct {
	Images []ImageOutput // Generated images
	Raw    any           // Raw provider response for advanced use cases
}

type ImageOutput struct {
	Data []byte // Image bytes
	MIME string // MIME type (e.g., "image/png")
}
```

### Client Options

```go
// WithLogger sets a custom logger
func WithLogger(l *slog.Logger) ClientOption

// WithLoggerFormat creates a logger with the specified format and level
// Format: "text" or "json"
// Level: grail.LoggerLevelDebug, LoggerLevelInfo, LoggerLevelWarn, LoggerLevelError
func WithLoggerFormat(format string, level LoggerLevel) ClientOption
```

Example:

```go
client := grail.NewClient(
	provider,
	grail.WithLoggerFormat("json", grail.LoggerLevelDebug),
)
```

### Helper Functions

```go
// Pointer is a helper to take the address of a literal value
// Useful for setting pointer fields in options
func Pointer[T any](v T) *T
```

Example:

```go
options := grail.TextOptions{
	Temperature: grail.Pointer(0.7),
	MaxTokens:   grail.Pointer[int32](100),
}
```

## Providers

<details>
<summary><strong>Available Providers</strong></summary>

- **OpenAI**: Uses the Responses API for both text and image generation
- **Gemini**: Supports Gemini 2.5 Flash models for text and image
- **Mock**: For testing without API calls

</details>

### OpenAI

```go
import "github.com/montanaflynn/grail/providers/openai"

// Default models: gpt-5.1 (text), gpt-image-1 (image)
provider, err := openai.New(
	openai.WithAPIKey("sk-..."),                    // Explicit key
	openai.WithAPIKeyFromEnv("OPENAI_API_KEY"),      // From env var
	openai.WithTextModel("gpt-5.1"),                 // Override text model
	openai.WithImageModel("gpt-image-1"),            // Override image model
	openai.WithLogger(logger),                       // Custom logger
)
```

> [!NOTE]
> **API Key**: Automatically uses `OPENAI_API_KEY` environment variable if no key is explicitly provided. If you use `WithAPIKey` or `WithAPIKeyFromEnv` with an empty value, the provider will return `openai.ErrAPIKeyRequired`.

**Image Options**:

```go
// Image format options
openai.WithImageFormat(openai.ImageFormatPNG)   // PNG (default)
openai.WithImageFormat(openai.ImageFormatJPEG)  // JPEG
openai.WithImageFormat(openai.ImageFormatWEBP)  // WebP

// Background options
openai.WithImageBackground(openai.ImageBackgroundAuto)        // Auto (default)
openai.WithImageBackground(openai.ImageBackgroundTransparent) // Transparent
openai.WithImageBackground(openai.ImageBackgroundOpaque)      // Opaque
```

Example with image options:

```go
res, err := client.GenerateImage(ctx, grail.ImageRequest{
	Input: []grail.Part{
		grail.Text("A logo with transparent background"),
	},
	ProviderOptions: []grail.ProviderOption{
		openai.WithImageFormat(openai.ImageFormatPNG),
		openai.WithImageBackground(openai.ImageBackgroundTransparent),
	},
})
```

### Gemini

```go
import "github.com/montanaflynn/grail/providers/gemini"

// Default models: gemini-2.5-flash (text), gemini-2.5-flash-image (image)
provider, err := gemini.New(
	context.Background(),
	gemini.WithAPIKey("..."),                    // Explicit key
	gemini.WithAPIKeyFromEnv("GEMINI_API_KEY"), // From env var
	gemini.WithTextModel("gemini-2.5-flash"),   // Override text model
	gemini.WithImageModel("gemini-2.5-flash-image"), // Override image model
	gemini.WithLogger(logger),                  // Custom logger
)
```

> [!NOTE]
> **API Key**: Automatically uses `GEMINI_API_KEY` environment variable if no key is explicitly provided. If you use `WithAPIKey` or `WithAPIKeyFromEnv` with an empty value, the provider will return `gemini.ErrAPIKeyRequired`.

### Mock Provider (Testing)

```go
import "github.com/montanaflynn/grail/providers/mock"

provider := mock.NewProvider()
provider.TextModelVal = "mock-text-model"
provider.ImageModelVal = "mock-image-model"

// Configure responses
provider.TextResponse = "Mock response"
provider.ImageResponse = []grail.ImageOutput{
	{Data: []byte("fake image"), MIME: "image/png"},
}
```

## Error Handling

> [!IMPORTANT]
> Grail uses typed errors with error codes for predictable error handling. Always check error codes rather than error messages for programmatic handling.

Grail uses typed errors for predictable error handling:

```go
type Error struct {
	Code     ErrorCode        // Error category
	Message  string           // Human-readable message
	Cause    error            // Underlying error
	Metadata map[string]any   // Additional context
}

type ErrorCode string

const (
	CodeInvalidInput      ErrorCode = "invalid_input"
	CodeBadOptions        ErrorCode = "bad_options"
	CodeMissingCredentials ErrorCode = "missing_credentials"
	CodeUnsupported       ErrorCode = "unsupported"
	CodeInternal          ErrorCode = "internal"
	CodeUnknown           ErrorCode = "unknown"
)
```

### Error Checking

```go
res, err := client.GenerateText(ctx, req)
if err != nil {
	// Check error code
	if grail.IsCode(err, grail.CodeInvalidInput) {
		// Fix request and retry
	}
	
	// Get error code directly
	code := grail.GetErrorCode(err)
	
	// Extract typed error
	if ge, ok := grail.AsError(err); ok {
		fmt.Printf("Code: %s, Message: %s\n", ge.Code, ge.Message)
	}
	
	return err
}
```

### Provider-Specific Errors

<details>
<summary><strong>Provider Error Handling</strong></summary>

Providers may return their own errors. For example:

```go
import "github.com/montanaflynn/grail/providers/openai"

provider, err := openai.New()
if err == openai.ErrAPIKeyRequired {
	// Handle missing API key
}
```

Common provider errors:
- `openai.ErrAPIKeyRequired`: OpenAI API key is missing
- `gemini.ErrAPIKeyRequired`: Gemini API key is missing

</details>

## Logging

The client logs all requests at the `Info` level by default. Enable debug logging to see full request/response details:

```go
client := grail.NewClient(
	provider,
	grail.WithLoggerFormat("text", grail.LoggerLevelDebug),
)
```

<details>
<summary><strong>Log Output Details</strong></summary>

Log output includes:
- Model name
- Input parts summary (type, length, MIME for images)
- Options (all fields, including zero values)
- Response metadata (text length, image count, raw response)

At debug level, you'll also see the full raw provider response for troubleshooting.

</details>

## Examples

The repository includes several examples:

<details>
<summary><strong>View all examples</strong></summary>

- **[Simple Text](examples/simple-text/main.go)**: Minimal text generation example
- **[Text Generation](examples/text-generation/main.go)**: Text generation with provider selection
- **[Text to Image](examples/text-to-image/main.go)**: Image generation from text prompts
- **[Image Understanding](examples/image-understanding/main.go)**: Text generation from images
- **[OpenAI Image Options](examples/openai-image-options/main.go)**: Provider-specific image options

</details>

Run examples:

```bash
# Text generation with OpenAI
cd examples/text-generation
go run main.go --openai

# Image generation with debug logging
cd examples/text-to-image
go run main.go --openai --debug
```

## Development

### Requirements

- Go 1.22 or later

### Building

```bash
# Run tests
go test ./...

# Format code
go fmt ./...

# Run linter
go vet ./...
```

### Make Targets

```bash
make          # Run fmt, lint, and test
make fmt      # Format code
make lint     # Run go vet
make test     # Run tests
```

### Git Hooks (Optional)

<details>
<summary><strong>Enable Git Hooks</strong></summary>

Enable pre-commit hooks for code quality:

```bash
git config core.hooksPath .githooks
```

Hooks:
- `commit-msg`: Enforces conventional commits
- `pre-commit`: Runs `gofmt` on staged Go files

</details>

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines and workflow.

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for a detailed list of changes. The changelog is auto-generated from conventional commits.

## License

MIT License - see [LICENSE](LICENSE) for details.
