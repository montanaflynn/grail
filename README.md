grail
=====

![CI](https://github.com/montanaflynn/grail/actions/workflows/ci.yml/badge.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Lightweight Go client that wraps multiple AI providers (currently OpenAI and Gemini) behind a consistent interface for text and image generation. Swap providers without changing your calling code.

Features
- Unified `Client` API for text and image generation
- Provider defaults with optional per-request model overrides
- Built-in request logging via `log/slog`
- Helpers for multimodal inputs (`grail.Text`, `grail.Image`)
- Ready-made providers: OpenAI, Gemini; plus a mock provider for tests

Install
```
go get github.com/montanaflynn/grail
```

Quick start (OpenAI)
```
package main

import (
	"context"
	"fmt"

	"github.com/montanaflynn/grail"
	"github.com/montanaflynn/grail/providers/openai"
)

func main() {
	prov, err := openai.New(openai.WithAPIKeyFromEnv("OPENAI_API_KEY"))
	if err != nil {
		panic(err)
	}

	client := grail.NewClient(prov)

	res, err := client.GenerateText(context.Background(), grail.TextRequest{
		Input: []grail.Part{
			grail.Text("Write a short haiku about Go."),
		},
		Options: grail.TextOptions{
			SystemPrompt: "You are a concise assistant.",
		},
	})
	if err != nil {
		panic(err)
	}

	fmt.Println(res.Text)
}
```

Examples
- Text generation: `examples/text-generation/main.go`
- Image generation: `examples/text-to-image/main.go`
- Image understanding: `examples/image-understanding/main.go`

Providers
- OpenAI: see `providers/openai`
- Gemini: see `providers/gemini`
- Mock (for tests): see `providers/mock`

Errors
- Grail exposes typed errors with coarse codes (e.g., `invalid_input`, `bad_options`, `missing_credentials`).
- Check codes using `grail.GetErrorCode(err)` or `grail.IsCode(err, grail.CodeInvalidInput)`.
- Example:
```
if err != nil {
    if grail.IsCode(err, grail.CodeInvalidInput) {
        // fix request and retry
    }
    return err
}
```

Logging
- `grail.NewClient` accepts `grail.WithLogger(*slog.Logger)` to enable structured logs.

Development
- Requires Go 1.22+
- Run tests: `go test ./...`
- Optional hooks (opt-in): `git config core.hooksPath .githooks`
  - `commit-msg` enforces conventional commits
  - `pre-commit` runs gofmt on staged Go files
- Make targets: `make fmt` (go fmt), `make lint` (go vet), `make test` (go test)
- `make` runs fmt, lint, and test in order

Contributing
- See `CONTRIBUTING.md` for workflow notes.


Changelog
- See [CHANGELOG.md](CHANGELOG.md) for a list of changes. The changelog is auto-generated from conventional commits.

License
- MIT; see `LICENSE`.

