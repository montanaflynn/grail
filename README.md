# grail

![CI](https://img.shields.io/github/actions/workflow/status/montanaflynn/grail/ci.yml)
[![Go Reference](https://img.shields.io/badge/go.dev-reference-00ADD8)](https://pkg.go.dev/github.com/montanaflynn/grail)

**Your Go code keeps working when AI models churn.**

grail is a Go SDK for text, image and JSON generation across OpenAI, Gemini and Anthropic. You ask for what you need (text or an image, the best or fast tier, JSON shaped like a struct), and grail picks a current model and provider, warns before a model you pinned shuts down, and can switch to its replacement for you.

```go
// Before: pinned to a model. Broke on 2026-10-23, when OpenAI shut gpt-image-1 down.
png, err := client.Image(ctx, prompt, grail.WithModel("gpt-image-1"))

// After: ask for a tier. Best now resolves to gpt-image-2.5-sunburst, and this line didn't change.
png, err := client.Image(ctx, prompt, grail.WithTier(grail.Best))
```

## Quick start

```bash
go get github.com/montanaflynn/grail
```

```go
func main() {
	ctx := context.Background()

	// Uses whichever of ANTHROPIC_API_KEY, OPENAI_API_KEY and GEMINI_API_KEY are set.
	client, err := auto.FromEnv(ctx)
	if err != nil {
		log.Fatal(err)
	}

	text, err := client.Text(ctx, "Write a haiku about Go.")
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(text)

	png, err := client.Image(ctx, "A gopher surfing at sunset", grail.WithTier(grail.Fast))
	if err != nil {
		log.Fatal(err)
	}
	if err := os.WriteFile("gopher.png", png, 0o644); err != nil {
		log.Fatal(err)
	}

	type City struct {
		Name    string `json:"name"`
		Country string `json:"country"`
	}
	city, err := grail.JSON[City](ctx, client, "The largest city in Japan")
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(city.Name, city.Country)
}
```

Imports: `github.com/montanaflynn/grail` and `github.com/montanaflynn/grail/auto`. `auto.FromEnv` adds a provider for each key that is set, in the order Anthropic, OpenAI, Gemini. Set `GRAIL_PROVIDERS=gemini,openai` to choose the providers and their order.

Call options: `grail.WithTier(grail.Best | grail.Fast)`, `grail.WithModel("...")` to pin a model, `grail.WithInputs(...)` to add images or PDFs after the prompt, and `grail.WithProviderOptions(...)` for provider-specific settings.

## Typed JSON

`grail.JSON[T]` derives a JSON Schema from `T` with `grail.SchemaFor[T]()` and sends it to each provider's native structured-output API, so the response matches the struct. Tags add descriptions and enums:

```go
type Ticket struct {
	Title    string `json:"title"`
	Priority string `json:"priority" jsonschema:"enum=low|medium|high"`
	Assignee string `json:"assignee,omitempty" jsonschema:"description=GitHub login"`
}
```

Every field is listed in `required`, as OpenAI strict mode demands. Optional fields (`omitempty`, `omitzero` and pointers) are nullable instead: `"type": ["string", "null"]`, and a `null` decodes to the zero value. Recursive types, interfaces and maps without string keys are rejected.

## Routing and fallback

One client can hold several providers. Each request goes to the first provider that can serve it. If that provider fails with a retryable error (rate limited, unavailable), the client falls back to the next capable one.

Claude for text, Gemini for images:

```go
claude, err := anthropic.New() // ANTHROPIC_API_KEY
gem, err := gemini.New(ctx)    // GEMINI_API_KEY
client := grail.NewClient(claude, grail.WithProviders(gem))

text, err := client.Text(ctx, "Write a tagline for a lighthouse.") // Claude
img, err := client.Image(ctx, "A lighthouse at dawn")              // Gemini: Claude can't generate images
```

`auto.FromEnv` with `GRAIL_PROVIDERS=anthropic,gemini` builds the same client.

Routing checks the capabilities the request needs: image or PDF inputs skip providers whose model can't read them. A response served after a fallback carries `Warning{Code: "fallback"}` naming the provider that failed:

```
fallback: anthropic failed (unavailable: overloaded); fell back to gemini
```

A pinned model (`WithModel`) goes to the provider that lists it and never falls back. When no provider can serve a request, the error code is `grail.Unsupported` and the message names the missing capability.

## Streaming

`client.Stream` yields text as it is generated, for OpenAI, Gemini and Anthropic. The last event has `Done` set and carries usage. Breaking out of the loop cancels the request.

```go
for ev, err := range client.Stream(ctx, grail.Request{
	Inputs: []grail.Input{grail.InputText("Tell a short story.")},
	Output: grail.OutputText(),
	Tier:   grail.Fast,
}) {
	if err != nil {
		log.Fatal(err)
	}
	fmt.Print(ev.Text)
}
```

Streaming supports text output only. It picks the first capable provider and doesn't fall back once the stream starts.

## Deprecation warnings and auto-migrate

grail's model catalog records each provider's deprecation notices. A request for a deprecated model still goes through, with a warning on the response:

```go
res, err := client.Generate(ctx, grail.Request{
	Inputs: []grail.Input{grail.InputText("Summarize the release notes.")},
	Output: grail.OutputText(),
	Model:  "gpt-4o",
})
for _, w := range res.Warnings {
	fmt.Printf("%s: %s\n", w.Code, w.Message)
}
// model_deprecated: gpt-4o is deprecated; shuts down 2027-01-14; use gpt-6-luna
```

The client also logs each warning once per model at `Warn` level, so `Text`, `Image` and `JSON` callers see it too:

```
WARN gpt-4o is deprecated; shuts down 2027-01-14; use gpt-6-luna code=model_deprecated provider=openai model=gpt-4o
```

With `grail.WithAutoMigrate()`, a model whose shutdown date has passed is replaced before the request is sent, so the pinned call from the top of this page works again:

```go
client := grail.NewClient(provider, grail.WithAutoMigrate()) // or auto.FromEnv(ctx, grail.WithAutoMigrate())
png, err := client.Image(ctx, prompt, grail.WithModel("gpt-image-1"))
// model_migrated: gpt-image-1 shut down 2026-10-23; migrated to gpt-image-2.5-sunburst
```

Migration takes one hop: if the replacement is itself deprecated, you get a `model_deprecated` warning for it as well. Without the option, the request goes out unchanged and the provider's error surfaces. The warning codes are the constants `grail.WarningModelDeprecated`, `grail.WarningModelMigrated` and `grail.WarningFallback`.

## Keeping the catalog current

Model churn is caught upstream too. [`.github/workflows/model-drift.yml`](.github/workflows/model-drift.yml) runs every night and compares each provider's live model list against grail's catalog. When they diverge (a model grail knows has shut down, or a new text or image model has shipped), it opens or updates a single `model-drift` issue, and closes it once the catalog catches up. Models grail deliberately leaves out are listed in [`skills/discover-llm-models/acknowledged.txt`](skills/discover-llm-models/acknowledged.txt). Fixes ship as patch releases that update the defaults behind `Best` and `Fast`. See the [discover-llm-models skill](skills/discover-llm-models/SKILL.md) for how drift is acted on.

## Providers

| Provider | Package | API key | Text best / fast | Image best / fast |
| --- | --- | --- | --- | --- |
| OpenAI | `providers/openai` | `OPENAI_API_KEY` | `gpt-6-astra` / `gpt-6-luna` | `gpt-image-2.5-sunburst` / `gpt-image-2.5-flare` |
| Gemini | `providers/gemini` | `GEMINI_API_KEY` | `gemini-4-argon` / `gemini-3.8-flash` | `gemini-3-pro-image` / `gemini-nano-banana-2.1` |
| Anthropic | `providers/anthropic` | `ANTHROPIC_API_KEY` | `claude-fable-5-1` / `claude-haiku-5-5` | none |
| ModelsLab | `providers/modelslab` | `MODELSLAB_API_KEY` | none | `flux` / `sdxl` |

Every provider accepts `WithAPIKey`, `WithAPIKeyFromEnv` and `WithLogger`. Each package also exports its models as constants (for example `openai.GPT6Luna`) with their capabilities and deprecation records. ModelsLab isn't picked up by `auto.FromEnv`; construct it explicitly.

### OpenAI

```go
provider, err := openai.New(
	openai.WithTextModel("gpt-6-astra"),             // default
	openai.WithImageModel("gpt-image-2.5-sunburst"), // default
)
```

Image options (pass with `grail.WithProviderOptions`): `WithImageFormat` (`png`, `jpeg`, `webp`), `WithImageBackground` (`auto`, `transparent`, `opaque`), `WithImageSize` (`auto`, `1024x1024`, `1536x1024`, `1024x1536`, ...), `WithImageModeration` (`auto`, `low`) and `WithImageOutputCompression` (0-100). Text options: `openai.TextOptions{Model, MaxTokens, Temperature, TopP, SystemPrompt}`.

### Gemini

```go
provider, err := gemini.New(ctx,
	gemini.WithTextModel("gemini-4-argon"),      // default
	gemini.WithImageModel("gemini-3-pro-image"), // default
)
```

Image options: `WithImageAspectRatio` (`1:1`, `16:9`, ...) and `WithImageSize` (`1K`, `2K`, `4K`). Text options: `gemini.TextOptions{Model, MaxTokens, Temperature, TopP, SystemPrompt}`.

### Anthropic

Claude handles text and JSON output with text, image and PDF inputs. It has no image generation, so image requests return `grail.Unsupported` (or route to another provider).

```go
provider, err := anthropic.New(
	anthropic.WithTextModel("claude-fable-5-1"), // default
)
```

Text options: `anthropic.TextOptions{Model, MaxTokens, SystemPrompt}`. `MaxTokens` defaults to 16000. There are no `Temperature`, `TopP` or `TopK` options. JSON output uses the Messages API's structured outputs, which require `"additionalProperties": false` on every object; `grail.SchemaFor` already sets it.

## Advanced: the Request API

`Text`, `Image`, `JSON` and `Stream` are thin wrappers over `Generate`, which takes an ordered list of inputs and an output spec and returns the full response: outputs, usage, the provider and models that served it, and warnings.

```go
res, err := client.Generate(ctx, grail.Request{
	Inputs: []grail.Input{
		grail.InputText("Describe this image in one sentence."),
		grail.InputImage(photo),
	},
	Output: grail.OutputText(),
	Tier:   grail.Best,
})
text, _ := res.Text()
fmt.Println(res.Provider.Name+":", text)
```

- Inputs: `InputText`, `InputImage`, `InputPDF`, `InputFile`, `InputFileReader`, and `client.InputImageFromURI` / `InputPDFFromURI` to download one.
- Outputs: `OutputText()`, `OutputImage(grail.ImageSpec{Count: n})`, `OutputJSON(schema)`. Read them with `res.Text()`, `res.Images()` and `res.DecodeJSON(&v)`.
- Model selection: `Request.Model` beats `Request.Tier`, which beats the provider's default. `client.ListModels` and `client.GetModel(ctx, role, tier)` show the catalog across providers.
- Errors carry a code: `grail.GetErrorCode(err)` returns `InvalidArgument`, `Unauthorized`, `RateLimited`, `Timeout`, `Unavailable`, `Unsupported`, `Refused`, `OutputInvalid` or `Internal`, and `grail.IsRetryable(err)` reports whether a retry may help.
- Client options: `WithProviders`, `WithAutoMigrate`, `WithLogger`, `WithLoggerFormat`, `WithHTTPClient`, `WithDownloadLimits`.

## Examples

Every snippet in this README is mirrored by an `Example` function in [`example_test.go`](example_test.go) and [`auto/example_test.go`](auto/example_test.go), so `go test ./...` compiles them. Runnable programs live in [`examples/`](examples/):

- **[Auto](examples/auto/main.go)**: Zero-config client from whichever API keys are set
- **[Simple Text](examples/simple-text/main.go)**: Minimal text generation with `client.Text`
- **[JSON Output](examples/json-output/main.go)**: Typed structured output with `grail.JSON[T]`
- **[Streaming](examples/streaming/main.go)**: Stream text as it is generated
- **[Anthropic Text](examples/anthropic-text/main.go)**: Text and JSON generation with Claude
- **[Model Selection](examples/model-selection/main.go)**: Best and fast tiers, model constants and capabilities
- **[Text Generation](examples/text-generation/main.go)**: Text generation with provider selection
- **[Text to Image](examples/text-to-image/main.go)**: Image generation from text prompts
- **[Image Understanding](examples/image-understanding/main.go)**: Text generation from images
- **[PDF Understanding](examples/pdf-understanding/main.go)**: Text generation from PDF documents
- **[PDF to Image](examples/pdf-to-image/main.go)**: Image generation from PDF documents (e.g., infographics)
- **[OpenAI Image Options](examples/openai-image-options/main.go)**: Format, background, size, moderation, compression
- **[Gemini Image Options](examples/gemini-image-options/main.go)**: Aspect ratio and size

## Development

```bash
make fmt-check lint test   # gofmt, go vet ./..., go test ./...
make                       # go fmt, vet and test
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines.

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for a detailed list of changes.

## License

MIT License - see [LICENSE](LICENSE) for details.
