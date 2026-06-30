---
name: discover-llm-models
description: >-
  Find current LLM model IDs from OpenAI and Gemini SDKs and official docs,
  then audit or update grail provider defaults and model catalogs. Use when
  checking for new model releases, updating default models, or listing available
  models from provider APIs.
---

# Discover LLM models for grail

Use this skill when auditing whether grail's provider defaults and `models.go`
constants are current. grail pins curated defaults in code; providers also expose
runtime discovery APIs.

## Where grail stores models

| Location | Purpose |
|----------|---------|
| `providers/openai/models.go` | OpenAI model constants |
| `providers/openai/openai.go` | OpenAI defaults and `ListModels()` catalog |
| `providers/gemini/models.go` | Gemini model constants |
| `providers/gemini/gemini.go` | Gemini defaults and `ListModels()` catalog |
| `providers/modelslab/models.go` | ModelsLab image models (no SDK; check their API docs) |

Selection priority in requests: explicit `Request.Model` > `Request.Tier` (`best`/`fast`) > provider default.

## Official discovery sources (preferred order)

1. **Provider SDK typed constants** — safest for grail's OpenAI provider because model IDs are compile-time checked.
2. **Provider SDK list APIs** — authoritative at runtime; good for Gemini and for spotting models not yet in SDK constants.
3. **Official release notes / changelog** — required for deprecations, GA renames, and shutdown dates.
4. **Provider model docs pages** — human-readable capability tables.

## OpenAI (`github.com/openai/openai-go/v3`)

### Typed constants (check first)

```bash
# Chat / Responses models
go doc -all github.com/openai/openai-go/v3/shared | rg 'ChatModel|ResponsesModel'

# Image models
go doc -all github.com/openai/openai-go/v3 | rg 'ImageModel'
```

Key packages:
- `shared.ChatModel*` — text models (current frontier: `ChatModelGPT5_4` → `gpt-5.4`)
- `openai.ImageModel*` — image models (current: `ImageModelGPTImage2` → `gpt-image-2`)

When adding a model to grail, prefer SDK constants over raw strings so upgrades surface missing IDs at compile time.

### Runtime list API

```go
client := openai.NewClient(option.WithAPIKey(os.Getenv("OPENAI_API_KEY")))
page, err := client.Models.List(ctx)
if err != nil { /* handle */ }
for _, m := range page.Data {
    fmt.Println(m.ID)
}
```

`client.Models.List` maps to `GET /v1/models`. Use this to find models announced in docs but not yet added to SDK constants (e.g. check for `gpt-5.5` before the Go SDK ships `ChatModelGPT5_5`).

### Docs to monitor

- Models catalog: https://developers.openai.com/api/docs/models
- Compare page: https://developers.openai.com/api/docs/models/compare
- Image models: https://developers.openai.com/api/docs/models/gpt-image-2

### grail update checklist (OpenAI)

1. Bump SDK: `go get github.com/openai/openai-go/v3@latest`
2. Re-run `go doc` commands above; note new `ChatModel*` / `ImageModel*` constants.
3. Update `DefaultTextModelName` / `DefaultImageModelName` in `providers/openai/openai.go` if frontier defaults changed.
4. Add new entries to `providers/openai/models.go` with capabilities.
5. Update `bestTextModel` / `fastTextModel` / image catalog slots in `New()`.
6. Run `go test ./providers/openai/...`

## Gemini (`google.golang.org/genai`)

Gemini does not publish model ID constants in the Go SDK. Use runtime listing plus official docs.

### Runtime list API

```go
client, err := genai.NewClient(ctx, &genai.ClientConfig{
    Backend: genai.BackendGeminiAPI,
    APIKey:  os.Getenv("GEMINI_API_KEY"),
})
if err != nil { /* handle */ }

for model, err := range client.Models.All(ctx) {
    if err != nil { /* handle */ }
  // model.Name is like "models/gemini-3.5-flash"
  fmt.Println(model.Name, model.DisplayName, model.SupportedActions)
}
```

Alternative paginated API: `client.Models.List(ctx, nil)`.

Filter listed models to those with `generateContent` or `generateImages` in `SupportedActions` depending on role.

### Docs to monitor

- Gemini 3 guide (current lineup): https://ai.google.dev/gemini-api/docs/gemini-3
- Release notes (deprecations + GA renames): https://ai.google.dev/gemini-api/docs/changelog
- Per-model pages: https://ai.google.dev/gemini-api/docs/models

### Recent migration patterns (as of mid-2026)

| Old (preview/shut down) | New (GA) | Notes |
|-------------------------|----------|-------|
| `gemini-3-pro-image-preview` | `gemini-3-pro-image` | Shut down 2026-06-25 |
| `gemini-3.1-flash-image-preview` | `gemini-3.1-flash-image` | Shut down 2026-06-25 |
| `gemini-3.1-flash-lite-preview` | `gemini-3.1-flash-lite` | Shut down 2026-05-25 |
| `gemini-3-flash-preview` | `gemini-3.5-flash` | 3.5 Flash is GA fast tier |
| `gemini-3-pro-preview` | `gemini-3.1-pro-preview` | Shut down 2026-03-09 |

Keep deprecated preview IDs as legacy constants (see `Gemini3Pro`, `Gemini3ProImagePreview`) so existing callers pinning old names get compile-time symbols, but do not use them as defaults.

### grail update checklist (Gemini)

1. Bump SDK: `go get google.golang.org/genai@latest`
2. Read release notes for shutdown dates — update defaults before shutdown.
3. Update string IDs in `providers/gemini/models.go`.
4. Update `DefaultTextModelName` / `DefaultImageModelName` and catalog slots in `providers/gemini/gemini.go`.
5. Add new models to `AllModels()` so `ListModels()` exposes them.
6. Run `go test ./providers/gemini/...`

## ModelsLab (no official Go SDK)

ModelsLab is HTTP-only in grail. Check:
- API docs: https://modelslab.com/docs
- Existing constants in `providers/modelslab/models.go`

No runtime discovery helper exists in grail; compare docs manually.

## Quick audit command sequence

```bash
# 1. Ensure latest SDKs
go get github.com/openai/openai-go/v3@latest google.golang.org/genai@latest
go mod tidy

# 2. Print OpenAI SDK model constants
go doc -all github.com/openai/openai-go/v3/shared | rg 'ChatModelGPT|ImageModel'

# 3. Compare with grail constants
rg 'Name:|DefaultTextModelName|DefaultImageModelName' providers/openai providers/gemini

# 4. Run tests
go test ./...
```

## Optional: live model listing (requires API keys)

```bash
# OpenAI — needs OPENAI_API_KEY
go run ./skills/discover-llm-models/cmd/listmodels -provider openai

# Gemini — needs GEMINI_API_KEY
go run ./skills/discover-llm-models/cmd/listmodels -provider gemini

# Both
go run ./skills/discover-llm-models/cmd/listmodels -provider all
```

The helper prints model IDs sorted, one per line, to stdout. Errors are actionable (missing key, unknown provider).

## When to change defaults vs. only add constants

- **Change defaults** when the provider docs mark a model as the recommended frontier/fast choice or when a preview is shutting down.
- **Only add constants** when a model is niche (dated snapshots, legacy pins, region-specific) or not a good default for grail's `best`/`fast` tiers.
- **Never remove** a constant without a deprecation comment and shutdown date if external callers may pin it.
