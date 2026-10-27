---
name: discover-llm-models
description: >-
  Find current LLM model IDs from OpenAI, Gemini and Anthropic SDKs and official
  docs, then audit or update grail provider defaults and model catalogs. Use when
  checking for new model releases, updating default models, listing available
  models from provider APIs, or acting on a nightly `model-drift` issue.
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
| `providers/anthropic/models.go` | Anthropic model constants and `KnownModels()` |
| `providers/anthropic/anthropic.go` | Anthropic defaults and `ListModels()` catalog |
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
- `shared.ChatModel*` — text models (current frontier: `ChatModelGPT6Astra` → `gpt-6-astra`; fast: `ChatModelGPT6Luna` → `gpt-6-luna`)
- `openai.ImageModel*` — image models (current: `ImageModelGPTImage2_5Sunburst` → `gpt-image-2.5-sunburst`; fast: `ImageModelGPTImage2_5Flare` → `gpt-image-2.5-flare`)

SDK constants carry `// Deprecated: Announced shutdown date: ...` comments — grep for them to find pinned models that are going away.

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

`client.Models.List` maps to `GET /v1/models`. Use this to find models announced in docs but not yet added to SDK constants (e.g. check for a new ID before the Go SDK ships its constant).

### Docs to monitor

- Models catalog: https://developers.openai.com/api/docs/models
- Compare page: https://developers.openai.com/api/docs/models/compare
- Deprecations (shutdown dates + replacements): https://developers.openai.com/api/docs/deprecations

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
  // model.Name is like "models/gemini-3.8-flash"
  fmt.Println(model.Name, model.DisplayName, model.SupportedActions)
}
```

Alternative paginated API: `client.Models.List(ctx, nil)`.

Filter listed models to those with `generateContent` or `generateImages` in `SupportedActions` depending on role.

### Docs to monitor

- Release notes (deprecations + GA renames): https://ai.google.dev/gemini-api/docs/changelog
- Per-model pages: https://ai.google.dev/gemini-api/docs/models

### Recent migration patterns (as of late October 2026)

| Old (preview/shut down) | New (GA) | Notes |
|-------------------------|----------|-------|
| `gemini-3-pro-image-preview` | `gemini-3-pro-image` | Shut down 2026-06-25 |
| `gemini-3.1-flash-image-preview` | `gemini-3.1-flash-image` | Shut down 2026-06-25 |
| `gemini-3.1-flash-lite-preview` | `gemini-3.1-flash-lite` | Shut down 2026-05-25 |
| `gemini-3.5-flash` | `gemini-3.8-flash` | Deprecated 2026-10-08, routes to `gemini-3.6-flash` |
| `gemini-3.7-flash` | `gemini-3.8-flash` | Deprecated 2026-10-08 |
| `gemini-3.1-flash-image` | `gemini-nano-banana-2.1` | Deprecated 2026-10-06, no shutdown date yet |
| `gemini-3.1-pro-preview` | `gemini-4-argon` | 4 Argon is the GA flagship (2026-10-13) |
| `gemini-3-flash-preview` | `gemini-3.8-flash` | Superseded preview |
| `gemini-3-pro-preview` | `gemini-3.1-pro-preview` | Shut down 2026-03-09 |

Keep deprecated preview IDs as legacy constants (see `Gemini3Pro`, `Gemini3ProImagePreview`) so existing callers pinning old names get compile-time symbols, but do not use them as defaults.

Note: Gemini deprecated the `temperature`, `top_p`, and `top_k` parameters on 2026-07-21. grail still passes them through from `TextOptions` when callers set them.

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
go doc -all github.com/openai/openai-go/v3/shared | rg 'ChatModelGPT'
go doc -all github.com/openai/openai-go/v3 | rg 'ImageModel'

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

# Anthropic — needs ANTHROPIC_API_KEY
go run ./skills/discover-llm-models/cmd/listmodels -provider anthropic

# Several, or all three
go run ./skills/discover-llm-models/cmd/listmodels -provider gemini,anthropic
go run ./skills/discover-llm-models/cmd/listmodels -provider all
```

The helper prints model IDs sorted, one per line, to stdout. Errors are actionable (missing key, unknown provider).

### Diff against grail's catalog

```bash
go run ./skills/discover-llm-models/cmd/listmodels -provider all -diff
```

`-diff` compares the live IDs with each provider's `KnownModels()` and prints a
Markdown section per provider:

- **Missing default/tier models**: `Default*ModelName` or a best/fast tier model
  is gone from the API. grail selects these, so requests fail. Fix first.
- **Missing**: a known model is gone from the API and grail has no past
  `Deprecation.Shutdown` for it, so it was most likely shut down. Known models
  already past their recorded shutdown date are expected to be gone and are not
  listed. A known alias counts as present when the API lists a dated snapshot of
  it (`claude-opus-4-5` via `claude-opus-4-5-20251101`). Anthropic models with
  lifecycle `retired` count as gone.
- **New**: served by the API but unknown to grail, filtered to text and image
  generation models. The filter drops fine-tunes (`ft:`), `-latest` aliases,
  snapshots (`-2026-10-20`, `-20251101`, `-0613`, `-001`) of an ID that is live
  or known, Gemini models without `generateContent`, and IDs containing
  `embed`, `tts`, `whisper`, `transcribe`, `audio`, `speech`, `realtime`,
  `live`, `moderation`, `search`, `computer-use`, `sora`, `veo`, `gemma`, `aqa`
  or `learnlm`. It also drops models older than grail's catalog: models released
  before the newest known model (OpenAI and Anthropic report release dates), or,
  where no dates are reported (Gemini), versions below the oldest default/tier
  model (`gemini-2.5-pro` when the oldest tier is `gemini-3-pro-image`).

Exit codes: `0` no missing models (new models alone are not an error), `1`
missing known or default/tier models, `2` usage error, `3` a provider could not
be listed (missing key or API error). The filtering and diff logic is a pure
function in `cmd/listmodels/diff.go` with unit tests in `diff_test.go`.

## Nightly drift check

`.github/workflows/model-drift.yml` runs `listmodels -diff` every night (06:17
UTC) and on manual dispatch. It reads the `OPENAI_API_KEY`, `GEMINI_API_KEY`
and `ANTHROPIC_API_KEY` repo secrets and skips any provider whose secret is
unset. It keeps a single open issue labeled `model-drift`:

- Drift (any missing or new model): creates the issue, or replaces the body of
  the open one with the latest diff. It never opens a second issue.
- No drift: closes the open issue with a comment linking the run.
- Missing models also fail the run, so the Actions tab shows it red. A listing
  error (bad key, API outage) fails the run without touching the issue.

### Acting on a `model-drift` issue

1. Re-run the diff locally to confirm: `go run ./skills/discover-llm-models/cmd/listmodels -provider <name> -diff`.
2. **Missing default/tier model**: pick a replacement from the provider docs,
   update `Default*ModelName` and the best/fast slots in `New()`, and ship a
   patch release. This is the urgent case.
3. **Missing model**: confirm the shutdown in the provider's deprecations page or
   changelog, then set `Deprecation.Shutdown` (and `Replacement`) on the
   constant in `providers/<name>/models.go`. Do not delete the constant.
4. **New model**: decide whether it belongs in grail. If yes, add a constant to
   `models.go` and to `KnownModels()`, and consider whether it should become a
   tier default (see below). A new model stays listed, and the issue stays open,
   until grail knows it or grail adds a newer model (OpenAI, Anthropic) or a
   newer tier generation (Gemini), so adding a niche model as a plain constant
   is the simplest way to clear it.
5. Run `make fmt-check lint test`. The issue closes itself on the next nightly
   run with no drift, or trigger the workflow manually after merging.

## When to change defaults vs. only add constants

- **Change defaults** when the provider docs mark a model as the recommended frontier/fast choice or when a preview is shutting down.
- **Only add constants** when a model is niche (dated snapshots, legacy pins, region-specific) or not a good default for grail's `best`/`fast` tiers.
- **Never remove** a constant without a deprecation comment and shutdown date if external callers may pin it.
