# Plan: Model churn resilience

## Overview

Reposition grail around one promise: **your code keeps working when models churn.** Callers ask for what they need (text or image, best or fast tier, JSON shaped like a struct), and grail picks a current model and provider for it. It warns before a pinned model is shut down and can switch to the replacement automatically. A nightly check catches catalog drift so patch releases keep defaults current.

The work restructures `grail.Client` into a concrete struct with a single provider-selection seam. That unblocks multi-provider routing, convenience helpers, and streaming without units colliding in `grail.go`. It also adds deprecation metadata to the model catalog, sends JSON schemas to providers' native structured-output APIs (today neither provider does), adds an Anthropic provider (text, JSON, vision and PDF; no image generation), and rewrites the README around the before/after deprecation story.

Backward compatibility is **not** a constraint: grail is pre-1.0, and breaking API changes are acceptable when they simplify the design. Note each break in `CHANGELOG.md` under Unreleased.

**Out of scope for the whole plan:** tool calling, agents, conversation memory and multi-turn chat, and fetching a model catalog over the network at runtime.

Conventions for every unit:
- Go 1.26. Run `make fmt-check lint test` (gofmt, `go vet ./...`, `go test ./...`). All must pass.
- New provider-facing behavior is tested with fakes or `httptest` servers, never live API calls.
- Conventional commits, no `Co-Authored-By` lines.

## Dependency graph

- Wave 1: Unit 1, Unit 2, Unit 4   (independent)
- Wave 2: Unit 3 (needs 2), Unit 5 (needs 1), Unit 6 (needs 1, 4), Unit 7 (needs 1)
- Wave 3: Unit 8 (needs 1, 3, 4), Unit 9 (needs 2, 3), Unit 10 (needs 1, 2, 7), Unit 11 (needs 3, 7)
- Wave 4: Unit 12 (needs 5, 6, 7, 8, 10, 11)

Critical path: Unit 1 → Unit 7 → Unit 10 → Unit 12 (also Unit 2 → Unit 3 → Unit 8 → Unit 12)

File ownership to keep parallel units from conflicting:
- `grail.go`: Unit 1 restructures it. Later units add new files (`helpers.go`, `json.go`, `router.go`, `stream.go`, `deprecation.go`) and touch `grail.go` only where stated.
- `providers/*/models.go`: Unit 2 (and Unit 3 for Anthropic).
- `generateJSON` in `providers/openai/openai.go` and `providers/gemini/gemini.go`: Unit 4. Streaming (Unit 8) edits the same files later, in wave 3.

## Unit 1: Make Client a concrete struct with a provider-selection seam

**Depends on:** none
**Wave:** 1

Replace the `grail.Client` interface and unexported `client` struct with an exported concrete `*grail.Client`. This lets later units add methods in their own files without editing a shared interface. Move model resolution and capability validation out of `Generate` into a single selection method. Later, routing (Unit 7) and streaming (Unit 8) can replace or reuse it without rewriting `Generate`.

**Acceptance criteria**
- [ ] `grail.NewClient(p Provider, opts ...ClientOption) *Client` returns a concrete exported struct. The `Client` interface type is removed, and all existing methods (`Generate`, `InputFileFromURI`, `InputImageFromURI`, `InputPDFFromURI`, `ListModels`, `GetModel`) are methods on `*Client`.
- [ ] The client stores its providers as a slice (`providers []ProviderExecutor`, length 1 for now). `Generate` calls an unexported `selectProvider(ctx, req) (ProviderExecutor, Request, error)`. That method does today's Model > Tier > default resolution and capability validation and returns the resolved request. `Generate` then calls `DoGenerate`.
- [ ] `validateModelCapabilities` uses the caller's `ctx` instead of `context.Background()`. All examples and tests compile, and `make fmt-check lint test` passes with no behavior change.

**Files likely to touch**
- `grail.go`: `Client` type, `NewClient`, `Generate`, `validateModelCapabilities`, `ListModels`, `GetModel`
- `grail_test.go`, `examples/*/main.go`: update for `*grail.Client` if anything references the interface type

**Out of scope**
- Multiple providers, fallback, new helpers, streaming

## Unit 2: Add deprecation metadata to the model catalog

**Depends on:** none
**Wave:** 1

Give every model a machine-readable deprecation record, and let providers expose their **full** catalog including legacy models. Today `AllModels()` only returns the configured best/fast slots plus a few extras, so the client can't see models like `openai.GPTImage1`.

**Acceptance criteria**
- [ ] `grail.Model` gains `Deprecation *Deprecation`. `type Deprecation struct { Announced, Shutdown time.Time; Replacement string; Notes string }`. A zero `Shutdown` means no date has been announced. Add `func (m Model) IsDeprecated() bool` and `func (m Model) IsShutDown(now time.Time) bool`.
- [ ] A new optional provider interface `KnownModelLister { KnownModels() []Model }` returns every model constant the package exports, including deprecated ones. It's implemented by `providers/openai` and `providers/gemini`, with the list kept in each `models.go` next to the constants.
- [ ] Deprecation records are filled in from the dates already in the doc comments: OpenAI `GPTImage1` (shut down 2026-10-23, use `gpt-image-2.5-sunburst`), `GPTImage1Mini` (2026-12-01, `gpt-image-2.5-flare`), `GPT5_4Nano` (2027-04-01, `gpt-6-luna`), `GPT4o` (2027-01-14, `gpt-6-luna`). Gemini `Gemini3Pro` (2026-03-09, `gemini-4-argon`), `Gemini3ProImagePreview` and `Gemini3_1FlashImagePreview` (2026-06-25, GA equivalents), `Gemini3_5Flash` (deprecated 2026-10-08, `gemini-3.8-flash`), `Gemini3_1FlashImage` (deprecated 2026-10-06, `gemini-nano-banana-2.1`), `Gemini3Flash` (`gemini-3.8-flash`). A test in each provider asserts that every `Replacement` names a model in `KnownModels()` that is not itself deprecated.

**Files likely to touch**
- `grail.go`: `Model`, new `Deprecation` type, `KnownModelLister` interface (or a new `deprecation.go`)
- `providers/openai/models.go`, `providers/gemini/models.go`: records plus `KnownModels()`
- `providers/openai/openai_test.go`, `providers/gemini/gemini_test.go`

**Out of scope**
- Emitting warnings or swapping models at request time (Unit 10)
- ModelsLab deprecation data

## Unit 3: Add an Anthropic provider

**Depends on:** Unit 2
**Wave:** 2

Add `providers/anthropic` supporting text generation, JSON output, image understanding and PDF understanding. Claude has no image generation, so requests for image output return a typed `grail.Unsupported` error.

**Acceptance criteria**
- [ ] `anthropic.New(opts...)` follows the shape of `providers/openai`: `WithAPIKey`, `WithAPIKeyFromEnv` (default `ANTHROPIC_API_KEY`), `WithTextModel`, `WithLogger`. It implements `ProviderExecutor`, `ModelResolver`, `ModelLister`, `ModelCatalog` and `KnownModelLister`. It uses the official Go SDK (`github.com/anthropics/anthropic-sdk-go`), and model IDs come from the SDK's typed constants where they exist.
- [ ] The text best and fast tiers are the current most-capable model and the current fast/cheap model. Verify them against the SDK constants and Anthropic's models docs at implementation time; do not copy IDs from memory. Image-role `ResolveModel` and image-output `DoGenerate` return `grail.Unsupported`. Image and PDF inputs map to the Messages API's image and document content blocks.
- [ ] JSON output passes the request schema to the API's native structured-output mechanism instead of only parsing text afterward. Text, JSON, image-input, PDF-input and image-output-unsupported paths are covered by tests against an `httptest` server. A README provider section and an `examples/anthropic-text` example are added.

**Files likely to touch**
- `providers/anthropic/anthropic.go`, `providers/anthropic/models.go`, `providers/anthropic/anthropic_test.go` (new)
- `examples/anthropic-text/main.go` (new), `README.md` (provider section only), `go.mod`, `go.sum`

**Out of scope**
- Streaming (Unit 8), tool use, extended-thinking controls beyond what `TextOptions` already covers

## Unit 4: Send JSON schemas to provider structured-output APIs

**Depends on:** none
**Wave:** 1

`grail.OutputJSON(schema)` currently ignores the schema. `generateJSON` in both providers asks for plain text and only checks that it parses (see the "JSON mode may not be available" comments). Wire the schema into each API's native structured output so the response is guaranteed to match it.

**Acceptance criteria**
- [ ] `OutputJSON` documents that `schema` is a JSON Schema value (a `map[string]any`, `json.RawMessage`, or anything that marshals to a JSON Schema object). A nil schema means "any valid JSON object".
- [ ] OpenAI `generateJSON` sets the Responses API text format to `json_schema` with the schema and `Strict` flag (JSON-object mode when the schema is nil). Gemini `generateJSON` sets `ResponseMIMEType: "application/json"` and passes the schema through the genai SDK's JSON-schema config field.
- [ ] Tests (fake HTTP servers or request capture) assert that the outgoing request carries the schema for both providers, and that the existing parse validation still returns `grail.OutputInvalid` on malformed output.

**Files likely to touch**
- `grail.go`: `OutputJSON` doc comment only
- `providers/openai/openai.go`: `generateJSON`
- `providers/gemini/gemini.go`: `generateJSON`
- `providers/openai/openai_test.go`, `providers/gemini/gemini_test.go`

**Out of scope**
- Generating schemas from Go types (Unit 6)

## Unit 5: Add Text and Image convenience methods

**Depends on:** Unit 1
**Wave:** 2

Cut the quick start down to one call each. These are thin wrappers over `Generate`.

**Acceptance criteria**
- [ ] `func (c *Client) Text(ctx, prompt string, opts ...CallOption) (string, error)` and `func (c *Client) Image(ctx, prompt string, opts ...CallOption) ([]byte, error)` live in a new `helpers.go`. `CallOption` covers at least `WithTier(ModelTier)`, `WithModel(string)`, `WithInputs(...Input)` (extra inputs such as images or PDFs after the prompt) and `WithProviderOptions(...ProviderOption)`.
- [ ] Exported package-level aliases `grail.Best` and `grail.Fast` (equal to `ModelTierBest`/`ModelTierFast`) make tier selection read naturally: `client.Image(ctx, "…", grail.WithTier(grail.Fast))`.
- [ ] Tests using `providers/mock` cover the text and image paths, option application, and error propagation when the response has no text or image part.

**Files likely to touch**
- `helpers.go`, `helpers_test.go` (new)
- `examples/simple-text/main.go`: switch to `client.Text`

**Out of scope**
- JSON helper (Unit 6), streaming (Unit 8)

## Unit 6: Add a generic JSON[T] helper with schema derivation

**Depends on:** Unit 1, Unit 4
**Wave:** 2

Let callers get typed structured output from a Go struct with no hand-written schema.

**Acceptance criteria**
- [ ] `func JSON[T any](ctx context.Context, c *Client, prompt string, opts ...CallOption) (T, error)` in a new `json.go` derives a JSON Schema from `T` by reflection. It requests output with `OutputJSON(schema)` and decodes into `T`. If Unit 5 has not merged yet, define `CallOption` here and let Unit 5 reuse it; whichever unit lands second adapts.
- [ ] Schema derivation (`SchemaFor[T]() map[string]any`, exported) handles structs, nested structs, slices, maps with string keys, pointers (nullable), primitive types and `time.Time` (string, `date-time`). It respects `json:"name,omitempty"` (omitempty fields are not `required`, `-` is skipped) and a `jsonschema:"description=…,enum=a|b"` tag. It sets `additionalProperties: false` so it works with OpenAI strict mode. No new third-party dependency.
- [ ] Table tests cover schema output for each supported kind, and a mock-provider test covers the end-to-end decode.

**Files likely to touch**
- `json.go`, `json_test.go` (new)
- `examples/` : add `examples/json-output/main.go`

**Out of scope**
- `oneOf`/`anyOf` unions, recursive types (return a clear error)

## Unit 7: Route across multiple providers with fallback

**Depends on:** Unit 1
**Wave:** 2

Let one client hold several providers. Each request goes to the first provider whose catalog can serve it. On a retryable failure, the client falls back to the next capable provider at the same tier.

**Acceptance criteria**
- [ ] `grail.WithProviders(ps ...Provider) ClientOption` appends providers after the one passed to `NewClient`, in priority order. `selectProvider` becomes a candidate list: providers whose `ResolveModel` succeeds for the request's role and tier (or whose catalog contains `req.Model` when it is set), filtered by the capabilities required by the inputs (image input needs `ImageUnderstanding`, PDF input needs `PDFUnderstanding`) and the output (image output skips text-only providers).
- [ ] `Generate` tries candidates in order. It moves to the next only when `grail.IsRetryable(err)` is true or the error is `RateLimited`/`Unavailable`. The response's `Provider` field and a `Warning{Code: "fallback"}` record which provider served it and which were skipped. Explicit `req.Model` disables fallback to other providers. With no capable provider, return `grail.Unsupported` naming the missing capability.
- [ ] `ListModels`/`GetModel` aggregate across providers. Tests with `providers/mock` cover routing by role, routing by input capability, fallback on a retryable error, no fallback on a non-retryable error, and no fallback when `req.Model` is set.

**Files likely to touch**
- `router.go`, `router_test.go` (new)
- `grail.go`: `clientOpt` (provider list), `NewClient`, replace the `selectProvider` body by calling into `router.go`
- `providers/mock/mock.go`: allow configuring capabilities and injected errors if not already possible

**Out of scope**
- Retrying the same provider (backoff), cost-based or latency-based routing

## Unit 8: Add text streaming

**Depends on:** Unit 1, Unit 3, Unit 4
**Wave:** 3

Stream text output for OpenAI, Gemini and Anthropic. Both existing providers carry `TODO: support streaming` comments.

**Acceptance criteria**
- [ ] `func (c *Client) Stream(ctx, req Request) iter.Seq2[StreamEvent, error]` lives in a new `stream.go`. `StreamEvent` carries a text delta and, on the final event, `Usage` and `Provider` info. Only `OutputText()` requests are supported; others yield a `grail.Unsupported` error. Provider selection goes through `selectProvider` (with multiple providers it picks the first candidate and does not fall back mid-stream).
- [ ] A new optional provider interface `StreamExecutor { DoStream(ctx, Request) iter.Seq2[StreamEvent, error] }` is implemented by `providers/openai` (Responses streaming), `providers/gemini` (`GenerateContentStream`) and `providers/anthropic` (Messages streaming). The TODO comments are removed. Breaking out of the loop cancels the underlying stream.
- [ ] Tests for each provider replay a canned SSE or stream response from an `httptest` server and assert the deltas and final usage. A `providers/mock` stream supports the client-level tests. `client.Text` can optionally use streaming internally, but doesn't have to.

**Files likely to touch**
- `stream.go`, `stream_test.go` (new)
- `providers/openai/openai.go`, `providers/gemini/gemini.go`, `providers/anthropic/anthropic.go`: `DoStream`
- `providers/mock/mock.go`
- `examples/streaming/main.go` (new)

**Out of scope**
- Streaming JSON or images

## Unit 9: Add a nightly model catalog drift check

**Depends on:** Unit 2, Unit 3
**Wave:** 3

Automate the manual `discover-llm-models` audit. Every night, compare live provider model lists against grail's `KnownModels()` and open or update a GitHub issue when they diverge.

**Acceptance criteria**
- [ ] `skills/discover-llm-models/cmd/listmodels` supports `-provider anthropic` (and includes it in `all`). A new `-diff` flag compares live IDs against the corresponding provider's `KnownModels()`. It prints `missing` (known to grail but gone from the API, which means it was shut down) and `new` (in the API but unknown to grail, filtered to text and image generation models), and exits non-zero when `missing` is non-empty or any default/tier model is missing.
- [ ] `.github/workflows/model-drift.yml` runs on a nightly `schedule` cron and on `workflow_dispatch`. It reads `OPENAI_API_KEY`, `GEMINI_API_KEY` and `ANTHROPIC_API_KEY` from repo secrets, skipping any provider whose secret is unset. On drift it creates or updates a single open issue labeled `model-drift` with the diff, instead of opening a new issue each night. It closes that issue when drift clears.
- [ ] The diff logic is a pure function with unit tests, and the `discover-llm-models` skill doc describes the nightly check and how to act on its issue.

**Files likely to touch**
- `skills/discover-llm-models/cmd/listmodels/main.go` (and a small testable `diff.go` beside it)
- `.github/workflows/model-drift.yml` (new)
- `skills/discover-llm-models/SKILL.md`

**Out of scope**
- Auto-editing `models.go` or cutting releases from the workflow. A human, or an agent following the skill, acts on the issue.
- Scraping deprecation announcements from docs

## Unit 10: Emit deprecation warnings and support auto-migration

**Depends on:** Unit 1, Unit 2, Unit 7
**Wave:** 3

This unit is the user-visible half of the value prop. Requests that use a deprecated model succeed but come back with a warning. Callers can opt in to having shut-down models replaced automatically.

**Acceptance criteria**
- [ ] After provider selection, the client looks up the resolved model in the selected provider's `KnownModels()`. If it is deprecated, the client appends `Warning{Code: "model_deprecated", Message: "<model> is deprecated; shuts down <date>; use <replacement>"}` to `Response.Warnings` and logs it once per model per client at `Warn` level.
- [ ] With `grail.WithAutoMigrate()`, a model whose `Shutdown` is in the past is replaced by its `Replacement` before the call. The response carries `Warning{Code: "model_migrated"}` naming both models. Without the option, the request goes through unchanged and the provider's error surfaces. The clock is injectable for tests (an unexported `now func() time.Time`).
- [ ] Mock-provider tests cover: no warning for a current model, a warning for a deprecated-not-shut-down model, a migration with the option on, no migration with it off, and a replacement chain resolved at most one hop.

**Files likely to touch**
- `deprecation.go`, `deprecation_test.go` (new)
- `grail.go`: `clientOpt` (auto-migrate flag, clock) and a one-line call from `Generate`
- `router.go`: only if the lookup needs the selected provider handle

**Out of scope**
- Fetching deprecation data at runtime

## Unit 11: Add grail/auto.FromEnv

**Depends on:** Unit 3, Unit 7
**Wave:** 3

Zero-config setup. Build a multi-provider client from whichever API keys are present. This has to live in its own package, because provider packages import `grail` and `grail` can't import them back.

**Acceptance criteria**
- [ ] `auto.FromEnv(ctx context.Context, opts ...grail.ClientOption) (*grail.Client, error)` in `github.com/montanaflynn/grail/auto` constructs providers for each of `OPENAI_API_KEY`, `GEMINI_API_KEY` and `ANTHROPIC_API_KEY` that is set, in a documented default priority order. `GRAIL_PROVIDERS` (comma-separated, e.g. `anthropic,gemini`) overrides the order and subset.
- [ ] It returns a clear error listing the env vars checked when none are set, and an error for unknown names in `GRAIL_PROVIDERS`.
- [ ] Tests use `t.Setenv` to cover none set, one set, all set, and override ordering, without making network calls.

**Files likely to touch**
- `auto/auto.go`, `auto/auto_test.go` (new)

**Out of scope**
- ModelsLab (opt-in only, via explicit construction)
- Config files

## Unit 12: Reposition the README around model churn

**Depends on:** Unit 5, Unit 6, Unit 7, Unit 8, Unit 10, Unit 11
**Wave:** 4

Rewrite the README pitch and quick start to sell the value prop with a concrete example.

**Acceptance criteria**
- [ ] The tagline and opening section lead with "your code keeps working when models churn". A before/after example shows code that pinned `gpt-image-1` breaking at its 2026-10-23 shutdown, while code using `Tier: Best` moved to `gpt-image-2.5-sunburst`. A second example shows the `model_deprecated` warning output.
- [ ] The quick start uses `auto.FromEnv`, `client.Text`, `client.Image` and `grail.JSON[T]` and fits in about 25 lines. It is followed by short sections on multi-provider routing and fallback (including "Claude for text, Gemini for images"), streaming, deprecation warnings and auto-migrate, and the nightly catalog check. The `Request`-level API moves to an "Advanced" section.
- [ ] Every code block in the README compiles: add an `examples/readme` program or `Example*` test functions mirroring them, and keep them in `go test ./...`. The `examples/` index is updated, and `CHANGELOG.md` Unreleased lists the breaking changes from Units 1 and 7.

**Files likely to touch**
- `README.md`, `CHANGELOG.md`
- `example_test.go` (new) or `examples/readme/main.go`

**Out of scope**
- A docs site, logo, or launch posts
