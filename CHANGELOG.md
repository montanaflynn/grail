<a name="unreleased"></a>
## [Unreleased]

### Code Refactoring

- make Client a concrete struct with a provider-selection seam

### Documentation

- reposition README around model churn and add compile-checked examples

### Features

- log per-request details at Debug so the client is quiet by default
- **auto:** add FromEnv to build a client from provider API keys
- **listmodels:** add anthropic and -diff against KnownModels, and a nightly model drift check
- **listmodels:** acknowledge models and tighten new-model filter
- **providers:** add Anthropic Claude provider (its TextOptions has no Temperature, TopP or TopK)
- **providers:** send JSON schemas to native structured-output APIs
- warn on deprecated models and support auto-migration
- add text streaming for OpenAI, Gemini and Anthropic
- route requests across multiple providers with fallback
- add generic JSON[T] helper with schema derivation
- add Text and Image convenience methods with call options
- add deprecation metadata and KnownModels to model catalog
- update Gemini defaults to GA image models and Gemini 3.5 Flash fast tier
- add skill for discovering LLM models via provider SDKs
- update OpenAI defaults to GPT-6 Astra/Luna and GPT Image 2.5 Sunburst/Flare
- update Gemini defaults to Gemini 4 Argon, Gemini 3.8 Flash, and Nano Banana 2.1

### Chores

- **deps:** bump github.com/openai/openai-go/v3 to 3.89.0 and google.golang.org/genai to 1.76.0
- require Go 1.26 (needed by latest openai-go and Google Cloud deps)

### BREAKING CHANGE

- `grail.Client` is now a concrete struct: `grail.NewClient` returns `*grail.Client` instead of the `grail.Client` interface. Replace `grail.Client` with `*grail.Client` in your types; to fake the client in tests, wrap it in your own interface or use `providers/mock`.
- When routing by tier or default model, a tier that no provider can resolve, or a request needing a capability no provider has, now returns a `grail.Unsupported` error instead of `grail.InvalidArgument`. Code checking `grail.GetErrorCode(err) == grail.InvalidArgument` for these cases should check `grail.Unsupported`.

<a name="v0.4.1"></a>
## [v0.4.1] - 2026-04-24

### Chores

- **deps:** bump github.com/openai/openai-go/v3 from 3.14.0 to 3.26.0 ([#25](https://github.com/montanaflynn/grail/issues/25))
- **deps:** bump google.golang.org/genai from 1.47.0 to 1.49.0 ([#24](https://github.com/montanaflynn/grail/issues/24))
- **deps:** bump github.com/openai/openai-go/v3 from 3.14.0 to 3.22.0 ([#20](https://github.com/montanaflynn/grail/issues/20))
- **deps:** bump google.golang.org/genai from 1.39.0 to 1.47.0 ([#19](https://github.com/montanaflynn/grail/issues/19))

### Features

- update defaults to latest OpenAI and Gemini models ([#36](https://github.com/montanaflynn/grail/issues/36))


<a name="v0.4.0"></a>
## [v0.4.0] - 2026-02-21

### Features

- Add ModelsLab provider for text-to-image generation ([#21](https://github.com/montanaflynn/grail/issues/21))


<a name="v0.3.0"></a>
## [v0.3.0] - 2025-12-17

### Features

- add ModelDescriber interface and update model-selection example ([#7](https://github.com/montanaflynn/grail/issues/7))
- add model role and tier taxonomy with request-level selection ([#6](https://github.com/montanaflynn/grail/issues/6))


<a name="v0.2.2"></a>
## [v0.2.2] - 2025-12-16

### Chores

- **deps:** bump golang.org/x/crypto ([#5](https://github.com/montanaflynn/grail/issues/5))


<a name="v0.2.1"></a>
## [v0.2.1] - 2025-12-16

### Bug Fixes

- detect MIME type for InputImage with empty MIME
- use type conversion instead of struct literal for ImageOutputInfo

### Code Refactoring

- **openai:** remove gpt-image-1.5 as it doesn't work yet

### Documentation

- add package and example documentation comments

### Features

- update OpenAI defaults to gpt-5.2 and gpt-image-1.5


<a name="v0.2.0"></a>
## [v0.2.0] - 2025-12-14

### Code Refactoring

- migrate to unified API with direction-first naming ([#4](https://github.com/montanaflynn/grail/issues/4))


<a name="v0.1.5"></a>
## [v0.1.5] - 2025-12-13

### Documentation

- add FUNDING.yml and SECURITY.md

### Features

- add provider-specific image size and aspect ratio options ([#3](https://github.com/montanaflynn/grail/issues/3))


<a name="v0.1.4"></a>
## [v0.1.4] - 2025-12-13

### Documentation

- update package documentation

### Features

- Add PDF file input support ([#2](https://github.com/montanaflynn/grail/issues/2))


<a name="v0.1.3"></a>
## [v0.1.3] - 2025-12-12

### Documentation

- update package documentation


<a name="v0.1.2"></a>
## [v0.1.2] - 2025-12-12

### Documentation

- improve README and add package documentation


<a name="v0.1.1"></a>
## [v0.1.1] - 2025-12-12

### Documentation

- add Links sections for pkg.go.dev sidebar
- fix package comment by removing markdown link syntax
- add links to provider sub-packages in documentation
- improve documentation


<a name="v0.1.0"></a>
## v0.1.0 - 2025-12-12

### Features

- initial release with unified AI provider interface


[Unreleased]: https://github.com/montanaflynn/grail/compare/v0.4.1...HEAD
[v0.4.1]: https://github.com/montanaflynn/grail/compare/v0.4.0...v0.4.1
[v0.4.0]: https://github.com/montanaflynn/grail/compare/v0.3.0...v0.4.0
[v0.3.0]: https://github.com/montanaflynn/grail/compare/v0.2.2...v0.3.0
[v0.2.2]: https://github.com/montanaflynn/grail/compare/v0.2.1...v0.2.2
[v0.2.1]: https://github.com/montanaflynn/grail/compare/v0.2.0...v0.2.1
[v0.2.0]: https://github.com/montanaflynn/grail/compare/v0.1.5...v0.2.0
[v0.1.5]: https://github.com/montanaflynn/grail/compare/v0.1.4...v0.1.5
[v0.1.4]: https://github.com/montanaflynn/grail/compare/v0.1.3...v0.1.4
[v0.1.3]: https://github.com/montanaflynn/grail/compare/v0.1.2...v0.1.3
[v0.1.2]: https://github.com/montanaflynn/grail/compare/v0.1.1...v0.1.2
[v0.1.1]: https://github.com/montanaflynn/grail/compare/v0.1.0...v0.1.1

