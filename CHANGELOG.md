<a name="unreleased"></a>
## [Unreleased]


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

