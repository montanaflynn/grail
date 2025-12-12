Contributing
============

Thanks for helping improve Grail.

Prerequisites
- Go 1.22 or newer

Setup
- Clone the repo
- (Optional) Enable local hooks: `git config core.hooksPath .githooks`
  - `commit-msg` enforces conventional commits
  - `pre-commit` runs gofmt on staged Go files
- Please follow the `CODE_OF_CONDUCT.md`.

Development workflow
- Install dependencies: `go mod download`
- Run tests: `go test ./...`
- Keep changes `gofmt`-ed (most editors do this automatically)
- Prefer small, focused PRs
- Make targets: `make fmt`, `make lint`, `make test`
- `make` runs fmt, lint, and test in order

Commit messages
- Conventional style: `<type>(optional-scope)!: summary`
- Common types: build, chore, ci, docs, feat, fix, perf, refactor, revert, style, test
- The hook in `.githooks/commit-msg` enforces this if you opt into it.

Reporting issues
- Include Go version, provider used (OpenAI/Gemini), and repro steps or sample code.

Code of Conduct

This project follows the [Contributor Covenant](https://www.contributor-covenant.org/version/2/1/code_of_conduct/).

In brief:
- Use respectful and inclusive language.
- Be considerate of differing viewpoints and experiences.
- Give and gracefully accept constructive feedback.
- Focus on what is best for the community.
