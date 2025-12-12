// Package grail provides a unified interface for AI text and image generation
// across multiple providers (OpenAI, Gemini, etc.). It supports multimodal
// inputs (ordered sequences of text and images) and provides type-safe error
// handling, structured logging, and flexible configuration options.
//
// Example usage:
//
//	provider, _ := openai.New()
//	client := grail.NewClient(provider)
//	res, _ := client.GenerateText(ctx, grail.TextRequest{
//		Input: []grail.Part{grail.Text("Hello, world!")},
//	})
//
// Sub-packages:
//
// This package provides the core client and interfaces. Provider implementations
// are available in sub-packages:
//
//   - [providers/openai](https://pkg.go.dev/github.com/montanaflynn/grail/providers/openai) - OpenAI provider
//   - [providers/gemini](https://pkg.go.dev/github.com/montanaflynn/grail/providers/gemini) - Google Gemini provider
package grail

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"strings"
)

// ErrorCode is a coarse-grained category for Grail errors.
type ErrorCode string

const (
	// CodeInvalidInput indicates a caller-provided input is invalid or missing.
	CodeInvalidInput ErrorCode = "invalid_input"
	// CodeBadOptions indicates caller-provided options are conflicting or invalid.
	CodeBadOptions ErrorCode = "bad_options"
	// CodeMissingCredentials indicates required credentials are absent.
	CodeMissingCredentials ErrorCode = "missing_credentials"
	// CodeUnsupported indicates the requested operation or feature is unsupported.
	CodeUnsupported ErrorCode = "unsupported"
	// CodeInternal indicates an unexpected internal failure.
	CodeInternal ErrorCode = "internal"
	// CodeUnknown is used when the error is not typed.
	CodeUnknown ErrorCode = "unknown"
)

// Error is a typed error used by the client and providers for predictable handling.
type Error struct {
	Code     ErrorCode
	Message  string
	Cause    error
	Metadata map[string]any
}

// Error implements the error interface.
func (e *Error) Error() string {
	if e == nil {
		return ""
	}
	if e.Cause != nil {
		return fmt.Sprintf("%s: %s: %v", e.Code, e.Message, e.Cause)
	}
	return fmt.Sprintf("%s: %s", e.Code, e.Message)
}

// Unwrap enables errors.Is/As traversal.
func (e *Error) Unwrap() error {
	return e.Cause
}

// Is allows errors.Is to match on error code.
func (e *Error) Is(target error) bool {
	t, ok := target.(*Error)
	if !ok {
		return false
	}
	// Match on code if set; otherwise fall back to pointer equality.
	if t.Code != "" && e.Code != t.Code {
		return false
	}
	return true
}

// ErrorOption mutates an Error during construction.
type ErrorOption func(*Error)

// WithCause attaches an underlying cause.
func WithCause(err error) ErrorOption {
	return func(e *Error) {
		e.Cause = err
	}
}

// WithMetadata sets metadata (caller-provided map is shallow-copied).
func WithMetadata(meta map[string]any) ErrorOption {
	return func(e *Error) {
		if len(meta) == 0 {
			return
		}
		e.Metadata = make(map[string]any, len(meta))
		for k, v := range meta {
			e.Metadata[k] = v
		}
	}
}

// NewError constructs an Error for the given code and message.
func NewError(code ErrorCode, msg string, opts ...ErrorOption) *Error {
	err := &Error{
		Code:    code,
		Message: msg,
	}
	for _, opt := range opts {
		if opt != nil {
			opt(err)
		}
	}
	return err
}

// InvalidInput creates a CodeInvalidInput error.
func InvalidInput(msg string, opts ...ErrorOption) *Error {
	return NewError(CodeInvalidInput, msg, opts...)
}

// BadOptions creates a CodeBadOptions error.
func BadOptions(msg string, opts ...ErrorOption) *Error {
	return NewError(CodeBadOptions, msg, opts...)
}

// MissingCredentials creates a CodeMissingCredentials error.
func MissingCredentials(msg string, opts ...ErrorOption) *Error {
	return NewError(CodeMissingCredentials, msg, opts...)
}

// Unsupported creates a CodeUnsupported error.
func Unsupported(msg string, opts ...ErrorOption) *Error {
	return NewError(CodeUnsupported, msg, opts...)
}

// Internal creates a CodeInternal error.
func Internal(msg string, opts ...ErrorOption) *Error {
	return NewError(CodeInternal, msg, opts...)
}

// GetErrorCode extracts the ErrorCode from an error, returning CodeUnknown if not present.
func GetErrorCode(err error) ErrorCode {
	var ge *Error
	if err == nil {
		return ""
	}
	if errors.As(err, &ge) && ge != nil {
		return ge.Code
	}
	return CodeUnknown
}

// AsError returns the underlying Error if present.
func AsError(err error) (*Error, bool) {
	var ge *Error
	if errors.As(err, &ge) {
		return ge, true
	}
	return nil, false
}

// IsCode reports whether the error chain contains an Error with the given code.
func IsCode(err error, code ErrorCode) bool {
	if err == nil {
		return false
	}
	return GetErrorCode(err) == code
}

// Provider is the pluggable backend surface; implement with Gemini, OpenAI, etc.
// See [providers](https://pkg.go.dev/github.com/montanaflynn/grail/providers) for available implementations.
type Provider interface {
	GenerateText(ctx context.Context, req TextRequest) (TextResult, error)
	GenerateImage(ctx context.Context, req ImageRequest) (ImageResult, error)
	DefaultTextModel() string  // configured/default text model used when Options.Model is empty
	DefaultImageModel() string // configured/default image model used when Options.Model is empty
}

// LoggerAware is an optional interface for providers to accept a logger from the client.
type LoggerAware interface {
	SetLogger(*slog.Logger)
}

// Client is a thin wrapper that delegates to a Provider. Swap providers to change backends.
// See [providers](https://pkg.go.dev/github.com/montanaflynn/grail/providers) for available implementations.
type Client struct {
	provider Provider
	log      *slog.Logger
}

// ClientOption mutates client configuration.
type ClientOption func(*ClientConfig)

// ClientConfig carries client-level settings.
type ClientConfig struct {
	Logger *slog.Logger
}

// WithLogger sets a custom logger for client-level logs.
func WithLogger(l *slog.Logger) ClientOption {
	return func(cfg *ClientConfig) {
		cfg.Logger = l
	}
}

// LoggerLevel is a small enum for convenience logger construction.
type LoggerLevel slog.Level

const (
	LoggerLevelDebug LoggerLevel = LoggerLevel(slog.LevelDebug)
	LoggerLevelInfo  LoggerLevel = LoggerLevel(slog.LevelInfo)
	LoggerLevelWarn  LoggerLevel = LoggerLevel(slog.LevelWarn)
	LoggerLevelError LoggerLevel = LoggerLevel(slog.LevelError)
)

var LoggerLevels = map[string]LoggerLevel{
	"debug": LoggerLevelDebug,
	"info":  LoggerLevelInfo,
	"warn":  LoggerLevelWarn,
	"error": LoggerLevelError,
}

// WithLoggerFormat builds a default logger at the given level and format ("text" or "json").
// This is a convenience if you don't want to construct a slog.Logger yourself.
func WithLoggerFormat(format string, level LoggerLevel) ClientOption {
	return func(cfg *ClientConfig) {
		handlerOpts := &slog.HandlerOptions{Level: slog.Level(level)}
		switch strings.ToLower(format) {
		case "json":
			cfg.Logger = slog.New(slog.NewJSONHandler(os.Stdout, handlerOpts))
		default:
			cfg.Logger = slog.New(slog.NewTextHandler(os.Stdout, handlerOpts))
		}
	}
}

// NewClient builds a Client from a Provider, applying functional options.
// See [providers](https://pkg.go.dev/github.com/montanaflynn/grail/providers) for available Provider implementations.
func NewClient(p Provider, opts ...ClientOption) *Client {
	cfg := ClientConfig{
		Logger: slog.Default(),
	}
	for _, opt := range opts {
		opt(&cfg)
	}
	if la, ok := p.(LoggerAware); ok {
		la.SetLogger(cfg.Logger)
	}
	return &Client{provider: p, log: cfg.Logger}
}

func (c *Client) GenerateText(ctx context.Context, req TextRequest) (TextResult, error) {
	if err := validateTextRequest(req); err != nil {
		return TextResult{}, err
	}

	model := req.Options.Model
	if model == "" {
		model = c.provider.DefaultTextModel()
	}
	if c.log != nil {
		attrs := []any{
			slog.String("model", model),
			slog.Int("parts", len(req.Input)),
			slog.Any("parts_summary", summarizeParts(req.Input)),
		}
		attrs = append(attrs, slog.Any("text_options", req.Options))
		c.log.Info("generate text request", attrs...)
	}
	return c.provider.GenerateText(ctx, req)
}

func (c *Client) GenerateImage(ctx context.Context, req ImageRequest) (ImageResult, error) {
	if err := validateImageRequest(req); err != nil {
		return ImageResult{}, err
	}

	model := req.Options.Model
	if model == "" {
		model = c.provider.DefaultImageModel()
	}
	if c.log != nil {
		attrs := []any{
			slog.String("model", model),
			slog.Int("parts", len(req.Input)),
			slog.Any("parts_summary", summarizeParts(req.Input)),
		}
		attrs = append(attrs, slog.Any("image_options", req.Options))
		c.log.Info("generate image request", attrs...)
	}
	return c.provider.GenerateImage(ctx, req)
}

// Part is an ordered multimodal input. Use Text(...) and Image(...) to build.
type Part interface {
	Kind() PartKind
}

// PartKind identifies the content type for a Part.
type PartKind string

const (
	PartText  PartKind = "text"
	PartImage PartKind = "image"
)

// TextPart represents a text segment.
type TextPart struct {
	Text string
}

func (TextPart) Kind() PartKind { return PartText }

// Text is a helper to build a text Part.
func Text(s string) Part { return TextPart{Text: s} }

// ImagePart represents an inline image payload.
type ImagePart struct {
	Data []byte // raw bytes, required
	MIME string // optional, defaults to image/png
}

func (ImagePart) Kind() PartKind { return PartImage }

// Image is a helper to build an image Part.
func Image(data []byte, mime string) Part {
	return ImagePart{Data: data, MIME: mime}
}

// TextRequest represents text generation from ordered multimodal input.
type TextRequest struct {
	Input   []Part
	Options TextOptions
}

// ImageRequest represents image generation from ordered multimodal input.
type ImageRequest struct {
	Input   []Part
	Options ImageOptions
	// ProviderOptions carries provider-specific option values (e.g., openai image tool options).
	// See [providers](https://pkg.go.dev/github.com/montanaflynn/grail/providers) for provider-specific options.
	ProviderOptions []ProviderOption
}

// TextOptions tunes per-call text generation. All fields are optional; nil or empty
// values let the provider choose its defaults.
type TextOptions struct {
	// Model override for this request. Leave empty to use the provider default.
	Model string
	// MaxTokens caps the number of output tokens (provider-enforced if supported).
	MaxTokens *int32
	// Temperature controls randomness; higher is more random. Use either Temperature or TopP, not both.
	Temperature *float32
	// TopP enables nucleus sampling (probability mass cutoff). Use either TopP or Temperature, not both.
	TopP *float32
	// SystemPrompt sets a system/developer instruction applied before user content.
	SystemPrompt string
}

// ImageOptions tunes image generation behavior; fields are optional.
type ImageOptions struct {
	Model        string
	SystemPrompt string
}

// ProviderOption is a provider-specific option attached to a request.
// Providers may type-assert to their own option types. Description is advisory.
// See [providers](https://pkg.go.dev/github.com/montanaflynn/grail/providers) for provider-specific option types.
type ProviderOption interface {
	Description() string
}

// TextResult holds the generated text and the raw provider response.
type TextResult struct {
	Text string
	Raw  any
}

// ImageResult holds generated images and the raw provider response.
type ImageResult struct {
	Images []ImageOutput
	Raw    any
}

// ImageOutput represents a single generated image.
type ImageOutput struct {
	Data []byte
	MIME string
}

// Pointer is a helper to take the address of a literal value (e.g., grail.Pointer(0.0)).
func Pointer[T any](v T) *T { return &v }

func summarizeParts(parts []Part) []map[string]any {
	var out []map[string]any
	for _, p := range parts {
		switch v := p.(type) {
		case TextPart:
			out = append(out, map[string]any{
				"type": "text",
				"len":  len(v.Text),
			})
		case ImagePart:
			out = append(out, map[string]any{
				"type": "image",
				"mime": v.MIME,
				"len":  len(v.Data),
			})
		default:
			out = append(out, map[string]any{
				"type": "unknown",
			})
		}
	}
	return out
}

func validateTextRequest(req TextRequest) error {
	if len(req.Input) == 0 {
		return InvalidInput("text input must not be empty")
	}
	if req.Options.Temperature != nil && req.Options.TopP != nil {
		return BadOptions("temperature and top_p are mutually exclusive",
			WithMetadata(map[string]any{
				"temperature": *req.Options.Temperature,
				"top_p":       *req.Options.TopP,
			}))
	}
	return nil
}

func validateImageRequest(req ImageRequest) error {
	if len(req.Input) == 0 {
		return InvalidInput("image input must not be empty")
	}
	return nil
}

// logging helpers removed; options structs are logged directly for maintainability.
