// Package gemini provides a Google Gemini implementation of the grail.Provider interface.
// It supports both text and image generation using Gemini models.
//
// Example usage:
//
//	provider, err := gemini.New(context.Background())
//	if err != nil {
//		log.Fatal(err)
//	}
//	client := grail.NewClient(provider)
//	res, err := client.GenerateText(ctx, grail.TextRequest{
//		Input: []grail.Part{grail.Text("Hello, world!")},
//	})
//
// The provider automatically uses the GEMINI_API_KEY environment variable
// if no API key is explicitly provided via WithAPIKey or WithAPIKeyFromEnv.
//
// Default models:
//   - Text: gemini-2.5-flash
//   - Image: gemini-2.5-flash-image
package gemini

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"strings"

	"github.com/montanaflynn/grail"

	"github.com/google/generative-ai-go/genai"
	"google.golang.org/api/option"
)

const (
	// DefaultTextModelName is the Gemini text model used when no override is provided.
	DefaultTextModelName = "gemini-2.5-flash"
	// DefaultImageModelName is the Gemini image model used when no override is provided.
	DefaultImageModelName = "gemini-2.5-flash-image"
)

var (
	// ErrAPIKeyRequired is returned when no API key is configured.
	ErrAPIKeyRequired = errors.New("gemini: API key required (set GEMINI_API_KEY or use WithAPIKey/WithAPIKeyFromEnv)")
)

// Option configures the Gemini provider.
type Option func(*settings)

type settings struct {
	apiKey     string
	apiKeySet  bool
	textModel  string
	imageModel string
	logger     *slog.Logger
}

// WithAPIKey sets the API key to use.
func WithAPIKey(key string) Option {
	return func(s *settings) {
		s.apiKeySet = true
		s.apiKey = key
	}
}

// WithAPIKeyFromEnv reads the API key from the given environment variable.
func WithAPIKeyFromEnv(env string) Option {
	return func(s *settings) {
		s.apiKeySet = true
		if v := os.Getenv(env); v != "" {
			s.apiKey = v
		}
	}
}

// WithTextModel overrides the default text model.
func WithTextModel(model string) Option {
	return func(s *settings) { s.textModel = model }
}

// WithImageModel overrides the default image model.
func WithImageModel(model string) Option {
	return func(s *settings) { s.imageModel = model }
}

// WithLogger sets a custom logger for provider-level logs.
func WithLogger(l *slog.Logger) Option {
	return func(s *settings) {
		if l != nil {
			s.logger = l
		}
	}
}

// Provider is a Gemini-backed implementation of grail.Provider.
type Provider struct {
	raw        *genai.Client
	textModel  string
	imageModel string
	log        *slog.Logger
}

// New constructs a Gemini provider using functional options.
func New(ctx context.Context, opts ...Option) (*Provider, error) {
	cfg := settings{
		textModel:  DefaultTextModelName,
		imageModel: DefaultImageModelName,
		logger:     slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: slog.LevelInfo})),
	}
	for _, opt := range opts {
		opt(&cfg)
	}

	switch {
	case cfg.apiKeySet && cfg.apiKey == "":
		return nil, ErrAPIKeyRequired
	case !cfg.apiKeySet && cfg.apiKey == "":
		cfg.apiKey = strings.TrimSpace(os.Getenv("GEMINI_API_KEY"))
		if cfg.apiKey == "" {
			return nil, ErrAPIKeyRequired
		}
	}

	clientOpts := []option.ClientOption{}
	if cfg.apiKey != "" {
		clientOpts = append(clientOpts, option.WithAPIKey(cfg.apiKey))
	}

	raw, err := genai.NewClient(ctx, clientOpts...)
	if err != nil {
		return nil, fmt.Errorf("new gemini client: %w", err)
	}

	return &Provider{
		raw:        raw,
		textModel:  cfg.textModel,
		imageModel: cfg.imageModel,
		log:        cfg.logger,
	}, nil
}

// SetLogger allows the client to inject a logger.
func (c *Provider) SetLogger(l *slog.Logger) {
	if l != nil {
		c.log = l
	}
}

// DefaultTextModel returns the configured/default text model.
func (c *Provider) DefaultTextModel() string {
	return c.textModel
}

// DefaultImageModel returns the configured/default image model.
func (c *Provider) DefaultImageModel() string {
	return c.imageModel
}

// GenerateText performs text generation using the configured Gemini text model.
func (c *Provider) GenerateText(ctx context.Context, req grail.TextRequest) (grail.TextResult, error) {
	if len(req.Input) == 0 {
		return grail.TextResult{}, errors.New("input must not be empty")
	}

	parts, err := toGenAIParts(req.Input)
	if err != nil {
		return grail.TextResult{}, err
	}

	modelName := req.Options.Model
	if modelName == "" {
		modelName = c.textModel
	}

	c.log.Debug("generate text request", slog.String("model", modelName), slog.Any("options", req.Options), slog.Any("parts", summarizeParts(req.Input)))

	model := c.raw.GenerativeModel(modelName)
	applyTextOptions(model, req.Options)

	resp, err := model.GenerateContent(ctx, parts...)
	if err != nil {
		return grail.TextResult{}, fmt.Errorf("generate text: %w", err)
	}

	return grail.TextResult{
		Text: firstText(resp),
		Raw:  resp,
	}, nil
}

// GenerateImage performs image generation using the configured Gemini image model.
func (c *Provider) GenerateImage(ctx context.Context, req grail.ImageRequest) (grail.ImageResult, error) {
	if len(req.Input) == 0 {
		return grail.ImageResult{}, errors.New("input must not be empty")
	}

	parts, err := toGenAIParts(req.Input)
	if err != nil {
		return grail.ImageResult{}, err
	}

	modelName := req.Options.Model
	if modelName == "" {
		modelName = c.imageModel
	}

	c.log.Debug("generate image request", slog.String("model", modelName), slog.Any("options", req.Options), slog.Any("parts", summarizeParts(req.Input)))

	model := c.raw.GenerativeModel(modelName)
	applyImageOptions(model, req.Options)

	resp, err := model.GenerateContent(ctx, parts...)
	if err != nil {
		return grail.ImageResult{}, fmt.Errorf("generate image: %w", err)
	}

	imgs := extractImages(resp)
	c.log.Debug("generate image response", slog.Int("images", len(imgs)), slog.Any("raw", resp))

	return grail.ImageResult{
		Images: imgs,
		Raw:    resp,
	}, nil
}

func toGenAIParts(input []grail.Part) ([]genai.Part, error) {
	out := make([]genai.Part, 0, len(input))
	for i, p := range input {
		switch v := p.(type) {
		case grail.TextPart:
			out = append(out, genai.Text(v.Text))
		case grail.ImagePart:
			if len(v.Data) == 0 {
				return nil, fmt.Errorf("part %d: image data is empty", i)
			}
			mime := v.MIME
			if mime == "" {
				mime = "image/png"
			}
			format := mime
			if strings.HasPrefix(mime, "image/") {
				format = mime[len("image/"):]
			}
			out = append(out, genai.ImageData(format, v.Data))
		default:
			return nil, fmt.Errorf("part %d: unknown part type %T", i, p)
		}
	}
	return out, nil
}

func applyTextOptions(model *genai.GenerativeModel, opts grail.TextOptions) {
	if opts.SystemPrompt != "" {
		model.SystemInstruction = &genai.Content{
			Parts: []genai.Part{
				genai.Text(opts.SystemPrompt),
			},
		}
	}
	if opts.Temperature != nil {
		model.SetTemperature(*opts.Temperature)
	}
	if opts.TopP != nil {
		model.SetTopP(*opts.TopP)
	}
	if opts.MaxTokens != nil {
		model.SetMaxOutputTokens(*opts.MaxTokens)
	}
}

func applyImageOptions(model *genai.GenerativeModel, opts grail.ImageOptions) {
	if opts.SystemPrompt != "" {
		model.SystemInstruction = &genai.Content{
			Parts: []genai.Part{
				genai.Text(opts.SystemPrompt),
			},
		}
	}
}

func firstText(resp *genai.GenerateContentResponse) string {
	for _, cand := range resp.Candidates {
		if cand == nil || cand.Content == nil {
			continue
		}
		for _, part := range cand.Content.Parts {
			if t, ok := part.(genai.Text); ok {
				return string(t)
			}
		}
	}
	return ""
}

func extractImages(resp *genai.GenerateContentResponse) []grail.ImageOutput {
	var out []grail.ImageOutput
	for _, cand := range resp.Candidates {
		if cand == nil || cand.Content == nil {
			continue
		}
		for _, part := range cand.Content.Parts {
			switch v := part.(type) {
			case genai.Blob:
				out = append(out, grail.ImageOutput{
					Data: v.Data,
					MIME: v.MIMEType,
				})
			case *genai.Blob:
				out = append(out, grail.ImageOutput{
					Data: v.Data,
					MIME: v.MIMEType,
				})
			}
		}
	}
	return out
}

func summarizeParts(parts []grail.Part) []map[string]any {
	var out []map[string]any
	for _, p := range parts {
		switch v := p.(type) {
		case grail.TextPart:
			out = append(out, map[string]any{
				"type": "text",
				"len":  len(v.Text),
				"text": v.Text,
			})
		case grail.ImagePart:
			out = append(out, map[string]any{
				"type": "image",
				"mime": v.MIME,
				"len":  len(v.Data),
			})
		default:
			out = append(out, map[string]any{
				"type": fmt.Sprintf("unknown:%T", p),
			})
		}
	}
	return out
}
