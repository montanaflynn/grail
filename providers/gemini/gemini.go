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

	"google.golang.org/genai"
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
	client     *genai.Client
	textModel  string
	imageModel string
	log        *slog.Logger
}

// ImageAspectRatio enumerates supported Gemini image aspect ratios.
type ImageAspectRatio string

const (
	ImageAspectRatio1_1  ImageAspectRatio = "1:1"
	ImageAspectRatio2_3  ImageAspectRatio = "2:3"
	ImageAspectRatio3_2  ImageAspectRatio = "3:2"
	ImageAspectRatio3_4  ImageAspectRatio = "3:4"
	ImageAspectRatio4_3  ImageAspectRatio = "4:3"
	ImageAspectRatio4_5  ImageAspectRatio = "4:5"
	ImageAspectRatio5_4  ImageAspectRatio = "5:4"
	ImageAspectRatio9_16 ImageAspectRatio = "9:16"
	ImageAspectRatio16_9 ImageAspectRatio = "16:9"
	ImageAspectRatio21_9 ImageAspectRatio = "21:9"
)

var ImageAspectRatios = map[string]ImageAspectRatio{
	"1:1":  ImageAspectRatio1_1,
	"2:3":  ImageAspectRatio2_3,
	"3:2":  ImageAspectRatio3_2,
	"3:4":  ImageAspectRatio3_4,
	"4:3":  ImageAspectRatio4_3,
	"4:5":  ImageAspectRatio4_5,
	"5:4":  ImageAspectRatio5_4,
	"9:16": ImageAspectRatio9_16,
	"16:9": ImageAspectRatio16_9,
	"21:9": ImageAspectRatio21_9,
}

// ImageSize enumerates supported Gemini image sizes.
type ImageSize string

const (
	ImageSize1K ImageSize = "1K"
	ImageSize2K ImageSize = "2K"
	ImageSize4K ImageSize = "4K"
)

var ImageSizes = map[string]ImageSize{
	"1K": ImageSize1K,
	"2K": ImageSize2K,
	"4K": ImageSize4K,
}

// ImageOption mutates Gemini image generation settings.
type ImageOption interface {
	grail.ProviderOption
	apply(*imageConfig)
}

type imageConfig struct {
	aspectRatio ImageAspectRatio
	size        ImageSize
}

type imageOptionFunc struct {
	desc string
	fn   func(*imageConfig)
}

func (o imageOptionFunc) Description() string { return o.desc }
func (o imageOptionFunc) apply(cfg *imageConfig) {
	if o.fn != nil {
		o.fn(cfg)
	}
}

// WithImageAspectRatio sets the Gemini image aspect ratio.
func WithImageAspectRatio(ratio ImageAspectRatio) ImageOption {
	return imageOptionFunc{
		desc: fmt.Sprintf("gemini image aspect ratio %s", ratio),
		fn: func(c *imageConfig) {
			if ratio != "" {
				c.aspectRatio = ratio
			}
		},
	}
}

// WithImageSize sets the Gemini image size.
func WithImageSize(size ImageSize) ImageOption {
	return imageOptionFunc{
		desc: fmt.Sprintf("gemini image size %s", size),
		fn: func(c *imageConfig) {
			if size != "" {
				c.size = size
			}
		},
	}
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

	clientConfig := &genai.ClientConfig{
		Backend: genai.BackendGeminiAPI,
	}
	if cfg.apiKey != "" {
		clientConfig.APIKey = cfg.apiKey
	}

	client, err := genai.NewClient(ctx, clientConfig)
	if err != nil {
		return nil, fmt.Errorf("new gemini client: %w", err)
	}

	return &Provider{
		client:     client,
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

	config := &genai.GenerateContentConfig{}
	applyTextOptions(config, req.Options)

	contents := []*genai.Content{
		genai.NewContentFromParts(parts, genai.RoleUser),
	}

	resp, err := c.client.Models.GenerateContent(ctx, modelName, contents, config)
	if err != nil {
		return grail.TextResult{}, fmt.Errorf("generate text: %w", err)
	}

	return grail.TextResult{
		Text: resp.Text(),
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

	config := &genai.GenerateContentConfig{}
	applyImageOptions(config, req.Options, req.ProviderOptions)

	contents := []*genai.Content{
		genai.NewContentFromParts(parts, genai.RoleUser),
	}

	resp, err := c.client.Models.GenerateContent(ctx, modelName, contents, config)
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

func toGenAIParts(input []grail.Part) ([]*genai.Part, error) {
	out := make([]*genai.Part, 0, len(input))
	for i, p := range input {
		switch v := p.(type) {
		case grail.TextPart:
			out = append(out, genai.NewPartFromText(v.Text))
		case grail.ImagePart:
			if len(v.Data) == 0 {
				return nil, fmt.Errorf("part %d: image data is empty", i)
			}
			mime := v.MIME
			if mime == "" {
				mime = "image/png"
			}
			out = append(out, genai.NewPartFromBytes(v.Data, mime))
		case grail.PDFPart:
			if len(v.Data) == 0 {
				return nil, fmt.Errorf("part %d: PDF data is empty", i)
			}
			mime := v.MIME
			if mime == "" {
				mime = "application/pdf"
			}
			out = append(out, genai.NewPartFromBytes(v.Data, mime))
		default:
			return nil, fmt.Errorf("part %d: unknown part type %T", i, p)
		}
	}
	return out, nil
}

func applyTextOptions(config *genai.GenerateContentConfig, opts grail.TextOptions) {
	if opts.SystemPrompt != "" {
		config.SystemInstruction = &genai.Content{
			Parts: []*genai.Part{
				{Text: opts.SystemPrompt},
			},
		}
	}
	if opts.Temperature != nil {
		config.Temperature = genai.Ptr(*opts.Temperature)
	}
	if opts.TopP != nil {
		config.TopP = genai.Ptr(*opts.TopP)
	}
	if opts.MaxTokens != nil {
		config.MaxOutputTokens = int32(*opts.MaxTokens)
	}
}

func applyImageOptions(config *genai.GenerateContentConfig, opts grail.ImageOptions, providerOpts []grail.ProviderOption) {
	if opts.SystemPrompt != "" {
		config.SystemInstruction = &genai.Content{
			Parts: []*genai.Part{
				{Text: opts.SystemPrompt},
			},
		}
	}

	cfg := imageConfig{}
	for _, opt := range providerOpts {
		if fn, ok := opt.(ImageOption); ok && fn != nil {
			fn.apply(&cfg)
		}
	}

	// Apply image config if aspect ratio or size is set
	if cfg.aspectRatio != "" || cfg.size != "" {
		config.ImageConfig = &genai.ImageConfig{}
		if cfg.aspectRatio != "" {
			config.ImageConfig.AspectRatio = string(cfg.aspectRatio)
		}
		if cfg.size != "" {
			config.ImageConfig.ImageSize = string(cfg.size)
		}
	}
}

func extractImages(resp *genai.GenerateContentResponse) []grail.ImageOutput {
	var out []grail.ImageOutput
	for _, cand := range resp.Candidates {
		if cand == nil || cand.Content == nil {
			continue
		}
		for _, part := range cand.Content.Parts {
			if part.InlineData != nil {
				out = append(out, grail.ImageOutput{
					Data: part.InlineData.Data,
					MIME: part.InlineData.MIMEType,
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
		case grail.PDFPart:
			out = append(out, map[string]any{
				"type": "pdf",
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
