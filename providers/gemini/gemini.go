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
//	res, err := client.Generate(ctx, grail.Request{
//		Inputs: []grail.Input{grail.InputText("Hello, world!")},
//		Output: grail.OutputText(),
//	})
//
// The provider automatically uses the GEMINI_API_KEY environment variable
// if no API key is explicitly provided via WithAPIKey or WithAPIKeyFromEnv.
//
// Default models:
//   - Text: gemini-3-flash-preview
//   - Image: gemini-2.5-flash-image
package gemini

import (
	"context"
	"encoding/json"
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
	DefaultTextModelName = "gemini-3-pro-preview"
	// DefaultImageModelName is the Gemini image model used when no override is provided.
	DefaultImageModelName = "gemini-3-pro-image-preview"
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

	// Model catalog slots
	bestTextModel  grail.Model
	fastTextModel  grail.Model
	bestImageModel grail.Model
	fastImageModel grail.Model
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
	"4:5":  ImageAspectRatio5_4,
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

// TextOptions provides Gemini-specific text generation options.
type TextOptions struct {
	Model        string
	MaxTokens    *int32
	Temperature  *float32
	TopP         *float32
	SystemPrompt string
}

func (TextOptions) ApplyProviderOption() {}

// ImageOptions provides Gemini-specific image generation options.
type ImageOptions struct {
	Model        string
	SystemPrompt string
}

func (ImageOptions) ApplyProviderOption() {}

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
	fn func(*imageConfig)
}

func (o imageOptionFunc) ApplyProviderOption() {}
func (o imageOptionFunc) apply(cfg *imageConfig) {
	if o.fn != nil {
		o.fn(cfg)
	}
}

// WithImageAspectRatio sets the Gemini image aspect ratio.
func WithImageAspectRatio(ratio ImageAspectRatio) ImageOption {
	return imageOptionFunc{
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
		// Initialize model catalog with defaults
		bestTextModel:  Gemini3Pro,
		fastTextModel:  Gemini3Flash,
		bestImageModel: Gemini3ProImage,
		fastImageModel: Gemini25FlashImage,
	}, nil
}

// SetLogger allows the client to inject a logger.
func (c *Provider) SetLogger(l *slog.Logger) {
	if l != nil {
		c.log = l
	}
}

// Name returns the provider name.
func (c *Provider) Name() string {
	return "gemini"
}

// ModelCatalog implementation

// SetBestTextModel sets the model to use for best-quality text generation.
func (c *Provider) SetBestTextModel(model grail.Model) { c.bestTextModel = model }

// SetFastTextModel sets the model to use for fast text generation.
func (c *Provider) SetFastTextModel(model grail.Model) { c.fastTextModel = model }

// SetBestImageModel sets the model to use for best-quality image generation.
func (c *Provider) SetBestImageModel(model grail.Model) { c.bestImageModel = model }

// SetFastImageModel sets the model to use for fast image generation.
func (c *Provider) SetFastImageModel(model grail.Model) { c.fastImageModel = model }

// BestTextModel returns the model used for best-quality text generation.
func (c *Provider) BestTextModel() grail.Model { return c.bestTextModel }

// FastTextModel returns the model used for fast text generation.
func (c *Provider) FastTextModel() grail.Model { return c.fastTextModel }

// BestImageModel returns the model used for best-quality image generation.
func (c *Provider) BestImageModel() grail.Model { return c.bestImageModel }

// FastImageModel returns the model used for fast image generation.
func (c *Provider) FastImageModel() grail.Model { return c.fastImageModel }

// AllModels returns all configured models.
func (c *Provider) AllModels() []grail.Model {
	return []grail.Model{
		c.bestTextModel,
		c.fastTextModel,
		c.bestImageModel,
		c.fastImageModel,
		// Additional models not set as best/fast
		Gemini25Flash,
		Gemini25FlashLite,
	}
}

// ListModels returns all available Gemini models and their capabilities.
func (c *Provider) ListModels(ctx context.Context) ([]grail.Model, error) {
	return c.AllModels(), nil
}

// ResolveModel resolves a role+tier to a model name.
func (c *Provider) ResolveModel(role grail.ModelRole, tier grail.ModelTier) (string, error) {
	switch {
	case role == grail.ModelRoleText && tier == grail.ModelTierBest:
		return c.bestTextModel.Name, nil
	case role == grail.ModelRoleText && tier == grail.ModelTierFast:
		return c.fastTextModel.Name, nil
	case role == grail.ModelRoleImage && tier == grail.ModelTierBest:
		return c.bestImageModel.Name, nil
	case role == grail.ModelRoleImage && tier == grail.ModelTierFast:
		return c.fastImageModel.Name, nil
	default:
		return "", fmt.Errorf("gemini: no %s model with tier %s", role, tier)
	}
}

// DoGenerate implements the ProviderExecutor interface.
func (c *Provider) DoGenerate(ctx context.Context, req grail.Request) (grail.Response, error) {
	// Convert inputs to Gemini format
	parts, err := c.toGenAIParts(req.Inputs)
	if err != nil {
		return grail.Response{}, grail.NewGrailError(grail.InvalidArgument, fmt.Sprintf("failed to convert inputs: %v", err)).WithCause(err).WithProviderName("gemini")
	}

	// Determine output type and route accordingly
	if grail.IsTextOutput(req.Output) {
		return c.generateText(ctx, req, parts)
	}
	if spec, isImage := grail.GetImageSpec(req.Output); isImage {
		return c.generateImage(ctx, req, parts, spec)
	}
	if schema, strict, isJSON := grail.GetJSONOutput(req.Output); isJSON {
		return c.generateJSON(ctx, req, parts, schema, strict)
	}
	return grail.Response{}, grail.NewGrailError(grail.Unsupported, fmt.Sprintf("unsupported output type: %T", req.Output)).WithProviderName("gemini")
}

func (c *Provider) generateText(ctx context.Context, req grail.Request, parts []*genai.Part) (grail.Response, error) {
	// Extract text options from provider options
	var textOpts TextOptions
	modelName := c.textModel
	// Request.Model takes precedence over provider default and ProviderOptions
	if req.Model != "" {
		modelName = req.Model
	} else {
		// Fall back to ProviderOptions if Request.Model not set
		for _, opt := range req.ProviderOptions {
			if to, ok := opt.(TextOptions); ok {
				textOpts = to
				if to.Model != "" {
					modelName = to.Model
				}
			}
		}
	}

	if c.log != nil {
		c.log.Debug("generate text request", slog.String("model", modelName))
	}

	config := &genai.GenerateContentConfig{}
	c.applyTextOptions(config, textOpts)

	contents := []*genai.Content{
		genai.NewContentFromParts(parts, genai.RoleUser),
	}

	resp, err := c.client.Models.GenerateContent(ctx, modelName, contents, config)
	if err != nil {
		return grail.Response{}, grail.NewGrailError(grail.Internal, fmt.Sprintf("generate text failed: %v", err)).WithCause(err).WithProviderName("gemini").WithRetryable(isRetryableError(err))
	}

	text := resp.Text()
	usage := extractUsage(resp)

	if c.log != nil {
		c.log.Debug("generate text response", slog.Any("usage", usage))
	}

	return grail.Response{
		Outputs: []grail.OutputPart{
			grail.NewTextOutputPart(text),
		},
		Usage: usage,
		Provider: grail.ProviderInfo{
			Name:  "gemini",
			Route: "generate_content",
			Models: []grail.ModelUse{
				{Role: "language", Name: modelName},
			},
		},
		RequestID: "",
		Warnings:  extractWarnings(resp),
	}, nil
}

func (c *Provider) generateImage(ctx context.Context, req grail.Request, parts []*genai.Part, spec grail.ImageSpec) (grail.Response, error) {
	// Extract image options from provider options
	var imageOpts ImageOptions
	modelName := c.imageModel
	cfg := imageConfig{}

	// Request.Model takes precedence for the image model
	if req.Model != "" {
		modelName = req.Model
	} else {
		// Fall back to ProviderOptions if Request.Model not set
		for _, opt := range req.ProviderOptions {
			if io, ok := opt.(ImageOptions); ok {
				imageOpts = io
				if io.Model != "" {
					modelName = io.Model
				}
			}
			if imgOpt, ok := opt.(ImageOption); ok {
				imgOpt.apply(&cfg)
			}
		}
	}

	if c.log != nil {
		c.log.Debug("generate image request", slog.String("model", modelName))
	}

	config := &genai.GenerateContentConfig{}
	c.applyImageOptions(config, imageOpts, &cfg)

	contents := []*genai.Content{
		genai.NewContentFromParts(parts, genai.RoleUser),
	}

	resp, err := c.client.Models.GenerateContent(ctx, modelName, contents, config)
	if err != nil {
		return grail.Response{}, grail.NewGrailError(grail.Internal, fmt.Sprintf("generate image failed: %v", err)).WithCause(err).WithProviderName("gemini").WithRetryable(isRetryableError(err))
	}

	images := extractImages(resp)
	usage := extractUsage(resp)

	if c.log != nil {
		c.log.Debug("generate image response", slog.Int("images", len(images)), slog.Any("usage", usage))
	}

	outputParts := make([]grail.OutputPart, 0, len(images))
	for _, img := range images {
		outputParts = append(outputParts, grail.NewImageOutputPart(img.Data, img.MIME, ""))
	}

	return grail.Response{
		Outputs: outputParts,
		Usage:   usage,
		Provider: grail.ProviderInfo{
			Name:  "gemini",
			Route: "generate_content",
			Models: []grail.ModelUse{
				{Role: "language", Name: modelName},
				{Role: "image_generation", Name: modelName},
			},
		},
		RequestID: "",
		Warnings:  extractWarnings(resp),
	}, nil
}

func (c *Provider) generateJSON(ctx context.Context, req grail.Request, parts []*genai.Part, schema any, strict bool) (grail.Response, error) {
	// Extract text options from provider options
	var textOpts TextOptions
	modelName := c.textModel
	// Request.Model takes precedence over provider default and ProviderOptions
	if req.Model != "" {
		modelName = req.Model
	} else {
		// Fall back to ProviderOptions if Request.Model not set
		for _, opt := range req.ProviderOptions {
			if to, ok := opt.(TextOptions); ok {
				textOpts = to
				if to.Model != "" {
					modelName = to.Model
				}
			}
		}
	}

	if c.log != nil {
		c.log.Debug("generate JSON request", slog.String("model", modelName))
	}

	config := &genai.GenerateContentConfig{}
	c.applyTextOptions(config, textOpts)
	// Note: Gemini may support JSON mode via response_mime_type or similar
	// For now, we'll generate text and validate as JSON

	contents := []*genai.Content{
		genai.NewContentFromParts(parts, genai.RoleUser),
	}

	resp, err := c.client.Models.GenerateContent(ctx, modelName, contents, config)
	if err != nil {
		return grail.Response{}, grail.NewGrailError(grail.Internal, fmt.Sprintf("generate JSON failed: %v", err)).WithCause(err).WithProviderName("gemini").WithRetryable(isRetryableError(err))
	}

	text := resp.Text()
	usage := extractUsage(resp)

	// Validate JSON if strict mode
	jsonBytes := []byte(text)
	if strict {
		var test any
		if err := json.Unmarshal(jsonBytes, &test); err != nil {
			return grail.Response{}, grail.NewGrailError(grail.OutputInvalid, fmt.Sprintf("invalid JSON output: %v", err)).WithProviderName("gemini")
		}
	}

	if c.log != nil {
		c.log.Debug("generate JSON response", slog.Any("usage", usage))
	}

	return grail.Response{
		Outputs: []grail.OutputPart{
			grail.NewJSONOutputPart(jsonBytes),
		},
		Usage: usage,
		Provider: grail.ProviderInfo{
			Name:  "gemini",
			Route: "generate_content",
			Models: []grail.ModelUse{
				{Role: "language", Name: modelName},
			},
		},
		RequestID: "",
		Warnings:  extractWarnings(resp),
	}, nil
}

// toGenAIParts converts grail.Inputs to Gemini API format.
func (c *Provider) toGenAIParts(inputs []grail.Input) ([]*genai.Part, error) {
	out := make([]*genai.Part, 0, len(inputs))
	for i, input := range inputs {
		text, isText := grail.AsTextInput(input)
		if isText {
			out = append(out, genai.NewPartFromText(text))
			continue
		}

		data, mime, _, isFile := grail.AsFileInput(input)
		if isFile {
			if len(data) == 0 {
				return nil, fmt.Errorf("input %d: file data is empty", i)
			}
			if mime == "" {
				// Try to detect MIME from data (e.g., from InputImage with empty MIME)
				mime = grail.SniffImageMIME(data)
				if mime == "" {
					mime = "application/octet-stream"
				}
			}
			out = append(out, genai.NewPartFromBytes(data, mime))
			continue
		}

		// FileReaderInput - read into memory for now
		// TODO: support streaming if Gemini API supports it
		return nil, fmt.Errorf("input %d: FileReaderInput not yet supported", i)
	}
	return out, nil
}

func (c *Provider) applyTextOptions(config *genai.GenerateContentConfig, opts TextOptions) {
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

func (c *Provider) applyImageOptions(config *genai.GenerateContentConfig, opts ImageOptions, imgCfg *imageConfig) {
	if opts.SystemPrompt != "" {
		config.SystemInstruction = &genai.Content{
			Parts: []*genai.Part{
				{Text: opts.SystemPrompt},
			},
		}
	}

	// Apply image config if aspect ratio or size is set
	if imgCfg.aspectRatio != "" || imgCfg.size != "" {
		config.ImageConfig = &genai.ImageConfig{}
		if imgCfg.aspectRatio != "" {
			config.ImageConfig.AspectRatio = string(imgCfg.aspectRatio)
		}
		if imgCfg.size != "" {
			config.ImageConfig.ImageSize = string(imgCfg.size)
		}
	}
}

func extractImages(resp *genai.GenerateContentResponse) []imageData {
	var out []imageData
	for _, cand := range resp.Candidates {
		if cand == nil || cand.Content == nil {
			continue
		}
		for _, part := range cand.Content.Parts {
			if part.InlineData != nil {
				out = append(out, imageData{
					Data: part.InlineData.Data,
					MIME: part.InlineData.MIMEType,
				})
			}
		}
	}
	return out
}

type imageData struct {
	Data []byte
	MIME string
}

func extractUsage(resp *genai.GenerateContentResponse) grail.Usage {
	if resp == nil || resp.UsageMetadata == nil {
		return grail.Usage{}
	}
	return grail.Usage{
		InputTokens:  int(resp.UsageMetadata.PromptTokenCount),
		OutputTokens: int(resp.UsageMetadata.CandidatesTokenCount),
		TotalTokens:  int(resp.UsageMetadata.TotalTokenCount),
	}
}

func extractWarnings(resp *genai.GenerateContentResponse) []grail.Warning {
	// Gemini SDK may not have warnings field in all versions
	// Return empty slice for now
	return nil
}

func isRetryableError(err error) bool {
	// Gemini SDK errors that are retryable
	errStr := err.Error()
	return strings.Contains(errStr, "rate_limit") ||
		strings.Contains(errStr, "timeout") ||
		strings.Contains(errStr, "temporary") ||
		strings.Contains(errStr, "503") ||
		strings.Contains(errStr, "429")
}
