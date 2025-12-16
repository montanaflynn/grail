// Package openai provides an OpenAI implementation of the grail.Provider interface.
// It uses the OpenAI Responses API for both text and image generation.
//
// Example usage:
//
//	provider, err := openai.New()
//	if err != nil {
//		log.Fatal(err)
//	}
//	client := grail.NewClient(provider)
//	res, err := client.Generate(ctx, grail.Request{
//		Inputs: []grail.Input{grail.InputText("Hello, world!")},
//		Output: grail.OutputText(),
//	})
//
// The provider automatically uses the OPENAI_API_KEY environment variable
// if no API key is explicitly provided via WithAPIKey or WithAPIKeyFromEnv.
//
// Default models:
//   - Text: gpt-5.2
//   - Image: gpt-image-1.5
//
// Available image models:
//   - gpt-image-1.5 (default)
//   - gpt-image-1
//   - gpt-image-1-mini
package openai

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"strings"

	"github.com/montanaflynn/grail"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	"github.com/openai/openai-go/v3/packages/param"
	"github.com/openai/openai-go/v3/responses"
	"github.com/openai/openai-go/v3/shared"
	"github.com/openai/openai-go/v3/shared/constant"
)

const (
	// DefaultTextModelName is the OpenAI text model used when no override is provided.
	DefaultTextModelName = shared.ChatModelGPT5_2
	// DefaultImageModelName is the OpenAI image model used when no override is provided.
	DefaultImageModelName = openai.ImageModelGPTImage1_5
)

var (
	// ErrAPIKeyRequired is returned when no API key is configured.
	ErrAPIKeyRequired = errors.New("openai: API key required (set OPENAI_API_KEY or use WithAPIKey/WithAPIKeyFromEnv)")
)

// Option configures the OpenAI provider.
type Option func(*settings)

type settings struct {
	apiKey     string
	apiKeySet  bool
	textModel  string
	imageModel string
	logger     *slog.Logger
	imgFormat  string
}

// WithAPIKey sets the API key explicitly.
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
		if v := strings.TrimSpace(os.Getenv(env)); v != "" {
			s.apiKey = v
		}
	}
}

// WithTextModel overrides the default text model (default: gpt-5.2).
func WithTextModel(model string) Option {
	return func(s *settings) { s.textModel = model }
}

// WithImageModel overrides the default image model (default: gpt-image-1.5).
// Available models: gpt-image-1.5, gpt-image-1, gpt-image-1-mini
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

// Provider is an OpenAI-backed implementation of grail.Provider.
type Provider struct {
	client     openai.Client
	textModel  string
	imageModel string
	log        *slog.Logger
	imgFormat  string
}

// ImageFormat enumerates supported OpenAI image output formats.
type ImageFormat string

const (
	ImageFormatPNG  ImageFormat = "png"
	ImageFormatJPEG ImageFormat = "jpeg"
	ImageFormatJPG  ImageFormat = "jpg"
	ImageFormatWEBP ImageFormat = "webp"
)

var ImageFormats = map[string]ImageFormat{
	"png":  ImageFormatPNG,
	"jpeg": ImageFormatJPEG,
	"jpg":  ImageFormatJPG,
	"webp": ImageFormatWEBP,
}

// ImageBackground enumerates supported OpenAI image backgrounds.
type ImageBackground string

const (
	ImageBackgroundAuto        ImageBackground = "auto"
	ImageBackgroundTransparent ImageBackground = "transparent"
	ImageBackgroundOpaque      ImageBackground = "opaque"
)

var ImageBackgrounds = map[string]ImageBackground{
	"auto":        ImageBackgroundAuto,
	"transparent": ImageBackgroundTransparent,
	"opaque":      ImageBackgroundOpaque,
}

// ImageSize enumerates supported OpenAI image sizes.
type ImageSize string

const (
	ImageSizeAuto      ImageSize = "auto"
	ImageSize1024x1024 ImageSize = "1024x1024"
	ImageSize1536x1024 ImageSize = "1536x1024"
	ImageSize1024x1536 ImageSize = "1024x1536"
	ImageSize256x256   ImageSize = "256x256"
	ImageSize512x512   ImageSize = "512x512"
	ImageSize1792x1024 ImageSize = "1792x1024"
	ImageSize1024x1792 ImageSize = "1024x1792"
)

var ImageSizes = map[string]ImageSize{
	"auto":      ImageSizeAuto,
	"1024x1024": ImageSize1024x1024,
	"1536x1024": ImageSize1536x1024,
	"1024x1536": ImageSize1024x1536,
	"256x256":   ImageSize256x256,
	"512x512":   ImageSize512x512,
	"1792x1024": ImageSize1792x1024,
	"1024x1792": ImageSize1024x1792,
}

// ImageModeration enumerates supported OpenAI image moderation levels.
type ImageModeration string

const (
	ImageModerationAuto ImageModeration = "auto"
	ImageModerationLow  ImageModeration = "low"
)

var ImageModerations = map[string]ImageModeration{
	"auto": ImageModerationAuto,
	"low":  ImageModerationLow,
}

// TextOptions provides OpenAI-specific text generation options.
type TextOptions struct {
	Model        string
	MaxTokens    *int32
	Temperature  *float32
	TopP         *float32
	SystemPrompt string
}

func (TextOptions) ApplyProviderOption() {}

// ImageOptions provides OpenAI-specific image generation options.
type ImageOptions struct {
	Model        string
	SystemPrompt string
}

func (ImageOptions) ApplyProviderOption() {}

// ImageOption mutates OpenAI image generation settings.
type ImageOption interface {
	grail.ProviderOption
	apply(*imageConfig)
}

type imageConfig struct {
	format            ImageFormat
	background        ImageBackground
	size              ImageSize
	moderation        ImageModeration
	outputCompression *int64
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

// WithImageFormat sets the OpenAI image output format.
func WithImageFormat(f ImageFormat) ImageOption {
	return imageOptionFunc{
		fn: func(c *imageConfig) {
			if f != "" {
				c.format = f
			}
		},
	}
}

// WithImageBackground sets the OpenAI image background mode.
func WithImageBackground(b ImageBackground) ImageOption {
	return imageOptionFunc{
		fn: func(c *imageConfig) {
			if b != "" {
				c.background = b
			}
		},
	}
}

// WithImageSize sets the OpenAI image size.
func WithImageSize(size ImageSize) ImageOption {
	return imageOptionFunc{
		fn: func(c *imageConfig) {
			if size != "" {
				c.size = size
			}
		},
	}
}

// WithImageModeration sets the OpenAI image moderation level.
func WithImageModeration(moderation ImageModeration) ImageOption {
	return imageOptionFunc{
		fn: func(c *imageConfig) {
			if moderation != "" {
				c.moderation = moderation
			}
		},
	}
}

// WithImageOutputCompression sets the OpenAI image output compression (0-100% for JPEG/WebP).
func WithImageOutputCompression(compression int) ImageOption {
	return imageOptionFunc{
		fn: func(c *imageConfig) {
			if compression >= 0 && compression <= 100 {
				comp := int64(compression)
				c.outputCompression = &comp
			}
		},
	}
}

// New constructs an OpenAI provider using functional options.
func New(opts ...Option) (*Provider, error) {
	cfg := settings{
		textModel:  DefaultTextModelName,
		imageModel: DefaultImageModelName,
		logger:     slog.Default(),
		imgFormat:  "png",
	}
	for _, opt := range opts {
		opt(&cfg)
	}

	switch {
	case cfg.apiKeySet && cfg.apiKey == "":
		return nil, ErrAPIKeyRequired
	case !cfg.apiKeySet && cfg.apiKey == "":
		cfg.apiKey = strings.TrimSpace(os.Getenv("OPENAI_API_KEY"))
		if cfg.apiKey == "" {
			return nil, ErrAPIKeyRequired
		}
	}

	clientOpts := []option.RequestOption{}
	if cfg.apiKey != "" {
		clientOpts = append(clientOpts, option.WithAPIKey(cfg.apiKey))
	}

	cl := openai.NewClient(clientOpts...)

	return &Provider{
		client:     cl,
		textModel:  cfg.textModel,
		imageModel: cfg.imageModel,
		log:        cfg.logger,
		imgFormat:  cfg.imgFormat,
	}, nil
}

// SetLogger allows the client to inject a logger.
func (p *Provider) SetLogger(l *slog.Logger) {
	if l != nil {
		p.log = l
	}
}

// Name returns the provider name.
func (p *Provider) Name() string {
	return "openai"
}

// DoGenerate implements the ProviderExecutor interface.
func (p *Provider) DoGenerate(ctx context.Context, req grail.Request) (grail.Response, error) {
	// Convert inputs to OpenAI format
	item, err := p.toResponseInput(req.Inputs)
	if err != nil {
		return grail.Response{}, grail.NewGrailError(grail.InvalidArgument, fmt.Sprintf("failed to convert inputs: %v", err)).WithCause(err).WithProviderName("openai")
	}

	// Determine output type and route accordingly
	if grail.IsTextOutput(req.Output) {
		return p.generateText(ctx, req, item)
	}
	if spec, isImage := grail.GetImageSpec(req.Output); isImage {
		return p.generateImage(ctx, req, item, spec)
	}
	if schema, strict, isJSON := grail.GetJSONOutput(req.Output); isJSON {
		return p.generateJSON(ctx, req, item, schema, strict)
	}
	return grail.Response{}, grail.NewGrailError(grail.Unsupported, fmt.Sprintf("unsupported output type: %T", req.Output)).WithProviderName("openai")
}

func (p *Provider) generateText(ctx context.Context, req grail.Request, item responses.ResponseInputItemUnionParam) (grail.Response, error) {
	// Extract text options from provider options
	var textOpts TextOptions
	model := p.textModel
	for _, opt := range req.ProviderOptions {
		if to, ok := opt.(TextOptions); ok {
			textOpts = to
			if to.Model != "" {
				model = to.Model
			}
		}
	}

	if p.log != nil {
		p.log.Debug("openai generate text request", slog.String("model", model))
	}

	params := responses.ResponseNewParams{
		Model: shared.ChatModel(model),
		Input: responses.ResponseNewParamsInputUnion{
			OfInputItemList: responses.ResponseInputParam{item},
		},
	}

	if textOpts.SystemPrompt != "" {
		params.Instructions = param.NewOpt(textOpts.SystemPrompt)
	}
	if textOpts.MaxTokens != nil {
		params.MaxOutputTokens = openai.Int(int64(*textOpts.MaxTokens))
	}
	if textOpts.Temperature != nil {
		params.Temperature = openai.Float(float64(*textOpts.Temperature))
	}
	if textOpts.TopP != nil {
		params.TopP = openai.Float(float64(*textOpts.TopP))
	}

	resp, err := p.client.Responses.New(ctx, params)
	if err != nil {
		ge := grail.NewGrailError(grail.Internal, fmt.Sprintf("openai generate text failed: %v", err)).WithCause(err).WithProviderName("openai").WithRetryable(isRetryableError(err))
		return grail.Response{}, ge
	}

	text := resp.OutputText()
	usage := extractUsage(resp)

	if p.log != nil {
		p.log.Debug("openai generate text response", slog.Any("usage", usage))
	}

	return grail.Response{
		Outputs: []grail.OutputPart{
			grail.NewTextOutputPart(text),
		},
		Usage: usage,
		Provider: grail.ProviderInfo{
			Name:  "openai",
			Route: "responses",
			Models: []grail.ModelUse{
				{Role: "language", Name: model},
			},
		},
		RequestID: resp.ID,
		Warnings:  extractWarnings(resp),
	}, nil
}

func (p *Provider) generateImage(ctx context.Context, req grail.Request, item responses.ResponseInputItemUnionParam, spec grail.ImageSpec) (grail.Response, error) {
	// Extract image options from provider options
	var imageOpts ImageOptions
	model := p.textModel
	cfg := imageConfig{
		format:     ImageFormat(p.imgFormat),
		background: ImageBackgroundAuto,
		size:       ImageSizeAuto,
		moderation: ImageModerationAuto,
	}

	for _, opt := range req.ProviderOptions {
		if io, ok := opt.(ImageOptions); ok {
			imageOpts = io
			if io.Model != "" {
				model = io.Model
			}
		}
		if imgOpt, ok := opt.(ImageOption); ok {
			imgOpt.apply(&cfg)
		}
	}

	size := string(cfg.size)
	if size == "" {
		size = "auto"
	}
	moderation := string(cfg.moderation)
	if moderation == "" {
		moderation = "auto"
	}

	imageGenParam := &responses.ToolImageGenerationParam{
		Type:          "image_generation",
		Model:         p.imageModel,
		OutputFormat:  string(cfg.format),
		Background:    string(cfg.background),
		Moderation:    moderation,
		Quality:       "auto",
		Size:          size,
		InputFidelity: "",
		PartialImages: param.NewOpt(int64(0)),
	}

	if cfg.outputCompression != nil {
		imageGenParam.OutputCompression = param.NewOpt(*cfg.outputCompression)
	} else {
		imageGenParam.OutputCompression = param.NewOpt(int64(100))
	}

	params := responses.ResponseNewParams{
		Model: shared.ChatModel(model),
		Input: responses.ResponseNewParamsInputUnion{
			OfInputItemList: responses.ResponseInputParam{item},
		},
		Tools: []responses.ToolUnionParam{
			{
				OfImageGeneration: imageGenParam,
			},
		},
	}

	if imageOpts.SystemPrompt != "" {
		params.Instructions = param.NewOpt(imageOpts.SystemPrompt)
	}

	if p.log != nil {
		// Log detailed request information
		logFields := []any{
			slog.String("language_model", model),
			slog.String("image_model", p.imageModel),
			slog.String("output_format", string(cfg.format)),
			slog.String("background", string(cfg.background)),
			slog.String("size", size),
			slog.String("moderation", moderation),
		}
		if cfg.outputCompression != nil {
			logFields = append(logFields, slog.Int64("compression", *cfg.outputCompression))
		} else {
			logFields = append(logFields, slog.Int("compression", 100))
		}
		if imageOpts.SystemPrompt != "" {
			logFields = append(logFields, slog.String("system_prompt", imageOpts.SystemPrompt))
		}
		// Try to marshal the full params for complete visibility
		if paramsJSON, err := json.MarshalIndent(params, "", "  "); err == nil {
			p.log.Debug("openai generate image request (full params)", append(logFields, slog.String("params", string(paramsJSON)))...)
		} else {
			p.log.Debug("openai generate image request", logFields...)
		}
	}

	resp, err := p.client.Responses.New(ctx, params)
	if err != nil {
		ge := grail.NewGrailError(grail.Internal, fmt.Sprintf("openai generate image failed: %v", err)).WithCause(err).WithProviderName("openai").WithRetryable(isRetryableError(err))
		return grail.Response{}, ge
	}

	images := extractImagesFromResponse(resp, string(cfg.format))
	usage := extractUsage(resp)

	if p.log != nil {
		p.log.Debug("openai generate image response", slog.Int("images", len(images)), slog.Any("usage", usage))
	}

	outputParts := make([]grail.OutputPart, 0, len(images))
	for _, img := range images {
		outputParts = append(outputParts, grail.NewImageOutputPart(img.Data, img.MIME, ""))
	}

	return grail.Response{
		Outputs: outputParts,
		Usage:   usage,
		Provider: grail.ProviderInfo{
			Name:  "openai",
			Route: "responses",
			Models: []grail.ModelUse{
				{Role: "language", Name: model},
				{Role: "image_generation", Name: p.imageModel},
			},
		},
		RequestID: resp.ID,
		Warnings:  extractWarnings(resp),
	}, nil
}

func (p *Provider) generateJSON(ctx context.Context, req grail.Request, item responses.ResponseInputItemUnionParam, schema any, strict bool) (grail.Response, error) {
	// JSON output is similar to text, but with response format
	var textOpts TextOptions
	model := p.textModel
	for _, opt := range req.ProviderOptions {
		if to, ok := opt.(TextOptions); ok {
			textOpts = to
			if to.Model != "" {
				model = to.Model
			}
		}
	}

	if p.log != nil {
		p.log.Debug("openai generate JSON request", slog.String("model", model))
	}

	params := responses.ResponseNewParams{
		Model: shared.ChatModel(model),
		Input: responses.ResponseNewParamsInputUnion{
			OfInputItemList: responses.ResponseInputParam{item},
		},
		// Note: JSON mode may not be available in all SDK versions
		// If ResponseFormat is not available, we'll validate JSON manually
	}

	if textOpts.SystemPrompt != "" {
		params.Instructions = param.NewOpt(textOpts.SystemPrompt)
	}
	if textOpts.MaxTokens != nil {
		params.MaxOutputTokens = openai.Int(int64(*textOpts.MaxTokens))
	}
	if textOpts.Temperature != nil {
		params.Temperature = openai.Float(float64(*textOpts.Temperature))
	}
	if textOpts.TopP != nil {
		params.TopP = openai.Float(float64(*textOpts.TopP))
	}

	resp, err := p.client.Responses.New(ctx, params)
	if err != nil {
		ge := grail.NewGrailError(grail.Internal, fmt.Sprintf("openai generate JSON failed: %v", err)).WithCause(err).WithProviderName("openai").WithRetryable(isRetryableError(err))
		return grail.Response{}, ge
	}

	text := resp.OutputText()
	usage := extractUsage(resp)

	// Validate JSON if strict mode
	jsonBytes := []byte(text)
	if strict {
		var test any
		if err := json.Unmarshal(jsonBytes, &test); err != nil {
			return grail.Response{}, grail.NewGrailError(grail.OutputInvalid, fmt.Sprintf("invalid JSON output: %v", err)).WithProviderName("openai")
		}
	}

	if p.log != nil {
		p.log.Debug("openai generate JSON response", slog.Any("usage", usage))
	}

	return grail.Response{
		Outputs: []grail.OutputPart{
			grail.NewJSONOutputPart(jsonBytes),
		},
		Usage: usage,
		Provider: grail.ProviderInfo{
			Name:  "openai",
			Route: "responses",
			Models: []grail.ModelUse{
				{Role: "language", Name: model},
			},
		},
		RequestID: resp.ID,
		Warnings:  extractWarnings(resp),
	}, nil
}

// toResponseInput converts grail.Inputs to OpenAI Response API format.
func (p *Provider) toResponseInput(inputs []grail.Input) (responses.ResponseInputItemUnionParam, error) {
	content := make(responses.ResponseInputMessageContentListParam, 0, len(inputs))
	for i, input := range inputs {
		text, isText := grail.AsTextInput(input)
		if isText {
			content = append(content, responses.ResponseInputContentUnionParam{
				OfInputText: &responses.ResponseInputTextParam{
					Text: text,
				},
			})
			continue
		}

		data, mime, name, isFile := grail.AsFileInput(input)
		if isFile {
			if len(data) == 0 {
				return responses.ResponseInputItemUnionParam{}, fmt.Errorf("input %d: file data is empty", i)
			}

			// Detect MIME if empty (e.g., from InputImage)
			if mime == "" {
				mime = grail.SniffImageMIME(data)
			}

			// Handle images
			if strings.HasPrefix(mime, "image/") {
				b64 := base64.StdEncoding.EncodeToString(data)
				dataURL := fmt.Sprintf("data:%s;base64,%s", mime, b64)
				content = append(content, responses.ResponseInputContentUnionParam{
					OfInputImage: &responses.ResponseInputImageParam{
						Detail:   responses.ResponseInputImageDetailAuto,
						ImageURL: openai.String(dataURL),
					},
				})
				continue
			}

			// Handle PDFs
			if mime == "application/pdf" {
				// Validate PDF magic bytes
				if len(data) < 4 || string(data[0:4]) != "%PDF" {
					return responses.ResponseInputItemUnionParam{}, fmt.Errorf("input %d: invalid PDF data (missing PDF header)", i)
				}
				b64 := base64.StdEncoding.EncodeToString(data)
				dataURL := fmt.Sprintf("data:%s;base64,%s", mime, b64)
				filename := name
				if filename == "" {
					filename = "document.pdf"
				}
				content = append(content, responses.ResponseInputContentUnionParam{
					OfInputFile: &responses.ResponseInputFileParam{
						FileData: param.NewOpt(dataURL),
						Filename: param.NewOpt(filename),
						Type:     constant.InputFile("").Default(),
					},
				})
				continue
			}

			// Other file types - treat as generic file
			if mime == "" {
				mime = "application/octet-stream"
			}
			b64 := base64.StdEncoding.EncodeToString(data)
			dataURL := fmt.Sprintf("data:%s;base64,%s", mime, b64)
			filename := name
			if filename == "" {
				filename = "file"
			}
			content = append(content, responses.ResponseInputContentUnionParam{
				OfInputFile: &responses.ResponseInputFileParam{
					FileData: param.NewOpt(dataURL),
					Filename: param.NewOpt(filename),
					Type:     constant.InputFile("").Default(),
				},
			})
			continue
		}

		// FileReaderInput - read into memory for now
		// TODO: support streaming if OpenAI API supports it
		return responses.ResponseInputItemUnionParam{}, fmt.Errorf("input %d: FileReaderInput not yet supported", i)
	}

	return responses.ResponseInputItemUnionParam{
		OfMessage: &responses.EasyInputMessageParam{
			Role:    responses.EasyInputMessageRoleUser,
			Type:    responses.EasyInputMessageTypeMessage,
			Content: responses.EasyInputMessageContentUnionParam{OfInputItemContentList: content},
		},
	}, nil
}

func extractImagesFromResponse(resp *responses.Response, outputFormat string) []imageData {
	if resp == nil {
		return nil
	}
	mime := mimeFromFormat(outputFormat)
	var out []imageData
	for _, item := range resp.Output {
		if item.Type == "image_generation_call" && item.Result != "" {
			buf, err := base64.StdEncoding.DecodeString(item.Result)
			if err == nil {
				out = append(out, imageData{
					Data: buf,
					MIME: mime,
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

func mimeFromFormat(format string) string {
	switch strings.ToLower(format) {
	case "jpeg", "jpg":
		return "image/jpeg"
	case "webp":
		return "image/webp"
	default:
		return "image/png"
	}
}

func extractUsage(resp *responses.Response) grail.Usage {
	if resp == nil {
		return grail.Usage{}
	}
	// Check if Usage field exists and has values
	usage := resp.Usage
	if usage.InputTokens == 0 && usage.OutputTokens == 0 && usage.TotalTokens == 0 {
		return grail.Usage{}
	}
	return grail.Usage{
		InputTokens:  int(usage.InputTokens),
		OutputTokens: int(usage.OutputTokens),
		TotalTokens:  int(usage.TotalTokens),
	}
}

func extractWarnings(resp *responses.Response) []grail.Warning {
	// OpenAI SDK may not have Warnings field in all versions
	// Return empty slice for now
	return nil
}

func isRetryableError(err error) bool {
	// OpenAI SDK errors that are retryable
	errStr := err.Error()
	return strings.Contains(errStr, "rate_limit") ||
		strings.Contains(errStr, "timeout") ||
		strings.Contains(errStr, "temporary") ||
		strings.Contains(errStr, "503") ||
		strings.Contains(errStr, "429")
}
