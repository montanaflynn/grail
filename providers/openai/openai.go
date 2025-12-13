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
//	res, err := client.GenerateText(ctx, grail.TextRequest{
//		Input: []grail.Part{grail.Text("Hello, world!")},
//	})
//
// The provider automatically uses the OPENAI_API_KEY environment variable
// if no API key is explicitly provided via WithAPIKey or WithAPIKeyFromEnv.
//
// Default models:
//   - Text: gpt-5.1
//   - Image: gpt-image-1
package openai

import (
	"context"
	"encoding/base64"
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
	DefaultTextModelName = shared.ChatModelGPT5_1
	// DefaultImageModelName is the OpenAI image model used when no override is provided.
	DefaultImageModelName = openai.ImageModelGPTImage1
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

// WithTextModel overrides the default text model (default: gpt-5.1).
func WithTextModel(model string) Option {
	return func(s *settings) { s.textModel = model }
}

// WithImageModel overrides the default image model (default: gpt-image-1).
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
	desc string
	fn   func(*imageConfig)
}

func (o imageOptionFunc) Description() string { return o.desc }
func (o imageOptionFunc) apply(cfg *imageConfig) {
	if o.fn != nil {
		o.fn(cfg)
	}
}

// WithImageFormat sets the OpenAI image output format.
func WithImageFormat(f ImageFormat) ImageOption {
	return imageOptionFunc{
		desc: fmt.Sprintf("openai image format %s", f),
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
		desc: fmt.Sprintf("openai image background %s", b),
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
		desc: fmt.Sprintf("openai image size %s", size),
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
		desc: fmt.Sprintf("openai image moderation %s", moderation),
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
		desc: fmt.Sprintf("openai image output compression %d%%", compression),
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

// DefaultTextModel returns the configured/default text model.
func (p *Provider) DefaultTextModel() string {
	return p.textModel
}

// DefaultImageModel returns the configured/default image model.
func (p *Provider) DefaultImageModel() string {
	return p.imageModel
}

// GenerateText performs text generation using OpenAI chat completions.
func (p *Provider) GenerateText(ctx context.Context, req grail.TextRequest) (grail.TextResult, error) {
	if len(req.Input) == 0 {
		return grail.TextResult{}, errors.New("input must not be empty")
	}

	item, err := toResponseInput(req.Input)
	if err != nil {
		return grail.TextResult{}, err
	}

	model := req.Options.Model
	if model == "" {
		model = p.textModel
	}

	if p.log != nil {
		p.log.Debug("openai generate text request", slog.String("model", model), slog.Any("options", req.Options), slog.Any("parts_summary", summarizeParts(req.Input)))
	}

	params := responses.ResponseNewParams{
		Model: shared.ChatModel(model),
		Input: responses.ResponseNewParamsInputUnion{
			OfInputItemList: responses.ResponseInputParam{item},
		},
	}

	if req.Options.SystemPrompt != "" {
		params.Instructions = param.NewOpt(req.Options.SystemPrompt)
	}
	if req.Options.MaxTokens != nil {
		params.MaxOutputTokens = openai.Int(int64(*req.Options.MaxTokens))
	}
	if req.Options.Temperature != nil {
		params.Temperature = openai.Float(float64(*req.Options.Temperature))
	}
	if req.Options.TopP != nil {
		params.TopP = openai.Float(float64(*req.Options.TopP))
	}

	resp, err := p.client.Responses.New(ctx, params)
	if err != nil {
		return grail.TextResult{}, fmt.Errorf("openai generate text: %w", err)
	}

	text := resp.OutputText()

	if p.log != nil {
		p.log.Debug("openai generate text response", slog.Any("raw", resp))
	}

	return grail.TextResult{
		Text: text,
		Raw:  resp,
	}, nil
}

// GenerateImage performs image generation using OpenAI Responses API.
func (p *Provider) GenerateImage(ctx context.Context, req grail.ImageRequest) (grail.ImageResult, error) {
	if len(req.Input) == 0 {
		return grail.ImageResult{}, errors.New("input must not be empty")
	}

	item, err := toResponseInput(req.Input)
	if err != nil {
		return grail.ImageResult{}, err
	}

	model := req.Options.Model
	if model == "" {
		model = p.textModel
	}

	if p.log != nil {
		p.log.Debug("openai generate image request", slog.String("model", model), slog.Any("parts_summary", summarizeParts(req.Input)))
	}

	cfg := imageConfig{
		format:     ImageFormat(p.imgFormat),
		background: ImageBackground("auto"),
		size:       ImageSizeAuto,
		moderation: ImageModerationAuto,
	}
	for _, opt := range req.ProviderOptions {
		if fn, ok := opt.(ImageOption); ok && fn != nil {
			fn.apply(&cfg)
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

	if req.Options.SystemPrompt != "" {
		params.Instructions = param.NewOpt(req.Options.SystemPrompt)
	}

	resp, err := p.client.Responses.New(ctx, params)
	if err != nil {
		return grail.ImageResult{}, fmt.Errorf("openai generate image: %w", err)
	}

	imgs := extractImagesFromResponse(resp, string(cfg.format))

	if p.log != nil {
		p.log.Debug("openai generate image response", slog.Int("images", len(imgs)), slog.Any("raw", resp))
	}

	return grail.ImageResult{
		Images: imgs,
		Raw:    resp,
	}, nil
}

func toChatParts(input []grail.Part) ([]openai.ChatCompletionContentPartUnionParam, error) {
	out := make([]openai.ChatCompletionContentPartUnionParam, 0, len(input))
	for i, p := range input {
		switch v := p.(type) {
		case grail.TextPart:
			out = append(out, openai.TextContentPart(v.Text))
		case grail.ImagePart:
			if len(v.Data) == 0 {
				return nil, fmt.Errorf("part %d: image data is empty", i)
			}
			mime := v.MIME
			if mime == "" {
				mime = "image/png"
			}
			b64 := base64.StdEncoding.EncodeToString(v.Data)
			url := fmt.Sprintf("data:%s;base64,%s", mime, b64)
			out = append(out, openai.ImageContentPart(openai.ChatCompletionContentPartImageImageURLParam{
				URL: url,
			}))
		default:
			return nil, fmt.Errorf("part %d: unknown part type %T", i, p)
		}
	}
	return out, nil
}

// toResponseInput flattens ordered parts into a single user message for Responses.
func toResponseInput(input []grail.Part) (responses.ResponseInputItemUnionParam, error) {
	content := make(responses.ResponseInputMessageContentListParam, 0, len(input))
	for i, p := range input {
		switch v := p.(type) {
		case grail.TextPart:
			content = append(content, responses.ResponseInputContentUnionParam{
				OfInputText: &responses.ResponseInputTextParam{
					Text: v.Text,
				},
			})
		case grail.ImagePart:
			if len(v.Data) == 0 {
				return responses.ResponseInputItemUnionParam{}, fmt.Errorf("part %d: image data is empty", i)
			}
			mime := v.MIME
			if mime == "" {
				mime = "image/png"
			}
			b64 := base64.StdEncoding.EncodeToString(v.Data)
			dataURL := fmt.Sprintf("data:%s;base64,%s", mime, b64)
			content = append(content, responses.ResponseInputContentUnionParam{
				OfInputImage: &responses.ResponseInputImageParam{
					Detail:   responses.ResponseInputImageDetailAuto,
					ImageURL: openai.String(dataURL),
				},
			})
		case grail.PDFPart:
			if len(v.Data) == 0 {
				return responses.ResponseInputItemUnionParam{}, fmt.Errorf("part %d: PDF data is empty", i)
			}
			// Validate PDF magic bytes
			if len(v.Data) < 4 || string(v.Data[0:4]) != "%PDF" {
				return responses.ResponseInputItemUnionParam{}, fmt.Errorf("part %d: invalid PDF data (missing PDF header)", i)
			}
			mime := v.MIME
			if mime == "" {
				mime = "application/pdf"
			}
			// Encode to base64
			b64 := base64.StdEncoding.EncodeToString(v.Data)
			// FileData should be a data URL: "data:application/pdf;base64,<base64>"
			dataURL := fmt.Sprintf("data:%s;base64,%s", mime, b64)
			// Use provided filename or default to "document.pdf"
			filename := v.Filename
			if filename == "" {
				filename = "document.pdf"
			}
			// Type must be "input_file" constant
			content = append(content, responses.ResponseInputContentUnionParam{
				OfInputFile: &responses.ResponseInputFileParam{
					FileData: param.NewOpt(dataURL),
					Filename: param.NewOpt(filename),
					Type:     constant.InputFile("").Default(),
				},
			})
		default:
			return responses.ResponseInputItemUnionParam{}, fmt.Errorf("part %d: unknown part type %T", i, p)
		}
	}

	return responses.ResponseInputItemUnionParam{
		OfMessage: &responses.EasyInputMessageParam{
			Role:    responses.EasyInputMessageRoleUser,
			Type:    responses.EasyInputMessageTypeMessage,
			Content: responses.EasyInputMessageContentUnionParam{OfInputItemContentList: content},
		},
	}, nil
}

func extractImagesFromResponse(resp *responses.Response, outputFormat string) []grail.ImageOutput {
	if resp == nil {
		return nil
	}
	mime := mimeFromFormat(outputFormat)
	var out []grail.ImageOutput
	for _, item := range resp.Output {
		if item.Type == "image_generation_call" && item.Result != "" {
			buf, err := base64.StdEncoding.DecodeString(item.Result)
			if err == nil {
				out = append(out, grail.ImageOutput{
					Data: buf,
					MIME: mime,
				})
			}
		}
	}
	return out
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

func summarizeParts(parts []grail.Part) []map[string]any {
	var out []map[string]any
	for _, p := range parts {
		switch v := p.(type) {
		case grail.TextPart:
			out = append(out, map[string]any{
				"type": "text",
				"len":  len(v.Text),
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
				"type": "unknown",
			})
		}
	}
	return out
}
