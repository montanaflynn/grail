// Package anthropic provides an Anthropic Claude implementation of the
// grail.Provider interface. It uses the Messages API for text and JSON
// generation, with image and PDF inputs. Claude cannot generate images, so
// image output requests return a grail.Unsupported error.
//
// Example usage:
//
//	provider, err := anthropic.New()
//	if err != nil {
//		log.Fatal(err)
//	}
//	client := grail.NewClient(provider)
//	res, err := client.Generate(ctx, grail.Request{
//		Inputs: []grail.Input{grail.InputText("Hello, world!")},
//		Output: grail.OutputText(),
//	})
//
// The provider automatically uses the ANTHROPIC_API_KEY environment variable
// if no API key is explicitly provided via WithAPIKey or WithAPIKeyFromEnv.
//
// Default models:
//   - Text (best): claude-fable-5-1
//   - Text (fast): claude-haiku-5-5
package anthropic

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"iter"
	"log/slog"
	"net/http"
	"os"
	"strings"

	"github.com/montanaflynn/grail"

	"github.com/anthropics/anthropic-sdk-go/v2"
	"github.com/anthropics/anthropic-sdk-go/v2/option"
)

const (
	// DefaultTextModelName is the Claude text model used when no override is provided.
	DefaultTextModelName = anthropic.ModelClaudeFable5_1
	// DefaultMaxTokens is the max_tokens sent when TextOptions.MaxTokens is not
	// set. The Messages API requires it; this value leaves room for long
	// answers while keeping non-streaming requests under SDK timeouts.
	DefaultMaxTokens = 16000

	providerName = "anthropic"
	route        = "messages"
)

var (
	// ErrAPIKeyRequired is returned when no API key is configured.
	ErrAPIKeyRequired = errors.New("anthropic: API key required (set ANTHROPIC_API_KEY or use WithAPIKey/WithAPIKeyFromEnv)")
)

// jsonObjectInstruction is added to the system prompt for JSON output without
// a schema, since the Messages API has no schema-less JSON mode.
const jsonObjectInstruction = "Respond with a single valid JSON object and nothing else."

// Option configures the Anthropic provider.
type Option func(*settings)

type settings struct {
	apiKey    string
	apiKeySet bool
	textModel string
	logger    *slog.Logger
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

// WithTextModel overrides the default text model (default: claude-fable-5-1).
func WithTextModel(model string) Option {
	return func(s *settings) { s.textModel = model }
}

// WithLogger sets a custom logger for provider-level logs.
func WithLogger(l *slog.Logger) Option {
	return func(s *settings) {
		if l != nil {
			s.logger = l
		}
	}
}

// Provider is an Anthropic-backed implementation of grail.Provider.
type Provider struct {
	client    anthropic.Client
	textModel string
	log       *slog.Logger

	// Model catalog slots
	bestTextModel grail.Model
	fastTextModel grail.Model
}

// TextOptions provides Anthropic-specific options for text and JSON output.
type TextOptions struct {
	Model        string
	MaxTokens    *int32 // default DefaultMaxTokens
	SystemPrompt string
}

func (TextOptions) ApplyProviderOption() {}

// New constructs an Anthropic provider using functional options.
func New(opts ...Option) (*Provider, error) {
	cfg := settings{
		textModel: DefaultTextModelName,
		logger:    slog.Default(),
	}
	for _, opt := range opts {
		opt(&cfg)
	}

	switch {
	case cfg.apiKeySet && cfg.apiKey == "":
		return nil, ErrAPIKeyRequired
	case !cfg.apiKeySet && cfg.apiKey == "":
		cfg.apiKey = strings.TrimSpace(os.Getenv("ANTHROPIC_API_KEY"))
		if cfg.apiKey == "" {
			return nil, ErrAPIKeyRequired
		}
	}

	return &Provider{
		client:    anthropic.NewClient(option.WithAPIKey(cfg.apiKey)),
		textModel: cfg.textModel,
		log:       cfg.logger,
		// Initialize model catalog with defaults
		bestTextModel: ClaudeFable5_1,
		fastTextModel: ClaudeHaiku5_5,
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
	return providerName
}

// ModelCatalog implementation

// SetBestTextModel sets the model to use for best-quality text generation.
func (p *Provider) SetBestTextModel(model grail.Model) { p.bestTextModel = model }

// SetFastTextModel sets the model to use for fast text generation.
func (p *Provider) SetFastTextModel(model grail.Model) { p.fastTextModel = model }

// SetBestImageModel is a no-op: Claude has no image generation models.
func (p *Provider) SetBestImageModel(grail.Model) {}

// SetFastImageModel is a no-op: Claude has no image generation models.
func (p *Provider) SetFastImageModel(grail.Model) {}

// BestTextModel returns the model used for best-quality text generation.
func (p *Provider) BestTextModel() grail.Model { return p.bestTextModel }

// FastTextModel returns the model used for fast text generation.
func (p *Provider) FastTextModel() grail.Model { return p.fastTextModel }

// BestImageModel returns the zero Model: Claude has no image generation models.
func (p *Provider) BestImageModel() grail.Model { return grail.Model{} }

// FastImageModel returns the zero Model: Claude has no image generation models.
func (p *Provider) FastImageModel() grail.Model { return grail.Model{} }

// AllModels returns all configured models.
func (p *Provider) AllModels() []grail.Model {
	return []grail.Model{
		p.bestTextModel,
		p.fastTextModel,
		// Additional models not set as best/fast
		ClaudeOpus5_5,
		ClaudeSonnet5_5,
	}
}

// ListModels returns all available Anthropic models and their capabilities.
func (p *Provider) ListModels(ctx context.Context) ([]grail.Model, error) {
	return p.AllModels(), nil
}

// ResolveModel resolves a role+tier to a model name. Image roles return a
// grail.Unsupported error.
func (p *Provider) ResolveModel(role grail.ModelRole, tier grail.ModelTier) (string, error) {
	if role == grail.ModelRoleImage {
		return "", errImageUnsupported()
	}
	switch {
	case role == grail.ModelRoleText && tier == grail.ModelTierBest:
		return p.bestTextModel.Name, nil
	case role == grail.ModelRoleText && tier == grail.ModelTierFast:
		return p.fastTextModel.Name, nil
	default:
		return "", fmt.Errorf("anthropic: no %s model with tier %s", role, tier)
	}
}

// DescribeModels returns the text model that will serve the request.
func (p *Provider) DescribeModels(req grail.Request) string {
	model, _ := p.textConfig(req)
	return model
}

// DoGenerate implements the ProviderExecutor interface.
func (p *Provider) DoGenerate(ctx context.Context, req grail.Request) (grail.Response, error) {
	if _, isImage := grail.GetImageSpec(req.Output); isImage {
		return grail.Response{}, errImageUnsupported()
	}

	content, err := toContentBlocks(req.Inputs)
	if err != nil {
		return grail.Response{}, grail.NewGrailError(grail.InvalidArgument, fmt.Sprintf("failed to convert inputs: %v", err)).WithCause(err).WithProviderName(providerName)
	}

	if grail.IsTextOutput(req.Output) {
		return p.generateText(ctx, req, content)
	}
	if schema, strict, isJSON := grail.GetJSONOutput(req.Output); isJSON {
		return p.generateJSON(ctx, req, content, schema, strict)
	}
	return grail.Response{}, grail.NewGrailError(grail.Unsupported, fmt.Sprintf("unsupported output type: %T", req.Output)).WithProviderName(providerName)
}

// textConfig returns the model (Request.Model > TextOptions.Model > provider
// default) and the TextOptions for req.
func (p *Provider) textConfig(req grail.Request) (string, TextOptions) {
	var textOpts TextOptions
	for _, opt := range req.ProviderOptions {
		if to, ok := opt.(TextOptions); ok {
			textOpts = to
		}
	}
	model := p.textModel
	if textOpts.Model != "" {
		model = textOpts.Model
	}
	if req.Model != "" {
		model = req.Model
	}
	return model, textOpts
}

// newParams builds Messages API params for content using the request's text
// options. extraSystem is appended to the system prompt when non-empty.
func (p *Provider) newParams(req grail.Request, content []anthropic.ContentBlockParamUnion, extraSystem string) (anthropic.MessageNewParams, string) {
	model, textOpts := p.textConfig(req)

	maxTokens := int64(DefaultMaxTokens)
	if textOpts.MaxTokens != nil {
		maxTokens = int64(*textOpts.MaxTokens)
	}

	params := anthropic.MessageNewParams{
		Model:     anthropic.Model(model),
		MaxTokens: maxTokens,
		Messages:  []anthropic.MessageParam{anthropic.NewUserMessage(content...)},
	}

	system := textOpts.SystemPrompt
	if extraSystem != "" {
		if system != "" {
			system += "\n\n"
		}
		system += extraSystem
	}
	if system != "" {
		params.System = []anthropic.TextBlockParam{{Text: system}}
	}
	return params, model
}

func (p *Provider) generateText(ctx context.Context, req grail.Request, content []anthropic.ContentBlockParamUnion) (grail.Response, error) {
	params, model := p.newParams(req, content, "")

	if p.log != nil {
		p.log.Debug("anthropic generate text request", slog.String("model", model))
	}

	msg, err := p.client.Messages.New(ctx, params)
	if err != nil {
		return grail.Response{}, toGrailError("generate text", err)
	}
	if err := checkRefusal(msg); err != nil {
		return grail.Response{}, err
	}

	usage := extractUsage(msg)
	if p.log != nil {
		p.log.Debug("anthropic generate text response", slog.Any("usage", usage))
	}

	return grail.Response{
		Outputs:   []grail.OutputPart{grail.NewTextOutputPart(extractText(msg))},
		Usage:     usage,
		Provider:  providerInfo(model),
		RequestID: msg.ID,
		Warnings:  extractWarnings(msg),
	}, nil
}

// DoStream implements grail.StreamExecutor using Messages API streaming.
// Only text output is supported.
func (p *Provider) DoStream(ctx context.Context, req grail.Request) iter.Seq2[grail.StreamEvent, error] {
	return func(yield func(grail.StreamEvent, error) bool) {
		if !grail.IsTextOutput(req.Output) {
			yield(grail.StreamEvent{}, grail.NewGrailError(grail.Unsupported, fmt.Sprintf("anthropic: streaming supports text output only, got %T", req.Output)).WithProviderName(providerName))
			return
		}
		content, err := toContentBlocks(req.Inputs)
		if err != nil {
			yield(grail.StreamEvent{}, grail.NewGrailError(grail.InvalidArgument, fmt.Sprintf("failed to convert inputs: %v", err)).WithCause(err).WithProviderName(providerName))
			return
		}
		params, model := p.newParams(req, content, "")

		if p.log != nil {
			p.log.Debug("anthropic stream text request", slog.String("model", model))
		}

		// Canceling the context and closing the stream releases the
		// connection when the consumer stops early.
		ctx, cancel := context.WithCancel(ctx)
		defer cancel()
		stream := p.client.Messages.NewStreaming(ctx, params)
		defer stream.Close()

		var msg anthropic.Message
		var stopped bool
		for stream.Next() {
			ev := stream.Current()
			if err := msg.Accumulate(ev); err != nil {
				yield(grail.StreamEvent{}, grail.NewGrailError(grail.Internal, fmt.Sprintf("anthropic stream text failed: %v", err)).WithCause(err).WithProviderName(providerName))
				return
			}
			switch ev.Type {
			case "content_block_delta":
				if ev.Delta.Type == "text_delta" && ev.Delta.Text != "" && !yield(grail.StreamEvent{Text: ev.Delta.Text}, nil) {
					return
				}
			case "message_stop":
				stopped = true
			}
		}
		if err := stream.Err(); err != nil {
			yield(grail.StreamEvent{}, toGrailError("stream text", err))
			return
		}
		if !stopped {
			yield(grail.StreamEvent{}, grail.NewGrailError(grail.Internal, "anthropic stream text ended before message_stop").WithProviderName(providerName).WithRequestID(msg.ID).WithRetryable(true))
			return
		}
		if err := checkRefusal(&msg); err != nil {
			yield(grail.StreamEvent{}, err)
			return
		}

		usage := extractUsage(&msg)
		if p.log != nil {
			p.log.Debug("anthropic stream text response", slog.Any("usage", usage))
		}
		yield(grail.StreamEvent{
			Done:      true,
			Usage:     usage,
			Provider:  providerInfo(model),
			RequestID: msg.ID,
			Warnings:  extractWarnings(&msg),
		}, nil)
	}
}

func (p *Provider) generateJSON(ctx context.Context, req grail.Request, content []anthropic.ContentBlockParamUnion, schema any, strict bool) (grail.Response, error) {
	schemaMap, err := grail.JSONSchemaMap(schema)
	if err != nil {
		return grail.Response{}, grail.NewGrailError(grail.InvalidArgument, fmt.Sprintf("invalid JSON schema: %v", err)).WithCause(err).WithProviderName(providerName)
	}

	// Use structured outputs when a schema is given. The API has no
	// schema-less JSON mode, so without a schema we instruct the model and
	// rely on the parse check below.
	var params anthropic.MessageNewParams
	var model string
	if schemaMap != nil {
		params, model = p.newParams(req, content, "")
		params.OutputConfig.Format = anthropic.JSONOutputFormatParam{Schema: schemaMap}
	} else {
		params, model = p.newParams(req, content, jsonObjectInstruction)
	}

	if p.log != nil {
		p.log.Debug("anthropic generate JSON request", slog.String("model", model))
	}

	msg, err := p.client.Messages.New(ctx, params)
	if err != nil {
		return grail.Response{}, toGrailError("generate JSON", err)
	}
	if err := checkRefusal(msg); err != nil {
		return grail.Response{}, err
	}

	jsonBytes := []byte(extractText(msg))
	if strict {
		var test any
		if err := json.Unmarshal(jsonBytes, &test); err != nil {
			return grail.Response{}, grail.NewGrailError(grail.OutputInvalid, fmt.Sprintf("invalid JSON output (stop reason %q): %v", msg.StopReason, err)).WithProviderName(providerName).WithRequestID(msg.ID)
		}
	}

	usage := extractUsage(msg)
	if p.log != nil {
		p.log.Debug("anthropic generate JSON response", slog.Any("usage", usage))
	}

	return grail.Response{
		Outputs:   []grail.OutputPart{grail.NewJSONOutputPart(jsonBytes)},
		Usage:     usage,
		Provider:  providerInfo(model),
		RequestID: msg.ID,
		Warnings:  extractWarnings(msg),
	}, nil
}

// toContentBlocks converts grail.Inputs to Messages API content blocks, in
// order. Images map to image blocks, PDFs to document blocks, and text/*
// files to plain-text document blocks.
func toContentBlocks(inputs []grail.Input) ([]anthropic.ContentBlockParamUnion, error) {
	blocks := make([]anthropic.ContentBlockParamUnion, 0, len(inputs))
	for i, input := range inputs {
		if text, ok := grail.AsTextInput(input); ok {
			blocks = append(blocks, anthropic.NewTextBlock(text))
			continue
		}

		data, mime, name, ok := grail.AsFileInput(input)
		if !ok {
			// FileReaderInput - not supported yet
			return nil, fmt.Errorf("input %d: FileReaderInput not yet supported", i)
		}
		if len(data) == 0 {
			return nil, fmt.Errorf("input %d: file data is empty", i)
		}

		// Detect MIME if empty (e.g., from InputImage)
		if mime == "" {
			mime = grail.SniffImageMIME(data)
		}

		switch {
		case strings.HasPrefix(mime, "image/"):
			mediaType, err := imageMediaType(mime)
			if err != nil {
				return nil, fmt.Errorf("input %d: %w", i, err)
			}
			blocks = append(blocks, anthropic.NewImageBlock(anthropic.Base64ImageSourceParam{
				Data:      base64.StdEncoding.EncodeToString(data),
				MediaType: mediaType,
			}))
		case mime == "application/pdf":
			if len(data) < 4 || string(data[0:4]) != "%PDF" {
				return nil, fmt.Errorf("input %d: invalid PDF data (missing PDF header)", i)
			}
			blocks = append(blocks, withTitle(anthropic.NewDocumentBlock(anthropic.Base64PDFSourceParam{
				Data: base64.StdEncoding.EncodeToString(data),
			}), name))
		case strings.HasPrefix(mime, "text/"):
			blocks = append(blocks, withTitle(anthropic.NewDocumentBlock(anthropic.PlainTextSourceParam{
				Data: string(data),
			}), name))
		default:
			return nil, fmt.Errorf("input %d: unsupported file type %q (supported: images, PDF, text)", i, mime)
		}
	}
	return blocks, nil
}

// withTitle sets a document block's title to the file name, if any.
func withTitle(block anthropic.ContentBlockParamUnion, name string) anthropic.ContentBlockParamUnion {
	if name != "" && block.OfDocument != nil {
		block.OfDocument.Title = anthropic.String(name)
	}
	return block
}

func imageMediaType(mime string) (anthropic.Base64ImageSourceMediaType, error) {
	switch mime {
	case "image/jpeg", "image/jpg":
		return anthropic.Base64ImageSourceMediaTypeImageJPEG, nil
	case "image/png":
		return anthropic.Base64ImageSourceMediaTypeImagePNG, nil
	case "image/gif":
		return anthropic.Base64ImageSourceMediaTypeImageGIF, nil
	case "image/webp":
		return anthropic.Base64ImageSourceMediaTypeImageWebP, nil
	default:
		return "", fmt.Errorf("unsupported image type %q (supported: jpeg, png, gif, webp)", mime)
	}
}

func errImageUnsupported() error {
	return grail.NewGrailError(grail.Unsupported, "anthropic: image generation is not supported; use a provider with image models").WithProviderName(providerName)
}

// checkRefusal returns a grail.Refused error when Claude declined the request.
func checkRefusal(msg *anthropic.Message) error {
	if msg.StopReason != anthropic.StopReasonRefusal {
		return nil
	}
	text := "anthropic: request refused"
	if c := msg.StopDetails.Category; c != "" {
		text += fmt.Sprintf(" (category %s)", c)
	}
	if e := msg.StopDetails.Explanation; e != "" {
		text += ": " + e
	}
	return grail.NewGrailError(grail.Refused, text).WithProviderName(providerName).WithRequestID(msg.ID)
}

// extractText concatenates the message's text blocks, skipping thinking and
// other non-text blocks.
func extractText(msg *anthropic.Message) string {
	var b strings.Builder
	for _, block := range msg.Content {
		if tb, ok := block.AsAny().(anthropic.TextBlock); ok {
			b.WriteString(tb.Text)
		}
	}
	return b.String()
}

func extractUsage(msg *anthropic.Message) grail.Usage {
	in := int(msg.Usage.InputTokens + msg.Usage.CacheCreationInputTokens + msg.Usage.CacheReadInputTokens)
	out := int(msg.Usage.OutputTokens)
	return grail.Usage{
		InputTokens:  in,
		OutputTokens: out,
		TotalTokens:  in + out,
	}
}

func extractWarnings(msg *anthropic.Message) []grail.Warning {
	if msg.StopReason == anthropic.StopReasonMaxTokens {
		return []grail.Warning{{
			Code:    "max_tokens",
			Message: "output was truncated at max_tokens; raise TextOptions.MaxTokens",
		}}
	}
	return nil
}

func providerInfo(model string) grail.ProviderInfo {
	return grail.ProviderInfo{
		Name:   providerName,
		Route:  route,
		Models: []grail.ModelUse{{Role: "language", Name: model}},
	}
}

// toGrailError maps an SDK error to a grail error code by HTTP status.
func toGrailError(op string, err error) error {
	msg := fmt.Sprintf("anthropic %s failed: %v", op, err)

	var apiErr *anthropic.Error
	if !errors.As(err, &apiErr) {
		if errors.Is(err, context.DeadlineExceeded) {
			return grail.NewGrailError(grail.Timeout, msg).WithCause(err).WithProviderName(providerName).WithRetryable(true)
		}
		return grail.NewGrailError(grail.Internal, msg).WithCause(err).WithProviderName(providerName).WithRetryable(isRetryableError(err))
	}

	code := grail.Internal
	switch s := apiErr.StatusCode; {
	case s == http.StatusBadRequest || s == http.StatusNotFound || s == http.StatusRequestEntityTooLarge:
		code = grail.InvalidArgument
	case s == http.StatusUnauthorized || s == http.StatusForbidden:
		code = grail.Unauthorized
	case s == http.StatusRequestTimeout:
		code = grail.Timeout
	case s == http.StatusTooManyRequests:
		code = grail.RateLimited
	case s >= 500: // includes 529 overloaded
		code = grail.Unavailable
	}
	return grail.NewGrailError(code, msg).
		WithCause(err).
		WithProviderName(providerName).
		WithRequestID(apiErr.RequestID).
		WithRetryable(code == grail.RateLimited || code == grail.Timeout || code == grail.Unavailable)
}

func isRetryableError(err error) bool {
	errStr := err.Error()
	return strings.Contains(errStr, "timeout") ||
		strings.Contains(errStr, "temporary") ||
		strings.Contains(errStr, "connection reset")
}
