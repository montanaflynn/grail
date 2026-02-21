// Package modelslab provides a ModelsLab implementation of the grail.Provider interface.
// It supports text-to-image generation using Flux, SDXL, and other community models.
//
// Example usage:
//
//	provider, err := modelslab.New()
//	if err != nil {
//		log.Fatal(err)
//	}
//	client := grail.NewClient(provider)
//	res, err := client.Generate(ctx, grail.Request{
//		Inputs: []grail.Input{grail.InputText("A sunset over mountains")},
//		Output: grail.OutputImage(grail.ImageSpec{Count: 1}),
//	})
//	if err != nil {
//		log.Fatal(err)
//	}
//	images, _ := res.Images()
//	os.WriteFile("output.png", images[0], 0644)
//
// The provider reads the API key from the MODELSLAB_API_KEY environment variable
// unless overridden via WithAPIKey or WithAPIKeyFromEnv.
//
// API docs: https://docs.modelslab.com/image-generation/overview
package modelslab

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/montanaflynn/grail"
)

const (
	// DefaultBaseURL is the base URL for the ModelsLab API.
	DefaultBaseURL = "https://modelslab.com/api/v6"

	// DefaultImageModel is the default model used when none is specified.
	DefaultImageModel = "flux"

	// DefaultImageSize is the default output image size.
	DefaultImageSize = "1024x1024"
)

var (
	// ErrAPIKeyRequired is returned when no API key is configured.
	ErrAPIKeyRequired = errors.New("modelslab: API key required (set MODELSLAB_API_KEY or use WithAPIKey/WithAPIKeyFromEnv)")
)

// Option configures the ModelsLab provider.
type Option func(*settings)

type settings struct {
	apiKey     string
	apiKeySet  bool
	imageModel string
	baseURL    string
	httpClient *http.Client
	logger     *slog.Logger
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

// WithImageModel overrides the default image model (default: "flux").
// See the model constants in this package or https://docs.modelslab.com for available models.
func WithImageModel(model string) Option {
	return func(s *settings) { s.imageModel = model }
}

// WithBaseURL overrides the default API base URL.
func WithBaseURL(url string) Option {
	return func(s *settings) { s.baseURL = url }
}

// WithHTTPClient sets a custom HTTP client.
func WithHTTPClient(hc *http.Client) Option {
	return func(s *settings) { s.httpClient = hc }
}

// WithLogger sets a custom logger for provider-level logs.
func WithLogger(l *slog.Logger) Option {
	return func(s *settings) {
		if l != nil {
			s.logger = l
		}
	}
}

// Provider is a ModelsLab-backed implementation of grail.Provider.
type Provider struct {
	apiKey     string
	imageModel string
	baseURL    string
	httpClient *http.Client
	log        *slog.Logger

	// Model catalog slots
	bestImageModel grail.Model
	fastImageModel grail.Model
}

// New creates a new ModelsLab provider.
// It reads the API key from MODELSLAB_API_KEY unless overridden.
func New(opts ...Option) (*Provider, error) {
	s := &settings{
		imageModel: DefaultImageModel,
		baseURL:    DefaultBaseURL,
		httpClient: &http.Client{Timeout: 120 * time.Second},
		logger:     slog.Default(),
	}

	for _, opt := range opts {
		opt(s)
	}

	// Auto-read from env if not explicitly set
	if !s.apiKeySet {
		if v := strings.TrimSpace(os.Getenv("MODELSLAB_API_KEY")); v != "" {
			s.apiKey = v
		}
	}

	if s.apiKey == "" {
		return nil, ErrAPIKeyRequired
	}

	p := &Provider{
		apiKey:     s.apiKey,
		imageModel: s.imageModel,
		baseURL:    s.baseURL,
		httpClient: s.httpClient,
		log:        s.logger,
	}

	// Default model catalog
	p.bestImageModel = Flux
	p.fastImageModel = SDXL

	return p, nil
}

// Name returns the provider name.
func (p *Provider) Name() string { return "modelslab" }

// SetLogger implements grail.LoggerAware.
func (p *Provider) SetLogger(l *slog.Logger) { p.log = l }

// ListModels implements grail.ModelLister.
func (p *Provider) ListModels(_ context.Context) ([]grail.Model, error) {
	return []grail.Model{Flux, FluxDev, SDXL, RealisticVision}, nil
}

// ResolveModel implements grail.ModelResolver for tier-based model selection.
func (p *Provider) ResolveModel(role grail.ModelRole, tier grail.ModelTier) (string, error) {
	if role != grail.ModelRoleImage {
		return "", grail.NewGrailError(grail.Unsupported,
			fmt.Sprintf("modelslab: unsupported model role %q (only %q is supported)", role, grail.ModelRoleImage))
	}
	switch tier {
	case grail.ModelTierBest:
		return p.bestImageModel.Name, nil
	case grail.ModelTierFast:
		return p.fastImageModel.Name, nil
	default:
		return p.imageModel, nil
	}
}

// DoGenerate implements grail.ProviderExecutor.
func (p *Provider) DoGenerate(ctx context.Context, req grail.Request) (grail.Response, error) {
	// Only image output is supported
	imageSpec, ok := grail.GetImageSpec(req.Output)
	if !ok {
		return grail.Response{}, grail.NewGrailError(grail.Unsupported,
			"modelslab: only image output is supported (use grail.OutputImage)").
			WithProviderName(p.Name())
	}

	// Extract the text prompt from inputs
	prompt, err := p.extractPrompt(req.Inputs)
	if err != nil {
		return grail.Response{}, err
	}

	// Determine count
	count := imageSpec.Count
	if count <= 0 {
		count = 1
	}

	// Determine model
	model := req.Model
	if model == "" {
		model = p.imageModel
	}

	images, err := p.generateImages(ctx, prompt, model, count)
	if err != nil {
		return grail.Response{}, err
	}

	parts := make([]grail.OutputPart, 0, len(images))
	for _, img := range images {
		parts = append(parts, grail.NewImageOutputPart(img, "image/png", ""))
	}

	return grail.Response{
		Outputs: parts,
		Provider: grail.ProviderInfo{
			Name:  p.Name(),
			Route: "images/text2img",
			Models: []grail.ModelUse{
				{Role: "image_generation", Name: model},
			},
		},
	}, nil
}

// text2imgRequest is the JSON body sent to ModelsLab.
type text2imgRequest struct {
	Key                string  `json:"key"`
	Prompt             string  `json:"prompt"`
	ModelID            string  `json:"model_id"`
	Width              string  `json:"width"`
	Height             string  `json:"height"`
	Samples            string  `json:"samples"`
	NumInferenceSteps  string  `json:"num_inference_steps"`
	GuidanceScale      float64 `json:"guidance_scale"`
	SafetyChecker      string  `json:"safety_checker"`
}

// text2imgResponse is the JSON response from ModelsLab.
type text2imgResponse struct {
	Status  string   `json:"status"`
	Output  []string `json:"output"`
	Message string   `json:"message"`
	Messege string   `json:"messege"` // ModelsLab typo in some responses
	ID      int64    `json:"id"`
}

func (p *Provider) generateImages(ctx context.Context, prompt, model string, count int) ([][]byte, error) {
	payload := text2imgRequest{
		Key:               p.apiKey,
		Prompt:            prompt,
		ModelID:           model,
		Width:             "1024",
		Height:            "1024",
		Samples:           fmt.Sprintf("%d", count),
		NumInferenceSteps: "30",
		GuidanceScale:     7.5,
		SafetyChecker:     "no",
	}

	body, err := json.Marshal(payload)
	if err != nil {
		return nil, grail.NewGrailError(grail.Internal, "failed to marshal request").WithCause(err).WithProviderName(p.Name())
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost,
		p.baseURL+"/images/text2img", bytes.NewReader(body))
	if err != nil {
		return nil, grail.NewGrailError(grail.Internal, "failed to create request").WithCause(err).WithProviderName(p.Name())
	}
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := p.httpClient.Do(httpReq)
	if err != nil {
		return nil, grail.NewGrailError(grail.Unavailable, "API request failed").WithCause(err).
			WithRetryable(true).WithProviderName(p.Name())
	}
	defer resp.Body.Close()

	if resp.StatusCode == http.StatusTooManyRequests {
		return nil, grail.NewGrailError(grail.RateLimited, "ModelsLab rate limit exceeded").
			WithRetryable(true).WithProviderName(p.Name())
	}
	if resp.StatusCode == http.StatusUnauthorized || resp.StatusCode == http.StatusForbidden {
		return nil, grail.NewGrailError(grail.Unauthorized, "invalid ModelsLab API key").WithProviderName(p.Name())
	}
	if resp.StatusCode != http.StatusOK {
		return nil, grail.NewGrailError(grail.Internal,
			fmt.Sprintf("unexpected HTTP status %d", resp.StatusCode)).WithProviderName(p.Name())
	}

	var apiResp text2imgResponse
	if err := json.NewDecoder(resp.Body).Decode(&apiResp); err != nil {
		return nil, grail.NewGrailError(grail.Internal, "failed to decode response").WithCause(err).WithProviderName(p.Name())
	}

	if apiResp.Status == "error" {
		msg := apiResp.Message
		if msg == "" {
			msg = apiResp.Messege
		}
		return nil, grail.NewGrailError(grail.Internal,
			fmt.Sprintf("ModelsLab API error: %s", msg)).WithProviderName(p.Name())
	}

	if len(apiResp.Output) == 0 {
		return nil, grail.NewGrailError(grail.Internal, "ModelsLab returned no images").WithProviderName(p.Name())
	}

	// Download all images from the returned URLs
	images := make([][]byte, 0, len(apiResp.Output))
	for _, imgURL := range apiResp.Output {
		data, err := p.downloadImage(ctx, imgURL)
		if err != nil {
			return nil, err
		}
		images = append(images, data)
	}

	return images, nil
}

func (p *Provider) downloadImage(ctx context.Context, url string) ([]byte, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, grail.NewGrailError(grail.Internal, "failed to create image download request").
			WithCause(err).WithProviderName(p.Name())
	}

	resp, err := p.httpClient.Do(req)
	if err != nil {
		return nil, grail.NewGrailError(grail.Unavailable, "failed to download generated image").
			WithCause(err).WithRetryable(true).WithProviderName(p.Name())
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, grail.NewGrailError(grail.Unavailable,
			fmt.Sprintf("image download failed with status %d", resp.StatusCode)).WithProviderName(p.Name())
	}

	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, grail.NewGrailError(grail.Internal, "failed to read image data").
			WithCause(err).WithProviderName(p.Name())
	}

	return data, nil
}

func (p *Provider) extractPrompt(inputs []grail.Input) (string, error) {
	var parts []string
	for _, input := range inputs {
		if text, ok := grail.AsTextInput(input); ok {
			parts = append(parts, text)
		}
	}
	if len(parts) == 0 {
		return "", grail.NewGrailError(grail.InvalidArgument,
			"modelslab: at least one text input is required for image generation").WithProviderName(p.Name())
	}
	return strings.Join(parts, " "), nil
}
