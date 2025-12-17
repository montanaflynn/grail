// Package grail provides a unified interface for AI text and image generation
// across multiple providers (OpenAI, Gemini, etc.). It supports multimodal
// inputs (ordered sequences of text, images, and PDFs) and provides type-safe error
// handling, structured logging, and flexible configuration options.
//
// Example usage:
//
//	provider, _ := openai.New()
//	client := grail.NewClient(provider)
//	res, _ := client.Generate(ctx, grail.Request{
//		Inputs: []grail.Input{grail.InputText("Hello, world!")},
//		Output: grail.OutputText(),
//	})
//
// Sub-packages:
//
// This package provides the core client and interfaces. Provider implementations
// are available in sub-packages:
//
//   - providers - All providers (https://pkg.go.dev/github.com/montanaflynn/grail/providers)
//   - providers/openai - OpenAI provider (https://pkg.go.dev/github.com/montanaflynn/grail/providers/openai)
//   - providers/gemini - Google Gemini provider (https://pkg.go.dev/github.com/montanaflynn/grail/providers/gemini)
//   - providers/mock - Mock provider (https://pkg.go.dev/github.com/montanaflynn/grail/providers/mock)
package grail

import (
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
)

//
// Errors
//

type ErrorCode string

const (
	InvalidArgument ErrorCode = "invalid_argument"
	Unauthorized    ErrorCode = "unauthorized"
	RateLimited     ErrorCode = "rate_limited"
	Timeout         ErrorCode = "timeout"
	Unavailable     ErrorCode = "unavailable"
	Unsupported     ErrorCode = "unsupported"
	Refused         ErrorCode = "refused"
	OutputInvalid   ErrorCode = "output_invalid"
	Internal        ErrorCode = "internal"
)

type GrailError interface {
	error
	Code() ErrorCode
	Retryable() bool
	ProviderName() string
	RequestID() string
}

type grailError struct {
	code         ErrorCode
	message      string
	cause        error
	retryable    bool
	providerName string
	requestID    string
}

func (e *grailError) Error() string {
	if e == nil {
		return ""
	}
	if e.cause != nil {
		return fmt.Sprintf("%s: %s: %v", e.code, e.message, e.cause)
	}
	return fmt.Sprintf("%s: %s", e.code, e.message)
}

func (e *grailError) Unwrap() error {
	return e.cause
}

func (e *grailError) Code() ErrorCode {
	return e.code
}

func (e *grailError) Retryable() bool {
	return e.retryable
}

func (e *grailError) ProviderName() string {
	return e.providerName
}

func (e *grailError) RequestID() string {
	return e.requestID
}

func NewGrailError(code ErrorCode, message string) *grailError {
	return &grailError{
		code:    code,
		message: message,
	}
}

func (e *grailError) WithCause(cause error) *grailError {
	e.cause = cause
	return e
}

func (e *grailError) WithRetryable(retryable bool) *grailError {
	e.retryable = retryable
	return e
}

func (e *grailError) WithProviderName(name string) *grailError {
	e.providerName = name
	return e
}

func (e *grailError) WithRequestID(id string) *grailError {
	e.requestID = id
	return e
}

func IsRetryable(err error) bool {
	var ge GrailError
	if errors.As(err, &ge) {
		// If Retryable is explicitly set, use it
		if ge.Retryable() {
			return true
		}
		// Otherwise, check the error code
		code := ge.Code()
		return code == RateLimited || code == Timeout || code == Unavailable
	}
	// For non-GrailError, check code
	code := GetErrorCode(err)
	return code == RateLimited || code == Timeout || code == Unavailable
}

func IsRateLimited(err error) bool {
	return GetErrorCode(err) == RateLimited
}

func IsRefused(err error) bool {
	return GetErrorCode(err) == Refused
}

func GetErrorCode(err error) ErrorCode {
	var ge GrailError
	if err == nil {
		return ""
	}
	if errors.As(err, &ge) {
		return ge.Code()
	}
	return Internal
}

//
// Usage / warnings
//

type Usage struct {
	InputTokens  int
	OutputTokens int
	TotalTokens  int
}

type Warning struct {
	Code    string
	Message string
}

//
// ProviderInfo (observability only; not control)
//

type ProviderInfo struct {
	Name   string
	Route  string // provider-defined (e.g. "responses", "images")
	Models []ModelUse
}

type ModelUse struct {
	Role string // "language", "image_generation", "moderation", etc.
	Name string // provider-native model identifier
}

// ModelRole describes the primary function of a model.
type ModelRole string

const (
	ModelRoleText  ModelRole = "text"  // Text/language generation
	ModelRoleImage ModelRole = "image" // Image generation
)

// ModelTier describes the quality/speed trade-off of a model.
type ModelTier string

const (
	ModelTierBest ModelTier = "best" // Highest quality, may be slower/costlier
	ModelTierFast ModelTier = "fast" // Speed/cost optimized
)

// ModelInfo describes a model and its capabilities.
type ModelInfo struct {
	Name         string            // Model identifier (e.g., "gpt-5.2", "gemini-3-flash-preview")
	Role         ModelRole         // text or image
	Tier         ModelTier         // best or fast
	Capabilities ModelCapabilities // What the model can do
	Description  string            // Optional description
	Tags         []string          // Taxonomy tags (e.g., "best", "fast", "latest", "preview", "multimodal")
}

// ModelCapabilities describes what a model can do.
type ModelCapabilities struct {
	Text       bool // Can generate text
	Image      bool // Can generate images
	ImageInput bool // Can accept image inputs
	PDFInput   bool // Can accept PDF inputs
	JSON       bool // Can generate structured JSON output
	Multimodal bool // Can handle multiple input types in one request
}

//
// Inputs
//

type Input interface{ isInput() }

type textInput struct {
	Text string
}

func (textInput) isInput() {}

func InputText(s string) Input {
	return textInput{Text: s}
}

type fileInput struct {
	Data []byte
	MIME string
	Name string // optional filename
}

func (fileInput) isInput() {}

func InputFile(data []byte, mime string, opts ...FileOpt) Input {
	fi := fileInput{
		Data: data,
		MIME: mime,
	}
	// Apply options
	fo := &fileOpt{}
	for _, opt := range opts {
		if opt != nil {
			opt.applyFileOpt(fo)
		}
	}
	if fo.name != "" {
		fi.Name = fo.name
	}
	return fi
}

func InputPDF(data []byte, opts ...FileOpt) Input {
	return InputFile(data, "application/pdf", opts...)
}

func InputImage(data []byte, opts ...FileOpt) Input {
	// Don't validate here - validation happens at Generate time
	// Use empty MIME as marker that this should be an image - validation will sniff and verify
	return InputFile(data, "", opts...)
}

type fileReaderInput struct {
	R    io.Reader
	Size int64 // -1 if unknown
	MIME string
	Name string
}

func (fileReaderInput) isInput() {}

func InputFileReader(r io.Reader, size int64, mime string, opts ...FileOpt) Input {
	fri := fileReaderInput{
		R:    r,
		Size: size,
		MIME: mime,
	}
	fo := &fileOpt{}
	for _, opt := range opts {
		if opt != nil {
			opt.applyFileOpt(fo)
		}
	}
	if fo.name != "" {
		fri.Name = fo.name
	}
	return fri
}

func InputTextFile(text string, mime string, opts ...FileOpt) Input {
	return InputFile([]byte(text), mime, opts...)
}

// Type assertion helpers for providers
func AsTextInput(input Input) (string, bool) {
	if ti, ok := input.(textInput); ok {
		return ti.Text, true
	}
	return "", false
}

func AsFileInput(input Input) ([]byte, string, string, bool) {
	if fi, ok := input.(fileInput); ok {
		return fi.Data, fi.MIME, fi.Name, true
	}
	return nil, "", "", false
}

func AsFileReaderInput(input Input) (io.Reader, int64, string, string, bool) {
	if fri, ok := input.(fileReaderInput); ok {
		return fri.R, fri.Size, fri.MIME, fri.Name, true
	}
	return nil, 0, "", "", false
}

// OutputPart construction helpers for providers
func NewTextOutputPart(text string) OutputPart {
	return textOutputPart{Text: text}
}

func NewImageOutputPart(data []byte, mime, name string) OutputPart {
	return imageOutputPart{Data: data, MIME: mime, Name: name}
}

func NewJSONOutputPart(jsonData []byte) OutputPart {
	return jsonOutputPart{JSON: jsonData}
}

// Output type checking helpers for providers
func IsTextOutput(output Output) bool {
	_, ok := output.(textOutput)
	return ok
}

func GetImageSpec(output Output) (ImageSpec, bool) {
	if imgOut, ok := output.(imageOutput); ok {
		return imgOut.Spec, true
	}
	return ImageSpec{}, false
}

func GetJSONOutput(output Output) (schema any, strict bool, ok bool) {
	if jsonOut, ok := output.(jsonOutput); ok {
		return jsonOut.Schema, jsonOut.Strict, true
	}
	return nil, false, false
}

//
// Output (single output per request, v1)
//

type Output interface{ isOutput() }

type textOutput struct{}

func (textOutput) isOutput() {}

func OutputText() Output {
	return textOutput{}
}

type ImageSpec struct {
	Count int // default 1
}

type imageOutput struct {
	Spec ImageSpec
}

func (imageOutput) isOutput() {}

func OutputImage(spec ImageSpec) Output {
	return imageOutput{Spec: spec}
}

type jsonOutput struct {
	Schema any
	Strict bool // default true
}

func (jsonOutput) isOutput() {}

func OutputJSON(schema any, opts ...JSONOpt) Output {
	jo := jsonOutput{
		Schema: schema,
		Strict: true, // default
	}
	joOpt := &jsonOpt{}
	for _, opt := range opts {
		if opt != nil {
			opt.applyJSONOpt(joOpt)
		}
	}
	if joOpt.strict != nil {
		jo.Strict = *joOpt.strict
	}
	return jo
}

//
// Output parts
//

type OutputPart interface{ isOutputPart() }

type textOutputPart struct {
	Text string
}

func (textOutputPart) isOutputPart() {}

type imageOutputPart struct {
	Data []byte
	MIME string
	Name string
}

func (imageOutputPart) isOutputPart() {}

type jsonOutputPart struct {
	JSON []byte
}

func (jsonOutputPart) isOutputPart() {}

//
// Request / Response
//

type Request struct {
	Inputs          []Input
	Output          Output
	Model           string    // Optional: explicit model name (highest priority)
	Tier            ModelTier // Optional: tier-based selection (if Model not set)
	ProviderOptions []ProviderOption
	Metadata        map[string]string
}

type Response struct {
	Outputs   []OutputPart
	Usage     Usage
	Provider  ProviderInfo
	RequestID string
	Warnings  []Warning
}

func (r Response) Text() (string, bool) {
	for _, part := range r.Outputs {
		if textPart, ok := part.(textOutputPart); ok {
			return textPart.Text, true
		}
	}
	return "", false
}

func (r Response) Images() ([][]byte, bool) {
	var images [][]byte
	for _, part := range r.Outputs {
		if imgPart, ok := part.(imageOutputPart); ok {
			images = append(images, imgPart.Data)
		}
	}
	return images, len(images) > 0
}

// ImageOutputs returns image output parts with MIME and name information.
func (r Response) ImageOutputs() []ImageOutputInfo {
	var infos []ImageOutputInfo
	for _, part := range r.Outputs {
		if imgPart, ok := part.(imageOutputPart); ok {
			infos = append(infos, ImageOutputInfo(imgPart))
		}
	}
	return infos
}

// ImageOutputInfo contains image data with MIME and optional name.
type ImageOutputInfo struct {
	Data []byte
	MIME string
	Name string
}

func (r Response) DecodeJSON(dst any) error {
	for _, part := range r.Outputs {
		if jsonPart, ok := part.(jsonOutputPart); ok {
			return json.Unmarshal(jsonPart.JSON, dst)
		}
	}
	return NewGrailError(OutputInvalid, "no JSON output part found in response")
}

//
// Provider options (typed per provider package)
//

type ProviderOption interface {
	ApplyProviderOption() // marker method - must be exported for provider packages
}

//
// Options
//

type FileOpt interface{ applyFileOpt(*fileOpt) }
type JSONOpt interface{ applyJSONOpt(*jsonOpt) }

func WithFileName(name string) FileOpt {
	return fileOptFunc(func(fo *fileOpt) {
		fo.name = name
	})
}

func WithStrictJSON(strict bool) JSONOpt {
	return jsonOptFunc(func(jo *jsonOpt) {
		jo.strict = &strict
	})
}

type fileOpt struct{ name string }

type fileOptFunc func(*fileOpt)

func (f fileOptFunc) applyFileOpt(fo *fileOpt) {
	f(fo)
}

type jsonOpt struct{ strict *bool }

type jsonOptFunc func(*jsonOpt)

func (f jsonOptFunc) applyJSONOpt(jo *jsonOpt) {
	f(jo)
}

//
// Client + Provider
//

type Client interface {
	Generate(ctx context.Context, req Request) (Response, error)

	// Explicit helpers for loading remote content (HTTP/S only).
	// These helpers perform network I/O using the client's HTTP client
	// and return concrete Inputs (bytes + MIME).
	InputFileFromURI(ctx context.Context, uri string, opts ...FileOpt) (Input, error)
	InputImageFromURI(ctx context.Context, uri string, opts ...FileOpt) (Input, error)
	InputPDFFromURI(ctx context.Context, uri string, opts ...FileOpt) (Input, error)

	// ListModels returns all available models for the provider and their capabilities.
	// Returns an error if the provider doesn't support model listing.
	ListModels(ctx context.Context) ([]ModelInfo, error)

	// GetModel returns the model matching the given role and tier.
	// Returns an error if no matching model is found.
	GetModel(ctx context.Context, role ModelRole, tier ModelTier) (ModelInfo, error)
}

type ClientOption interface{ applyClientOpt(*clientOpt) }

func WithHTTPClient(hc *http.Client) ClientOption {
	return clientOptFunc(func(co *clientOpt) {
		co.httpClient = hc
	})
}

func WithDownloadLimits(maxBytes int64, timeout time.Duration) ClientOption {
	return clientOptFunc(func(co *clientOpt) {
		co.downloadMaxBytes = maxBytes
		co.downloadTimeout = timeout
	})
}

type Provider interface {
	Name() string
}

// ProviderExecutor is the internal execution seam (implemented by provider packages).
// This is exported so provider packages can implement it, but it's not part of the
// public API contract - users should not implement this directly.
type ProviderExecutor interface {
	Provider
	DoGenerate(ctx context.Context, req Request) (Response, error)
}

type clientOpt struct {
	httpClient       *http.Client
	downloadMaxBytes int64
	downloadTimeout  time.Duration
	logger           *slog.Logger
}

type clientOptFunc func(*clientOpt)

func (f clientOptFunc) applyClientOpt(co *clientOpt) {
	f(co)
}

// LoggerAware is an optional interface for providers to accept a logger from the client.
type LoggerAware interface {
	SetLogger(*slog.Logger)
}

// ModelLister is an optional interface for providers to list available models.
type ModelLister interface {
	ListModels(ctx context.Context) ([]ModelInfo, error)
}

// ModelResolver resolves a role+tier to a model name.
// Providers implement this to support tier-based selection.
type ModelResolver interface {
	ResolveModel(role ModelRole, tier ModelTier) (string, error)
}

// WithLogger sets a custom logger for client-level logs.
func WithLogger(l *slog.Logger) ClientOption {
	return clientOptFunc(func(co *clientOpt) {
		co.logger = l
	})
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
	return clientOptFunc(func(co *clientOpt) {
		handlerOpts := &slog.HandlerOptions{Level: slog.Level(level)}
		switch strings.ToLower(format) {
		case "json":
			co.logger = slog.New(slog.NewJSONHandler(os.Stdout, handlerOpts))
		default:
			co.logger = slog.New(slog.NewTextHandler(os.Stdout, handlerOpts))
		}
	})
}

type client struct {
	provider         ProviderExecutor
	httpClient       *http.Client
	downloadMaxBytes int64
	downloadTimeout  time.Duration
	log              *slog.Logger
}

func NewClient(p Provider, opts ...ClientOption) Client {
	co := &clientOpt{
		httpClient:       http.DefaultClient,
		downloadMaxBytes: 100 * 1024 * 1024, // 100 MB default
		downloadTimeout:  30 * time.Second,
		logger:           slog.Default(),
	}
	for _, opt := range opts {
		if opt != nil {
			opt.applyClientOpt(co)
		}
	}

	executor, ok := p.(ProviderExecutor)
	if !ok {
		// This should not happen in practice, but handle gracefully
		return &client{
			provider:         nil,
			httpClient:       co.httpClient,
			downloadMaxBytes: co.downloadMaxBytes,
			downloadTimeout:  co.downloadTimeout,
			log:              co.logger,
		}
	}

	if la, ok := p.(LoggerAware); ok {
		la.SetLogger(co.logger)
	}

	return &client{
		provider:         executor,
		httpClient:       co.httpClient,
		downloadMaxBytes: co.downloadMaxBytes,
		downloadTimeout:  co.downloadTimeout,
		log:              co.logger,
	}
}

func (c *client) Generate(ctx context.Context, req Request) (Response, error) {
	if err := validateRequest(req); err != nil {
		return Response{}, err
	}

	if c.provider == nil {
		return Response{}, NewGrailError(Internal, "provider executor not available")
	}

	// Resolve model selection: Model > Tier > Provider default
	if req.Model == "" && req.Tier != "" {
		role := roleFromOutput(req.Output)
		if resolver, ok := c.provider.(ModelResolver); ok {
			resolved, err := resolver.ResolveModel(role, req.Tier)
			if err != nil {
				return Response{}, NewGrailError(InvalidArgument, fmt.Sprintf("failed to resolve model for role=%s tier=%s: %v", role, req.Tier, err)).WithCause(err)
			}
			req.Model = resolved
		}
	}

	if c.log != nil {
		c.log.Info("generate request",
			slog.Int("inputs", len(req.Inputs)),
			slog.String("output_type", getOutputType(req.Output)),
			slog.String("model", req.Model),
		)
	}

	return c.provider.DoGenerate(ctx, req)
}

func (c *client) ListModels(ctx context.Context) ([]ModelInfo, error) {
	if c.provider == nil {
		return nil, NewGrailError(Internal, "provider executor not available")
	}

	lister, ok := c.provider.(ModelLister)
	if !ok {
		return nil, NewGrailError(Unsupported, fmt.Sprintf("provider %s does not support model listing", c.provider.Name()))
	}

	return lister.ListModels(ctx)
}

func (c *client) GetModel(ctx context.Context, role ModelRole, tier ModelTier) (ModelInfo, error) {
	models, err := c.ListModels(ctx)
	if err != nil {
		return ModelInfo{}, err
	}

	for _, m := range models {
		if m.Role == role && m.Tier == tier {
			return m, nil
		}
	}

	return ModelInfo{}, NewGrailError(Unsupported, fmt.Sprintf("no model found for role=%s tier=%s", role, tier))
}

func (c *client) InputFileFromURI(ctx context.Context, uri string, opts ...FileOpt) (Input, error) {
	return c.downloadFile(ctx, uri, "", opts...)
}

func (c *client) InputImageFromURI(ctx context.Context, uri string, opts ...FileOpt) (Input, error) {
	return c.downloadFile(ctx, uri, "image/", opts...)
}

func (c *client) InputPDFFromURI(ctx context.Context, uri string, opts ...FileOpt) (Input, error) {
	return c.downloadFile(ctx, uri, "application/pdf", opts...)
}

func (c *client) downloadFile(ctx context.Context, uri string, expectedMIME string, opts ...FileOpt) (Input, error) {
	ctx, cancel := context.WithTimeout(ctx, c.downloadTimeout)
	defer cancel()

	req, err := http.NewRequestWithContext(ctx, "GET", uri, nil)
	if err != nil {
		return nil, NewGrailError(InvalidArgument, fmt.Sprintf("invalid URI: %v", err)).WithCause(err)
	}

	resp, err := c.httpClient.Do(req)
	if err != nil {
		if errors.Is(err, context.DeadlineExceeded) {
			return nil, NewGrailError(Timeout, "download timeout").WithCause(err).WithRetryable(true)
		}
		return nil, NewGrailError(Unavailable, fmt.Sprintf("download failed: %v", err)).WithCause(err).WithRetryable(true)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, NewGrailError(Unavailable, fmt.Sprintf("download failed with status %d", resp.StatusCode))
	}

	// Check content length
	if resp.ContentLength > c.downloadMaxBytes {
		return nil, NewGrailError(InvalidArgument, fmt.Sprintf("file size %d exceeds maximum %d bytes", resp.ContentLength, c.downloadMaxBytes))
	}

	// Read with limit
	limitedReader := io.LimitReader(resp.Body, c.downloadMaxBytes+1)
	data, err := io.ReadAll(limitedReader)
	if err != nil {
		return nil, NewGrailError(Unavailable, fmt.Sprintf("failed to read response: %v", err)).WithCause(err)
	}

	if int64(len(data)) > c.downloadMaxBytes {
		return nil, NewGrailError(InvalidArgument, fmt.Sprintf("file size exceeds maximum %d bytes", c.downloadMaxBytes))
	}

	mime := resp.Header.Get("Content-Type")
	if mime == "" {
		mime = "application/octet-stream"
	}

	// Validate MIME if expected
	if expectedMIME != "" {
		if expectedMIME == "application/pdf" {
			if mime != "application/pdf" {
				return nil, NewGrailError(InvalidArgument, fmt.Sprintf("expected PDF, got %s", mime))
			}
		} else if strings.HasPrefix(expectedMIME, "image/") {
			if !strings.HasPrefix(mime, "image/") {
				return nil, NewGrailError(InvalidArgument, fmt.Sprintf("expected image, got %s", mime))
			}
		}
	}

	// Apply file options
	fo := &fileOpt{}
	for _, opt := range opts {
		if opt != nil {
			opt.applyFileOpt(fo)
		}
	}

	return InputFile(data, mime, opts...), nil
}

//
// Local filesystem helpers (explicit I/O)
//

func InputFileFromPath(path string, opts ...FileOpt) (Input, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, NewGrailError(InvalidArgument, fmt.Sprintf("failed to read file: %v", err)).WithCause(err)
	}

	// Try to detect MIME from extension
	mime := detectMIMEFromPath(path)
	return InputFile(data, mime, opts...), nil
}

func InputPDFFromPath(path string, opts ...FileOpt) (Input, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, NewGrailError(InvalidArgument, fmt.Sprintf("failed to read file: %v", err)).WithCause(err)
	}
	return InputPDF(data, opts...), nil
}

func InputImageFromPath(path string, opts ...FileOpt) (Input, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, NewGrailError(InvalidArgument, fmt.Sprintf("failed to read file: %v", err)).WithCause(err)
	}
	return InputImage(data, opts...), nil
}

//
// Validation
//

const (
	MaxPDFSize  = 50 * 1024 * 1024  // 50 MB
	MaxFileSize = 100 * 1024 * 1024 // 100 MB
)

func validateRequest(req Request) error {
	if len(req.Inputs) == 0 {
		return NewGrailError(InvalidArgument, "inputs must not be empty")
	}

	if req.Output == nil {
		return NewGrailError(InvalidArgument, "output must be specified")
	}

	for i, input := range req.Inputs {
		switch v := input.(type) {
		case fileInput:
			if len(v.Data) == 0 {
				return NewGrailError(InvalidArgument, fmt.Sprintf("input %d: file data is empty", i))
			}
			if len(v.Data) > MaxFileSize {
				return NewGrailError(InvalidArgument, fmt.Sprintf("input %d: file size %d exceeds maximum %d bytes", i, len(v.Data), MaxFileSize))
			}

			// Handle empty MIME (e.g., from ImageInput - means it should be an image)
			mime := v.MIME
			if mime == "" {
				// Try to sniff MIME from data
				mime = sniffImageMIME(v.Data)
				if mime == "" || !strings.HasPrefix(mime, "image/") {
					// Empty MIME from ImageInput means it should be an image
					return NewGrailError(InvalidArgument, fmt.Sprintf("input %d: expected image/*, got %s", i, mime))
				}
			}

			// Special validation for PDFs
			if mime == "application/pdf" {
				if len(v.Data) > MaxPDFSize {
					return NewGrailError(InvalidArgument, fmt.Sprintf("input %d: PDF file size %d exceeds maximum %d bytes", i, len(v.Data), MaxPDFSize))
				}
			}
		case textInput:
			// Text input is always valid
		case fileReaderInput:
			if v.MIME == "" {
				return NewGrailError(InvalidArgument, fmt.Sprintf("input %d: MIME type must be specified", i))
			}
			if v.Size > 0 && v.Size > MaxFileSize {
				return NewGrailError(InvalidArgument, fmt.Sprintf("input %d: file size %d exceeds maximum %d bytes", i, v.Size, MaxFileSize))
			}
		}
	}

	return nil
}

//
// Helpers
//

func getOutputType(output Output) string {
	switch output.(type) {
	case textOutput:
		return "text"
	case imageOutput:
		return "image"
	case jsonOutput:
		return "json"
	default:
		return "unknown"
	}
}

// roleFromOutput determines the ModelRole from the Output type.
func roleFromOutput(output Output) ModelRole {
	if IsTextOutput(output) {
		return ModelRoleText
	}
	if _, isImage := GetImageSpec(output); isImage {
		return ModelRoleImage
	}
	// JSON output also uses text models
	return ModelRoleText
}

// SniffImageMIME detects image MIME type from magic bytes.
// It supports PNG, JPEG, GIF, and WebP formats.
func SniffImageMIME(data []byte) string {
	if len(data) < 4 {
		return ""
	}

	// Check magic bytes for common image formats
	if len(data) >= 4 && string(data[0:4]) == "\x89PNG" {
		return "image/png"
	}
	if len(data) >= 2 && data[0] == 0xFF && data[1] == 0xD8 {
		return "image/jpeg"
	}
	if len(data) >= 6 && (string(data[0:6]) == "GIF87a" || string(data[0:6]) == "GIF89a") {
		return "image/gif"
	}
	if len(data) >= 12 && string(data[0:4]) == "RIFF" && string(data[8:12]) == "WEBP" {
		return "image/webp"
	}

	return ""
}

func sniffImageMIME(data []byte) string {
	return SniffImageMIME(data)
}

func detectMIMEFromPath(path string) string {
	ext := strings.ToLower(path[strings.LastIndex(path, "."):])
	switch ext {
	case ".pdf":
		return "application/pdf"
	case ".png":
		return "image/png"
	case ".jpg", ".jpeg":
		return "image/jpeg"
	case ".gif":
		return "image/gif"
	case ".webp":
		return "image/webp"
	case ".txt":
		return "text/plain"
	case ".md", ".markdown":
		return "text/markdown"
	case ".html", ".htm":
		return "text/html"
	case ".json":
		return "application/json"
	case ".go":
		return "text/x-go"
	default:
		return "application/octet-stream"
	}
}

// Pointer is a helper to take the address of a literal value (e.g., grail.Pointer(0.0)).
func Pointer[T any](v T) *T {
	return &v
}
