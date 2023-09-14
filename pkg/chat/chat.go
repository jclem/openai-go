// Package chat provides a chat client for the OpenAI API.
package chat

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"

	"github.com/jclem/openai-go/internal/service"
	"github.com/jclem/sseparser"
)

// A Message is a message in a chat prompt.
type Message struct {
	Role         string        `json:"role"`
	Content      *string       `json:"content"`
	Name         *string       `json:"name,omitempty"`
	FunctionCall *FunctionCall `json:"function_call,omitempty"`
}

// MessageOpt is a functional option for configuring a message.
type MessageOpt func(*Message)

// WithMessageContent sets the content for the message.
func WithMessageContent(content string) MessageOpt {
	return func(m *Message) {
		m.Content = &content
	}
}

// WithMessageName sets the name for the message.
func WithMessageName(name string) MessageOpt {
	return func(m *Message) {
		m.Name = &name
	}
}

// WithMessageFunctionCall sets the function call for the message.
func WithMessageFunctionCall(functionCall FunctionCall) MessageOpt {
	return func(m *Message) {
		m.FunctionCall = &functionCall
	}
}

// NewMessage creates a new message.
func NewMessage(role string, opts ...MessageOpt) Message {
	m := Message{Role: role}

	for _, opt := range opts {
		opt(&m)
	}

	return m
}

// A FunctionCall represents a request to call a function.
type FunctionCall struct {
	Name      string          `json:"name"`
	Arguments json.RawMessage `json:"arguments"`
}

type completionRequest struct {
	apiKey string

	Model            string               `json:"model"`
	Messages         []Message            `json:"messages"`
	Functions        []FunctionDefinition `json:"functions,omitempty"`
	FunctionCall     *functionCallSetting `json:"function_call,omitempty"`
	Temperature      *float64             `json:"temperature,omitempty"`
	TopP             *float64             `json:"top_p,omitempty"`
	N                *int                 `json:"n,omitempty"`
	Stream           *bool                `json:"stream,omitempty"`
	Stop             []string             `json:"stop,omitempty"`
	MaxTokens        *int                 `json:"max_tokens,omitempty"`
	PresencePenalty  *float64             `json:"presence_penalty,omitempty"`
	FrequencyPenalty *float64             `json:"frequency_penalty,omitempty"`
	LogitBias        map[string]float64   `json:"logit_bias,omitempty"`
	User             *string              `json:"user,omitempty"`
}

// A FunctionDefinition represents a function definition.
type FunctionDefinition struct {
	Name        string  `json:"name"`
	Description *string `json:"description,omitempty"`
	Parameters  any     `json:"parameters"`
}

// FunctionDefinitionOpt is a functional option for configuring a function definition.
type FunctionDefinitionOpt func(*FunctionDefinition)

// WithFunctionDescription sets the description for the function definition.
func WithFunctionDescription(description string) FunctionDefinitionOpt {
	return func(f *FunctionDefinition) {
		f.Description = &description
	}
}

// NewFunctionDefinition creates a new function definition.
func NewFunctionDefinition(name string, parameters any, opts ...FunctionDefinitionOpt) FunctionDefinition {
	f := FunctionDefinition{Name: name, Parameters: parameters}

	for _, opt := range opts {
		opt(&f)
	}

	return f
}

type functionCallSetting struct {
	Value string
	Name  string
}

// ErrInvalidFunctionCallSetting is returned when a function call setting is invalid.
var ErrInvalidFunctionCallSetting = errors.New("function call setting must have a value or a name")

func (f functionCallSetting) MarshalJSON() ([]byte, error) {
	if f.Value != "" {
		b, err := json.Marshal(f.Value)
		if err != nil {
			return nil, fmt.Errorf("error marshaling function call setting value: %w", err)
		}

		return b, nil
	}

	if f.Name != "" {
		obj := map[string]string{"name": f.Name}

		b, err := json.Marshal(obj)
		if err != nil {
			return nil, fmt.Errorf("error marshaling function call setting name: %w", err)
		}

		return b, nil
	}

	return nil, ErrInvalidFunctionCallSetting
}

func (f *functionCallSetting) UnmarshalJSON(b []byte) error {
	var v string
	if err := json.Unmarshal(b, &v); err == nil {
		f.Value = v

		return nil
	}

	var obj map[string]string
	if err := json.Unmarshal(b, &obj); err == nil {
		f.Name = obj["name"]

		return nil
	}

	return ErrInvalidFunctionCallSetting
}

// A CompletionResponse defines a response to a request to get a completion.
type CompletionResponse struct {
	ID      string             `json:"id"`
	Object  string             `json:"object"`
	Created int64              `json:"created"`
	Model   string             `json:"model"`
	Choices []CompletionChoice `json:"choices"`
	Usage   Usage              `json:"usage"`
}

// GetChoiceAt returns the choice at the given index.
//
// If the index is out of bounds, it returns false (but a negative index will
// panic).
func (r *CompletionResponse) GetChoiceAt(index int) (CompletionChoice, bool) {
	if index >= len(r.Choices) {
		return CompletionChoice{}, false
	}

	return r.Choices[index], true
}

// GetContentAt returns the content of the choice at the given index.
func (r *CompletionResponse) GetContentAt(index int) (string, bool) {
	choice, ok := r.GetChoiceAt(index)
	if !ok {
		return "", false
	}

	if choice.Message.Content == nil {
		return "", false
	}

	return *choice.Message.Content, true
}

// GetFunctionCallAt returns the function call of the choice at the given index.
func (r *CompletionResponse) GetFunctionCallAt(index int) (FunctionCall, bool) {
	choice, ok := r.GetChoiceAt(index)
	if !ok {
		return FunctionCall{}, false
	}

	if choice.Message.FunctionCall == nil {
		return FunctionCall{}, false
	}

	return *choice.Message.FunctionCall, true
}

// A CompletionChoice defines a completion choice in a completion response.
type CompletionChoice struct {
	Index        int     `json:"index"`
	Message      Message `json:"message"`
	FinishReason string  `json:"finish_reason"`
}

// A Usage defines usage statistics.
type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// CreateCompletionOpt is a functional option for configuring a completion request.
type CreateCompletionOpt func(*completionRequest)

// WithFunctions sets the functions for the completion request.
func WithFunctions(functions ...FunctionDefinition) CreateCompletionOpt {
	return func(r *completionRequest) {
		r.Functions = functions
	}
}

// WithFunctionCallBySetting sets the function call for the completion request.
//
// Use this option if you're passing a predefined value such as "none" or "auto".
func WithFunctionCallBySetting(value string) CreateCompletionOpt {
	return func(r *completionRequest) {
		r.FunctionCall = &functionCallSetting{Value: value}
	}
}

// WithFunctionCallByName sets the function call for the completion request.
//
// Use this option if you're passing a specific function by name.
func WithFunctionCallByName(name string) CreateCompletionOpt {
	return func(r *completionRequest) {
		r.FunctionCall = &functionCallSetting{Name: name}
	}
}

// WithTemperature sets the temperature for the completion request.
func WithTemperature(temperature float64) CreateCompletionOpt {
	return func(r *completionRequest) {
		r.Temperature = &temperature
	}
}

// WithTopP sets the top p for the completion request.
func WithTopP(topP float64) CreateCompletionOpt {
	return func(r *completionRequest) {
		r.TopP = &topP
	}
}

// WithN sets the n for the completion request.
func WithN(n int) CreateCompletionOpt {
	return func(r *completionRequest) {
		r.N = &n
	}
}

// WithStream sets the stream for the completion request.
func WithStream(stream bool) CreateCompletionOpt {
	return func(r *completionRequest) {
		r.Stream = &stream
	}
}

// WithStop sets the stop for the completion request.
func WithStop(stop ...string) CreateCompletionOpt {
	return func(r *completionRequest) {
		r.Stop = stop
	}
}

// WithMaxTokens sets the max tokens for the completion request.
func WithMaxTokens(maxTokens int) CreateCompletionOpt {
	return func(r *completionRequest) {
		r.MaxTokens = &maxTokens
	}
}

// WithPresencePenalty sets the presence penalty for the completion request.
func WithPresencePenalty(presencePenalty float64) CreateCompletionOpt {
	return func(r *completionRequest) {
		r.PresencePenalty = &presencePenalty
	}
}

// WithFrequencyPenalty sets the frequency penalty for the completion request.
func WithFrequencyPenalty(frequencyPenalty float64) CreateCompletionOpt {
	return func(r *completionRequest) {
		r.FrequencyPenalty = &frequencyPenalty
	}
}

// WithLogitBias sets the logit bias for the completion request.
func WithLogitBias(logitBias map[string]float64) CreateCompletionOpt {
	return func(r *completionRequest) {
		r.LogitBias = logitBias
	}
}

// WithUser sets the user for the completion request.
func WithUser(user string) CreateCompletionOpt {
	return func(r *completionRequest) {
		r.User = &user
	}
}

// WithAPIKey sets the API key for the completion request.
func WithAPIKey(apiKey string) CreateCompletionOpt {
	return func(r *completionRequest) {
		r.apiKey = apiKey
	}
}

// Service is a service wrapping an OpenAI-compatible completions API.
type Service service.Service

// CreateCompletion implements the OpenAIClient interface using an HTTP request.
//
// It returns a parsed completion response.
func (h *Service) CreateCompletion(
	ctx context.Context,
	model string,
	messages []Message,
	opts ...CreateCompletionOpt,
) (*CompletionResponse, error) {
	req := completionRequest{Model: model, Messages: messages}

	for _, opt := range opts {
		opt(&req)
	}

	httpReq, err := h.Client.NewRequestWithContext(ctx, http.MethodPost, "/chat/completions", req,
		service.WithAPIKey(req.apiKey))
	if err != nil {
		return nil, fmt.Errorf("error creating HTTP request: %w", err)
	}

	var resp CompletionResponse
	if _, err := h.Client.Do(httpReq, &resp); err != nil { //nolint: bodyclose // False positive.
		return nil, fmt.Errorf("error performing HTTP request: %w", err)
	}

	return &resp, nil
}

// CreateStreamingCompletion implements the OpenAIClient interface using an HTTP request.
//
// It returns a StreamingCompletionResponse. The caller is responsible for
// closing the response body included on the response struct.
func (h *Service) CreateStreamingCompletion(
	ctx context.Context,
	model string,
	messages []Message,
	opts ...CreateCompletionOpt,
) (*StreamingCompletionResponse, error) {
	req := completionRequest{Model: model, Messages: messages}

	opts = append(opts, WithStream(true))

	for _, opt := range opts {
		opt(&req)
	}

	httpReq, err := h.Client.NewRequestWithContext(ctx, http.MethodPost, "/chat/completions", req,
		service.WithAPIKey(req.apiKey))
	if err != nil {
		return nil, fmt.Errorf("error creating HTTP request: %w", err)
	}

	httpResp, err := h.Client.Do(httpReq, nil) //nolint: bodyclose // False positive.
	if err != nil {
		return nil, fmt.Errorf("error performing HTTP request: %w", err)
	}

	return newStreamingCompletionResponse(httpResp.Body), nil
}

type streamingCompletionEvent struct {
	Data StreamingCompletionObject `sse:"data"`
}

// A StreamingCompletionObject is a single chunk of a streaming chat
// completion response.
type StreamingCompletionObject struct {
	ID      string                      `json:"id"`
	Object  string                      `json:"object"`
	Created int64                       `json:"created"`
	Model   string                      `json:"model"`
	Choices []StreamingCompletionChoice `json:"choices"`
}

// GetChoiceAt returns the choice at the given index.
func (o *StreamingCompletionObject) GetChoiceAt(index int) (StreamingCompletionChoice, bool) {
	if index >= len(o.Choices) {
		return StreamingCompletionChoice{}, false
	}

	return o.Choices[index], true
}

// GetContentAt returns the content of the choice at the given index.
func (o *StreamingCompletionObject) GetContentAt(index int) (string, bool) {
	choice, ok := o.GetChoiceAt(index)
	if !ok {
		return "", false
	}

	if choice.Delta.Content == nil {
		return "", false
	}

	return *choice.Delta.Content, true
}

// GetFunctionCallAt returns the function call of the choice at the given index.
func (o *StreamingCompletionObject) GetFunctionCallAt(index int) (FunctionCall, bool) {
	choice, ok := o.GetChoiceAt(index)
	if !ok {
		return FunctionCall{}, false
	}

	if choice.Delta.FunctionCall == nil {
		return FunctionCall{}, false
	}

	return *choice.Delta.FunctionCall, true
}

const streamDoneString = "[DONE]"

// ErrStreamDone is returned when the stream is done (marked by "[DONE]").
var ErrStreamDone = errors.New("completion stream is done")

// UnmarshalSSEValue implements sseparser.UnmarshalerSSEValue.
func (o *StreamingCompletionObject) UnmarshalSSEValue(v string) error {
	if v == streamDoneString {
		return ErrStreamDone
	}

	if err := json.Unmarshal([]byte(v), o); err != nil {
		return fmt.Errorf("error unmarshaling streaming completion object: %w", err)
	}

	return nil
}

// A StreamingCompletionChoice is a single choice in a streaming chat
// completion response.
type StreamingCompletionChoice struct {
	Index        int                      `json:"index"`
	Delta        StreamingCompletionDelta `json:"delta"`
	FinishReason *string                  `json:"finish_reason"`
}

// A StreamingCompletionDelta is a single delta in a streaming chat
// completion response.
type StreamingCompletionDelta struct {
	Role         string        `json:"role"`
	Content      *string       `json:"content"`
	FunctionCall *FunctionCall `json:"function_call"`
}

// A StreamingCompletionResponse is a streaming response to a request to get
// a completion. It reads an io.ReadCloser and emits
// StreamingCompletionObjects.
//
// The caller is responsible for closing the stream (`stream.Close()`).
type StreamingCompletionResponse struct {
	closer  io.Closer
	scanner *sseparser.StreamScanner
}

// Next returns the next object in the streaming response.
//
// When the stream is complete, it returns nil, nil.
func (s *StreamingCompletionResponse) Next() (*StreamingCompletionObject, error) {
	var evt streamingCompletionEvent

	_, err := s.scanner.UnmarshalNext(&evt)
	if err != nil {
		if errors.Is(err, sseparser.ErrStreamEOF) {
			return nil, fmt.Errorf("stream ended before [DONE]: %w", err)
		}

		if errors.Is(err, ErrStreamDone) {
			return nil, ErrStreamDone
		}

		return nil, fmt.Errorf("error reading next object from stream: %w", err)
	}

	return &evt.Data, nil
}

// Close closes the stream.
func (s *StreamingCompletionResponse) Close() error {
	if err := s.closer.Close(); err != nil {
		return fmt.Errorf("error closing stream: %w", err)
	}

	return nil
}

func newStreamingCompletionResponse(rc io.ReadCloser) *StreamingCompletionResponse {
	scanner := sseparser.NewStreamScanner(rc)

	return &StreamingCompletionResponse{closer: rc, scanner: scanner}
}
