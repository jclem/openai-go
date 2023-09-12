// Package openai provides a client for the OpenAI API.
package openai

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"

	"github.com/jclem/sseparser"
)

// An HTTPDoer is an HTTP client that can perform HTTP requests.
type HTTPDoer interface {
	Do(*http.Request) (*http.Response, error)
}

// An OpenAIClient is a client for the OpenAI API.
type OpenAIClient interface {
	// DoChatCompletion generates a completion from a chat prompt. It's a
	// low-level API that returns an HTTP response directly.
	DoChatCompletion(ctx context.Context, model string, messages []Message, opts ...CreateChatCompletionOpt) (*http.Response, error)

	// CreateChatCompletion generates a completion from a chat prompt. It returns a completion response.
	CreateChatCompletion(ctx context.Context, model string, messages []Message, opts ...CreateChatCompletionOpt) (*ChatCompletionResponse, error)

	// CreateStreamingChatCompletion generates a completion from a chat prompt. It returns a streaming completion response.
	CreateStreamingChatCompletion(ctx context.Context, model string, messages []Message, opts ...CreateChatCompletionOpt) (*StreamingChatCompletionResponse, error)
}

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

type chatCompletionRequest struct {
	apiKey string `json:"-"`

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

	return nil, errors.New("function call setting must have a value or a name")
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

	return errors.New(`function call setting must be a string or a map with a "name" key and string value`)
}

// A ChatCompletionResponse defines a response to a request to get a chat completion.
type ChatCompletionResponse struct {
	ID      string                 `json:"id"`
	Object  string                 `json:"object"`
	Created int64                  `json:"created"`
	Model   string                 `json:"model"`
	Choices []ChatCompletionChoice `json:"choices"`
	Usage   Usage                  `json:"usage"`
}

// GetChoiceAt returns the choice at the given index.
func (r *ChatCompletionResponse) GetChoiceAt(index int) (ChatCompletionChoice, bool) {
	if index < 0 || index >= len(r.Choices) {
		return ChatCompletionChoice{}, false
	}

	return r.Choices[index], true
}

// GetContentAt returns the content of the choice at the given index.
func (r *ChatCompletionResponse) GetContentAt(index int) (string, bool) {
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
func (r *ChatCompletionResponse) GetFunctionCallAt(index int) (FunctionCall, bool) {
	choice, ok := r.GetChoiceAt(index)
	if !ok {
		return FunctionCall{}, false
	}

	if choice.Message.FunctionCall == nil {
		return FunctionCall{}, false
	}

	return *choice.Message.FunctionCall, true
}

// A ChatCompletionChoice defines a completion choice in a chat completion response.
type ChatCompletionChoice struct {
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

// CreateChatCompletionOpt is a functional option for configuring a chat completion request.
type CreateChatCompletionOpt func(*chatCompletionRequest)

// WithFunctions sets the functions for the chat completion request.
func WithFunctions(functions ...FunctionDefinition) CreateChatCompletionOpt {
	return func(r *chatCompletionRequest) {
		r.Functions = functions
	}
}

// WithFunctionCallBySetting sets the function call for the chat completion request.
//
// Use this option if you're passing a predefined value such as "none" or "auto".
func WithFunctionCallBySetting(value string) CreateChatCompletionOpt {
	return func(r *chatCompletionRequest) {
		r.FunctionCall = &functionCallSetting{Value: value}
	}
}

// WithFunctionCallByName sets the function call for the chat completion request.
//
// Use this option if you're passing a specific function by name.
func WithFunctionCallByName(name string) CreateChatCompletionOpt {
	return func(r *chatCompletionRequest) {
		r.FunctionCall = &functionCallSetting{Name: name}
	}
}

// WithTemperature sets the temperature for the chat completion request.
func WithTemperature(temperature float64) CreateChatCompletionOpt {
	return func(r *chatCompletionRequest) {
		r.Temperature = &temperature
	}
}

// WithTopP sets the top p for the chat completion request.
func WithTopP(topP float64) CreateChatCompletionOpt {
	return func(r *chatCompletionRequest) {
		r.TopP = &topP
	}
}

// WithN sets the n for the chat completion request.
func WithN(n int) CreateChatCompletionOpt {
	return func(r *chatCompletionRequest) {
		r.N = &n
	}
}

// WithStream sets the stream for the chat completion request.
func WithStream(stream bool) CreateChatCompletionOpt {
	return func(r *chatCompletionRequest) {
		r.Stream = &stream
	}
}

// WithStop sets the stop for the chat completion request.
func WithStop(stop ...string) CreateChatCompletionOpt {
	return func(r *chatCompletionRequest) {
		r.Stop = stop
	}
}

// WithMaxTokens sets the max tokens for the chat completion request.
func WithMaxTokens(maxTokens int) CreateChatCompletionOpt {
	return func(r *chatCompletionRequest) {
		r.MaxTokens = &maxTokens
	}
}

// WithPresencePenalty sets the presence penalty for the chat completion request.
func WithPresencePenalty(presencePenalty float64) CreateChatCompletionOpt {
	return func(r *chatCompletionRequest) {
		r.PresencePenalty = &presencePenalty
	}
}

// WithFrequencyPenalty sets the frequency penalty for the chat completion request.
func WithFrequencyPenalty(frequencyPenalty float64) CreateChatCompletionOpt {
	return func(r *chatCompletionRequest) {
		r.FrequencyPenalty = &frequencyPenalty
	}
}

// WithLogitBias sets the logit bias for the chat completion request.
func WithLogitBias(logitBias map[string]float64) CreateChatCompletionOpt {
	return func(r *chatCompletionRequest) {
		r.LogitBias = logitBias
	}
}

// WithUser sets the user for the chat completion request.
func WithUser(user string) CreateChatCompletionOpt {
	return func(r *chatCompletionRequest) {
		r.User = &user
	}
}

// WithAPIKey sets the API key for the chat completion request.
func WithAPIKey(apiKey string) CreateChatCompletionOpt {
	return func(r *chatCompletionRequest) {
		r.apiKey = apiKey
	}
}

// An ErrUnexpectedStatusCode is an error returned when the HTTP response has an
// unexpected status code.
//
// In includes the expected and actual codes, as well as the response.
type ErrUnexpectedStatusCode struct {
	Expected int
	Actual   int
	Response *http.Response
}

// Error implements the error interface.
func (e ErrUnexpectedStatusCode) Error() string {
	return fmt.Sprintf("unexpected status code %d (expected %d)", e.Actual, e.Expected)
}

// HTTPClient is an HTTP client for the OpenAI API.
type HTTPClient struct {
	key  string
	http HTTPDoer
}

// DoChatCompletion implements the OpenAIClient interface using an HTTP request.
//
// If the request returns a non-200 status code, an ErrUnexpectedStatusCode is
// returned.
//
// The caller is responsible for closing the response body included on the
// response struct.
func (h *HTTPClient) DoChatCompletion(ctx context.Context, model string, messages []Message, opts ...CreateChatCompletionOpt) (*http.Response, error) {
	req := chatCompletionRequest{Model: model, Messages: messages}

	for _, opt := range opts {
		opt(&req)
	}

	b, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("error marshaling chat completion request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, "https://api.openai.com/v1/chat/completions", bytes.NewReader(b))
	if err != nil {
		return nil, fmt.Errorf("error creating HTTP request: %w", err)
	}

	var apiKey string
	if req.apiKey != "" {
		apiKey = req.apiKey
	} else {
		apiKey = h.key
	}

	httpReq.Header.Set("Authorization", fmt.Sprintf("Bearer %s", apiKey))
	httpReq.Header.Set("Content-Type", "application/json")

	httpResp, err := h.http.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("error performing HTTP request: %w", err)
	}

	if httpResp.StatusCode != http.StatusOK {
		return nil, ErrUnexpectedStatusCode{Expected: http.StatusOK, Actual: httpResp.StatusCode, Response: httpResp}
	}

	return httpResp, nil
}

// CreateChatCompletion implements the OpenAIClient interface using an HTTP request.
//
// It returns a parsed completion response.
func (h *HTTPClient) CreateChatCompletion(ctx context.Context, model string, messages []Message, opts ...CreateChatCompletionOpt) (resp *ChatCompletionResponse, err error) {
	httpResp, err := h.DoChatCompletion(ctx, model, messages, opts...)
	if err != nil {
		return nil, err
	}

	defer func() {
		if closeErr := httpResp.Body.Close(); closeErr != nil && err == nil {
			err = fmt.Errorf("error closing HTTP response body: %w", closeErr)
		}
	}()

	if err := json.NewDecoder(httpResp.Body).Decode(&resp); err != nil {
		return nil, fmt.Errorf("error decoding chat completion response: %w", err)
	}

	return resp, nil
}

// CreateStreamingChatCompletion implements the OpenAIClient interface using an HTTP request.
//
// It returns a StreamingChatCompletionResponse. The caller is responsible for
// closing the response body included on the response struct.
func (h *HTTPClient) CreateStreamingChatCompletion(ctx context.Context, model string, messages []Message, opts ...CreateChatCompletionOpt) (*StreamingChatCompletionResponse, error) {
	opts = append(opts, WithStream(true))
	httpResp, err := h.DoChatCompletion(ctx, model, messages, opts...)
	if err != nil {
		return nil, err
	}

	return newStreamingChatCompletionResponse(httpResp.Body), nil
}

type streamingChatCompletionEvent struct {
	Data StreamingChatCompletionObject `sse:"data"`
}

// A StreamingChatCompletionObject is a single chunk of a streaming chat
// completion response.
type StreamingChatCompletionObject struct {
	ID      string                          `json:"id"`
	Object  string                          `json:"object"`
	Created int64                           `json:"created"`
	Model   string                          `json:"model"`
	Choices []StreamingChatCompletionChoice `json:"choices"`
}

const streamDoneString = "[DONE]"

var errStreamIsDone = errors.New("completion stream is done")

func (o *StreamingChatCompletionObject) UnmarshalSSEValue(v string) error {
	if v == streamDoneString {
		return errStreamIsDone
	}

	if err := json.Unmarshal([]byte(v), o); err != nil {
		return fmt.Errorf("error unmarshaling streaming chat completion object: %w", err)
	}

	return nil
}

// A StreamingChatCompletionChoice is a single choice in a streaming chat
// completion response.
type StreamingChatCompletionChoice struct {
	Index        int                          `json:"index"`
	Delta        StreamingChatCompletionDelta `json:"delta"`
	FinishReason *string                      `json:"finish_reason"`
}

// A StreamingChatCompletionDelta is a single delta in a streaming chat
// completion response.
type StreamingChatCompletionDelta struct {
	Role         string       `json:"role"`
	Content      *string      `json:"content"`
	FunctionCall FunctionCall `json:"function_call"`
}

// A StreamingChatCompletionResponse is a streaming response to a request to get
// a chat completion. It reads an io.ReadCloser and emits
// StreamingChatCompletionObjects.
//
// The caller is responsible for closing the ReadCloser.
type StreamingChatCompletionResponse struct {
	ReadCloser io.ReadCloser
	scanner    *sseparser.StreamScanner
}

// Next returns the next object in the streaming response.
//
// When the stream is complete, it returns nil, nil.
func (s *StreamingChatCompletionResponse) Next() (*StreamingChatCompletionObject, error) {
	var evt streamingChatCompletionEvent
	_, err := s.scanner.UnmarshalNext(&evt)
	if err != nil {
		if errors.Is(err, sseparser.ErrStreamEOF) {
			return nil, nil
		}

		if errors.Is(err, errStreamIsDone) {
			return nil, nil
		}

		return nil, fmt.Errorf("error reading next object from stream: %w", err)
	}

	return &evt.Data, nil
}

func newStreamingChatCompletionResponse(rc io.ReadCloser) *StreamingChatCompletionResponse {
	scanner := sseparser.NewStreamScanner(rc)
	return &StreamingChatCompletionResponse{ReadCloser: rc, scanner: scanner}
}

// NewHTTPClient creates a new HTTP client for the OpenAI API.
func NewHTTPClient(opts ...Opt) *HTTPClient {
	client := HTTPClient{
		http: http.DefaultClient,
	}

	for _, opt := range opts {
		opt(&client)
	}

	return &client
}

// Opt is a functional option for configuring the HTTP client.
type Opt func(*HTTPClient)

// WithKey sets the API key for the HTTP client.
func WithKey(key string) Opt {
	return func(c *HTTPClient) {
		c.key = key
	}
}

// WithHTTPDoer sets the HTTP round tripper for the HTTP client.
func WithHTTPDoer(doer HTTPDoer) Opt {
	return func(c *HTTPClient) {
		c.http = doer
	}
}
