// Package embeddings provides a embeddings client for the OpenAI API.
package embeddings

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"

	"github.com/jclem/openai-go/internal/httputil"
)

// An EmbeddingsClient is a chat client for the OpenAI API.
type EmbeddingsClient interface {
	// CreateEmbeddings generates embeddings from a list of inputs.
	CreateEmbeddings(ctx context.Context, model string, inputs []string, opts ...CreateEmbeddingsOpt) (*EmbeddingsResponse, error)
}

type embeddingRequest struct {
	apiKey string `json:"-"`

	Model string   `json:"model"`
	Input []string `json:"input"`
	User  *string  `json:"user,omitempty"`
}

// CreateEmbeddingsOpt is a functional option for configuring an embedding request.
type CreateEmbeddingsOpt func(*embeddingRequest)

// WithUser sets the user ID for the request.
func WithUser(user string) CreateEmbeddingsOpt {
	return func(req *embeddingRequest) {
		req.User = &user
	}
}

// EmbeddingsResponse is a response from the CreateEmbedding API.
type EmbeddingsResponse struct {
	Object string      `json:"object"`
	Data   []Embedding `json:"data"`
	Model  string      `json:"model"`
	Usage  Usage       `json:"usage"`
}

// Embedding is a single embedding object.
type Embedding struct {
	Index     int       `json:"index"`
	Object    string    `json:"object"`
	Embedding []float64 `json:"embedding"`
}

// Usage is an embeddings usage count object.
type Usage struct {
	PromptTokens int `json:"prompt_tokens"`
	TotalTokens  int `json:"total_tokens"`
}

// HTTPClient is an HTTP chat client for the OpenAI API.
type HTTPClient struct {
	key  string
	http httputil.HTTPDoer
}

// CreateEmbeddings creates embeddings from a list of inputs.
func (h *HTTPClient) CreateEmbeddings(ctx context.Context, model string, inputs []string, opts ...CreateEmbeddingsOpt) (resp *EmbeddingsResponse, err error) {
	req := embeddingRequest{Model: model, Input: inputs}

	for _, opt := range opts {
		opt(&req)
	}

	b, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("error marshaling embeddings request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, "https://api.openai.com/v1/embeddings", bytes.NewReader(b))
	if err != nil {
		return nil, fmt.Errorf("error creating HTTP request: %w", err)
	}

	apiKey := req.apiKey
	if req.apiKey == "" {
		apiKey = h.key
	}

	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", fmt.Sprintf("Bearer %s", apiKey))

	httpResp, err := h.http.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("error performing HTTP request: %w", err)
	}

	if httpResp.StatusCode != http.StatusOK {
		return nil, httputil.ErrUnexpectedStatusCode{Expected: http.StatusOK, Actual: httpResp.StatusCode, Response: httpResp}
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

// NewHTTPClient creates a new embeddings HTTP client for the OpenAI API.
func NewHTTPClient(opts ...HTTPClientOpt) *HTTPClient {
	client := HTTPClient{
		http: http.DefaultClient,
	}

	for _, opt := range opts {
		opt(&client)
	}

	return &client
}

// HTTPClientOpt is a functional option for configuring the HTTP client.
type HTTPClientOpt func(*HTTPClient)

// WithKey sets the API key for the HTTP client.
func WithKey(key string) HTTPClientOpt {
	return func(c *HTTPClient) {
		c.key = key
	}
}

// WithHTTPDoer sets the HTTP round tripper for the HTTP client.
func WithHTTPDoer(doer httputil.HTTPDoer) HTTPClientOpt {
	return func(c *HTTPClient) {
		c.http = doer
	}
}
