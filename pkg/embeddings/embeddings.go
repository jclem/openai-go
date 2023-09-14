// Package embeddings provides a embeddings client for the OpenAI API.
package embeddings

import (
	"context"
	"fmt"
	"net/http"

	"github.com/jclem/openai-go/internal/service"
)

type embeddingRequest struct {
	apiKey string `json:"-"`

	Model string   `json:"model"`
	Input []string `json:"input"`
	User  *string  `json:"user,omitempty"`
}

// CreateOpt is a functional option for configuring an embedding request.
type CreateOpt func(*embeddingRequest)

// WithUser sets the user ID for the request.
func WithUser(user string) CreateOpt {
	return func(req *embeddingRequest) {
		req.User = &user
	}
}

// EmbeddingsResponse is a response from the embeddings API.
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

// EmbeddingsService is a service wrapping an OpenAI-compatible embeddings API.
type EmbeddingsService service.Service

// Create creates embeddings from a list of inputs.
func (h *EmbeddingsService) Create(ctx context.Context, model string, inputs []string, opts ...CreateOpt) (resp *EmbeddingsResponse, err error) {
	req := embeddingRequest{Model: model, Input: inputs}

	for _, opt := range opts {
		opt(&req)
	}

	httpReq, err := h.Client.NewRequest(http.MethodPost, "/embeddings", req, service.WithAPIKey(req.apiKey))
	if err != nil {
		return nil, fmt.Errorf("error creating embeddings request: %w", err)
	}

	if _, err := h.Client.Do(ctx, httpReq, &resp); err != nil {
		return nil, fmt.Errorf("error performing embeddings request: %w", err)
	}

	return resp, nil
}
