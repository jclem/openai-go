// Package embeddings provides a embeddings client for the OpenAI API.
package embeddings

import (
	"context"
	"fmt"
	"net/http"

	"github.com/jclem/openai-go/internal/service"
)

type request struct {
	apiKey string

	Model string   `json:"model"`
	Input []string `json:"input"`
	User  *string  `json:"user,omitempty"`
}

// CreateOpt is a functional option for configuring an embedding request.
type CreateOpt func(*request)

// WithUser sets the user ID for the request.
func WithUser(user string) CreateOpt {
	return func(req *request) {
		req.User = &user
	}
}

// Response is a response from the embeddings API.
type Response struct {
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

// Service is a service wrapping an OpenAI-compatible embeddings API.
type Service service.Service

// Create creates embeddings from a list of inputs.
func (h *Service) Create(
	ctx context.Context,
	model string,
	inputs []string,
	opts ...CreateOpt,
) (*Response, error) {
	req := request{Model: model, Input: inputs}

	for _, opt := range opts {
		opt(&req)
	}

	httpReq, err := h.Client.NewRequestWithContext(ctx, http.MethodPost, "/embeddings", req,
		service.WithAPIKey(req.apiKey))
	if err != nil {
		return nil, fmt.Errorf("error creating embeddings request: %w", err)
	}

	var resp Response
	if _, err := h.Client.Do(httpReq, &resp); err != nil { //nolint: bodyclose // False positive.
		return nil, fmt.Errorf("error performing embeddings request: %w", err)
	}

	return &resp, nil
}
