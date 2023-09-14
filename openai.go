// Package openai provides an OpenAI-compatible API client.
package openai

import (
	"net/http"
	"net/url"

	"github.com/jclem/openai-go/internal/service"
	"github.com/jclem/openai-go/pkg/chat"
	"github.com/jclem/openai-go/pkg/embeddings"
)

// DefaultBaseURL is the default base URL for the OpenAI API.
var DefaultBaseURL = &url.URL{
	Scheme: "https",
	Host:   "api.openai.com",
	Path:   "/v1",
}

// A Doer is an interface for performing HTTP requests.
type Doer interface {
	Do(*http.Request) (*http.Response, error)
}

// A Client is an OpenAI-compatible API client.
type Client struct {
	Chat       *chat.Service
	Embeddings *embeddings.Service

	key     string
	baseURL *url.URL
	doer    service.Doer
	common  *service.Service
}

// NewClient creates a new Client.
func NewClient(opts ...ClientOpt) *Client {
	c := Client{
		baseURL: DefaultBaseURL,
		doer:    http.DefaultClient,
	}

	for _, opt := range opts {
		opt(&c)
	}

	c.common = service.New(c.baseURL, c.key, c.doer)
	c.Chat = (*chat.Service)(c.common)

	return &c
}

// A ClientOpt is a functional option for configuring a Client.
type ClientOpt func(*Client)

// WithKey sets the API key for the Client.
//
// If no key is provided, one must be provided for every request.
func WithKey(key string) ClientOpt {
	return func(c *Client) {
		c.key = key
	}
}

// WithBaseURL sets the base URL for the Client.
//
// The default value is "https://api.openai.com/v1".
func WithBaseURL(baseURL *url.URL) ClientOpt {
	return func(c *Client) {
		c.baseURL = baseURL
	}
}

// WithDoer sets the HTTP client for the Client.
func WithDoer(doer Doer) ClientOpt {
	return func(c *Client) {
		c.doer = doer
	}
}
