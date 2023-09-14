// Package service provides common types used by services.
package service

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
)

// A Doer is an interface for performing HTTP requests.
type Doer interface {
	Do(*http.Request) (*http.Response, error)
}

// An UnexpectedStatusCodeError is an error returned when the HTTP response has an
// unexpected status code.
//
// In includes the expected and actual codes, as well as the response.
type UnexpectedStatusCodeError struct {
	Expected int
	Actual   int
	Response *http.Response
}

// Error implements the error interface.
func (e UnexpectedStatusCodeError) Error() string {
	return fmt.Sprintf("unexpected status code %d (expected %d)", e.Actual, e.Expected)
}

// A Client is a struct used by services to make HTTP requests.
type Client struct {
	baseURL *url.URL
	key     string
	doer    Doer
}

// NewRequestWithContext creates a new HTTP request.
func (c *Client) NewRequestWithContext(
	ctx context.Context,
	method,
	path string,
	body any,
	opts ...RequestOpt,
) (*http.Request, error) {
	u := c.baseURL.JoinPath(path)

	var buf io.ReadWriter
	if body != nil {
		buf = &bytes.Buffer{}
		enc := json.NewEncoder(buf)
		enc.SetEscapeHTML(false)

		if err := enc.Encode(body); err != nil {
			return nil, fmt.Errorf("failed to encode body: %w", err)
		}
	}

	req, err := http.NewRequestWithContext(ctx, method, u.String(), buf)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}

	req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", c.key))

	for _, opt := range opts {
		opt(req)
	}

	return req, nil
}

// Do performs an HTTP request.
//
// If v is nil, the response body is not closed, and the caller must close it.
func (c *Client) Do(req *http.Request, v any) (*http.Response, error) {
	resp, err := c.doer.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to perform request: %w", err)
	}

	if v != nil {
		defer resp.Body.Close() //nolint: errcheck // No handling would be done here.
	}

	if !(200 <= resp.StatusCode && resp.StatusCode <= 299) { //revive:disable-line:add-constant
		return resp, UnexpectedStatusCodeError{
			Expected: http.StatusOK,
			Actual:   resp.StatusCode,
			Response: resp,
		}
	}

	switch v := v.(type) {
	case nil:
	case io.Writer:
		_, err = io.Copy(v, resp.Body)
	default:
		decErr := json.NewDecoder(resp.Body).Decode(v)
		if errors.Is(decErr, io.EOF) {
			decErr = nil
		}

		if decErr != nil {
			err = decErr
		}
	}

	return resp, err
}

// A RequestOpt is a functional option for configuring a Request.
type RequestOpt func(*http.Request)

// WithAPIKey sets the API key for the request.
//
// The default service key is used if this option is not provided or the value
// is empty.
func WithAPIKey(key string) RequestOpt {
	return func(r *http.Request) {
		if key != "" {
			r.Header.Set("Authorization", fmt.Sprintf("Bearer %s", key))
		}
	}
}

// A Service is a common type for services, which wrap OpenAI API requests.
type Service struct {
	Client Client
}

// New creates a new Service.
func New(baseURL *url.URL, key string, doer Doer) *Service {
	return &Service{
		Client: Client{
			baseURL: baseURL,
			key:     key,
			doer:    doer,
		},
	}
}
