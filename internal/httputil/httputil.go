// Package httputil provides HTTP utilities.
package httputil

import (
	"fmt"
	"net/http"
)

// An HTTPDoer is an HTTP client that can perform HTTP requests.
type HTTPDoer interface {
	Do(*http.Request) (*http.Response, error)
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
