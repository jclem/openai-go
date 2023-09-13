// Package httptesting provides internal testing utilities.
package httptesting

import (
	"io"
	"net/http"
)

// TestDoer is a test HTTPDoer.
type TestDoer struct {
	req  *http.Request
	resp *http.Response
	err  error
}

// Do implements the HTTPDoer interface.
func (t *TestDoer) Do(req *http.Request) (*http.Response, error) {
	t.req = req
	return t.resp, t.err
}

// NewTestDoer creates a new TestDoer.
func NewTestDoer(resp *http.Response, err error) TestDoer {
	return TestDoer{resp: resp, err: err}
}

// TestBody is a test HTTP body.
type TestBody struct {
	io.Reader
}

// Close implements the io.Closer interface.
func (t TestBody) Close() error {
	return nil
}

// NewTestBody creates a new TestBody.
func NewTestBody(r io.Reader) TestBody {
	return TestBody{r}
}
