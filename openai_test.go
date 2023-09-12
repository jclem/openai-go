package openai_test

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/jclem/openai-go"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestChatCompletionResponse_GetChoiceAt(t *testing.T) {
	r := openai.ChatCompletionResponse{
		Choices: []openai.ChatCompletionChoice{
			{
				Message: openai.NewMessage("user", openai.WithMessageContent("ack")),
			},
		},
	}

	_, ok := r.GetChoiceAt(1)
	require.False(t, ok)

	ch, ok := r.GetChoiceAt(0)
	require.True(t, ok)
	assert.Equal(t, ch, r.Choices[0])

}

func TestChatCompletionResponse_GetContentAt(t *testing.T) {
	r := openai.ChatCompletionResponse{
		Choices: []openai.ChatCompletionChoice{
			{
				Message: openai.NewMessage("user", openai.WithMessageContent("ack")),
			},
		},
	}

	_, ok := r.GetChoiceAt(1)
	require.False(t, ok)

	ch, ok := r.GetContentAt(0)
	require.True(t, ok)
	assert.Equal(t, ch, *(r.Choices[0].Message.Content))

	_, ok = r.GetFunctionCallAt(0)
	require.False(t, ok)
}

func TestChatCompletionResponse_GetFunctionCallAt(t *testing.T) {
	call := openai.FunctionCall{
		Name:      "my-function-call",
		Arguments: []byte(`{"foo": "bar"}`),
	}

	r := openai.ChatCompletionResponse{
		Choices: []openai.ChatCompletionChoice{
			{
				Message: openai.NewMessage(
					"user",
					openai.WithMessageFunctionCall(call),
				),
			},
		},
	}

	_, ok := r.GetChoiceAt(1)
	require.False(t, ok)

	ch, ok := r.GetFunctionCallAt(0)
	require.True(t, ok)
	assert.Equal(t, ch, call)

	_, ok = r.GetContentAt(0)
	require.False(t, ok)
}

func TestHTTPClient_DoChatCompletion_OK(t *testing.T) {
	r := &http.Response{}
	r.StatusCode = http.StatusOK
	doer := newTestDoer(r, nil)
	testKey := "api-key"

	c := openai.NewHTTPClient(
		openai.WithHTTPDoer(&doer),
		openai.WithKey(testKey),
	)

	resp, err := c.DoChatCompletion(
		context.Background(),
		"gpt-3.5-turbo",
		[]openai.Message{openai.NewMessage("user", openai.WithMessageContent("Hello, world"))},
		openai.WithTemperature(0.5),
		openai.WithFunctions(
			openai.NewFunctionDefinition("my-function-call", map[string]string{"foo": "bar"}),
		),
	)

	require.NoError(t, err)

	assert.Equal(t, resp, r)
	assert.Equal(t, "POST", doer.req.Method)

	assert.Equal(t, doer.req.Header.Get("Authorization"), "Bearer "+testKey)

	var reqb map[string]any
	err = json.NewDecoder(doer.req.Body).Decode(&reqb)
	require.NoError(t, err)

	assert.Equal(t, "gpt-3.5-turbo", reqb["model"])

	m := reqb["messages"].([]any)[0].(map[string]any)
	assert.Equal(t, "user", m["role"])
	assert.Equal(t, "Hello, world", m["content"])

	f := reqb["functions"].([]any)[0].(map[string]any)
	assert.Equal(t, "my-function-call", f["name"])
	assert.Equal(t, map[string]any{"foo": "bar"}, f["parameters"])
}

func TestHTTPClient_DoChatCompletion_NotOk(t *testing.T) {
	r := &http.Response{}
	r.StatusCode = http.StatusInternalServerError
	doer := newTestDoer(r, nil)

	c := openai.NewHTTPClient(openai.WithHTTPDoer(&doer))

	resp, err := c.DoChatCompletion(
		context.Background(),
		"gpt-3.5-turbo",
		[]openai.Message{},
	)
	require.Nil(t, resp)

	var staterr openai.ErrUnexpectedStatusCode
	assert.ErrorAs(t, err, &staterr)
}

func TestHTTPClient_DoChatCompletion_Error(t *testing.T) {
	reqErr := errors.New("oops")
	doer := newTestDoer(nil, reqErr)

	c := openai.NewHTTPClient(openai.WithHTTPDoer(&doer))

	resp, err := c.DoChatCompletion(
		context.Background(),
		"gpt-3.5-turbo",
		[]openai.Message{},
	)
	require.Nil(t, resp)

	assert.ErrorIs(t, err, reqErr)
}

func TestHTTPClient_CreateChatCompletion(t *testing.T) {
	compresp := openai.ChatCompletionResponse{
		Choices: []openai.ChatCompletionChoice{
			{
				Message: openai.NewMessage("user", openai.WithMessageContent("ack")),
			},
		},
	}
	bodyb, err := json.Marshal(compresp)
	require.NoError(t, err)

	r := &http.Response{}
	r.StatusCode = http.StatusOK
	r.Body = newTestBody(bytes.NewReader(bodyb))
	doer := newTestDoer(r, nil)
	testKey := "api-key"

	c := openai.NewHTTPClient(
		openai.WithHTTPDoer(&doer),
		openai.WithKey(testKey),
	)

	resp, err := c.CreateChatCompletion(
		context.Background(),
		"gpt-3.5-turbo",
		[]openai.Message{openai.NewMessage("user", openai.WithMessageContent("Hello, world"))},
		openai.WithTemperature(0.5),
		openai.WithFunctions(
			openai.NewFunctionDefinition("my-function-call", map[string]string{"foo": "bar"}),
		),
	)

	require.NoError(t, err)

	assert.Equal(t, &compresp, resp)
}

func TestHTTPClient_CreateStreamingChatCompletion(t *testing.T) {
	msg := "ack"
	r := &http.Response{}
	r.StatusCode = http.StatusOK
	sse := fmt.Sprintf(`data: {"choices": [{"index": 0, "delta": {"role": "user", "content": "%s"}}]}
	
[DONE]

`, msg)
	r.Body = newTestBody(strings.NewReader(sse))
	doer := newTestDoer(r, nil)
	testKey := "api-key"

	c := openai.NewHTTPClient(
		openai.WithHTTPDoer(&doer),
		openai.WithKey(testKey),
	)

	stream, err := c.CreateStreamingChatCompletion(
		context.Background(),
		"gpt-3.5-turbo",
		[]openai.Message{},
	)

	require.NoError(t, err)

	obj, err := stream.Next()
	require.NoError(t, err)
	assert.Equal(t, &openai.StreamingChatCompletionObject{
		Choices: []openai.StreamingChatCompletionChoice{
			{Index: 0, Delta: openai.StreamingChatCompletionDelta{Role: "user", Content: &msg}},
		},
	}, obj)
}

type testdoer struct {
	req  *http.Request
	resp *http.Response
	err  error
}

func (t *testdoer) Do(req *http.Request) (*http.Response, error) {
	t.req = req
	return t.resp, t.err
}

func newTestDoer(resp *http.Response, err error) testdoer {
	return testdoer{resp: resp, err: err}
}

type testbody struct {
	io.Reader
}

func (t testbody) Close() error {
	return nil
}

func newTestBody(r io.Reader) testbody {
	return testbody{r}
}
