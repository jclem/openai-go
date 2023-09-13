package chat_test

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"testing"

	"github.com/jclem/openai-go/internal/httptesting"
	"github.com/jclem/openai-go/pkg/chat"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestChatCompletionResponse_GetChoiceAt(t *testing.T) {
	r := chat.ChatCompletionResponse{
		Choices: []chat.ChatCompletionChoice{
			{
				Message: chat.NewMessage("user", chat.WithMessageContent("ack")),
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
	r := chat.ChatCompletionResponse{
		Choices: []chat.ChatCompletionChoice{
			{
				Message: chat.NewMessage("user", chat.WithMessageContent("ack")),
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
	call := chat.FunctionCall{
		Name:      "my-function-call",
		Arguments: []byte(`{"foo": "bar"}`),
	}

	r := chat.ChatCompletionResponse{
		Choices: []chat.ChatCompletionChoice{
			{
				Message: chat.NewMessage(
					"user",
					chat.WithMessageFunctionCall(call),
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

func TestHTTPClient_CreateChatCompletion(t *testing.T) {
	compresp := chat.ChatCompletionResponse{
		Choices: []chat.ChatCompletionChoice{
			{
				Message: chat.NewMessage("user", chat.WithMessageContent("ack")),
			},
		},
	}
	bodyb, err := json.Marshal(compresp)
	require.NoError(t, err)

	r := &http.Response{}
	r.StatusCode = http.StatusOK
	r.Body = httptesting.NewTestBody(bytes.NewReader(bodyb))
	doer := httptesting.NewTestDoer(r, nil)
	testKey := "api-key"

	c := chat.NewHTTPClient(
		chat.WithHTTPDoer(&doer),
		chat.WithKey(testKey),
	)

	resp, err := c.CreateChatCompletion(
		context.Background(),
		"gpt-3.5-turbo",
		[]chat.Message{chat.NewMessage("user", chat.WithMessageContent("Hello, world"))},
		chat.WithTemperature(0.5),
		chat.WithFunctions(
			chat.NewFunctionDefinition("my-function-call", map[string]string{"foo": "bar"}),
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
	r.Body = httptesting.NewTestBody(strings.NewReader(sse))
	doer := httptesting.NewTestDoer(r, nil)
	testKey := "api-key"

	c := chat.NewHTTPClient(
		chat.WithHTTPDoer(&doer),
		chat.WithKey(testKey),
	)

	stream, err := c.CreateStreamingChatCompletion(
		context.Background(),
		"gpt-3.5-turbo",
		[]chat.Message{},
	)

	require.NoError(t, err)

	obj, err := stream.Next()
	require.NoError(t, err)
	assert.Equal(t, &chat.StreamingChatCompletionObject{
		Choices: []chat.StreamingChatCompletionChoice{
			{Index: 0, Delta: chat.StreamingChatCompletionDelta{Role: "user", Content: &msg}},
		},
	}, obj)
}
