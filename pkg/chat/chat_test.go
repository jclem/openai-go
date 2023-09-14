package chat_test

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"testing"

	"github.com/jclem/openai-go"
	"github.com/jclem/openai-go/internal/httptesting"
	"github.com/jclem/openai-go/internal/service"
	"github.com/jclem/openai-go/pkg/chat"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestChatCompletionResponse_GetChoiceAt(t *testing.T) {
	t.Parallel()

	r := chat.CompletionResponse{
		Choices: []chat.CompletionChoice{
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
	t.Parallel()

	r := chat.CompletionResponse{
		Choices: []chat.CompletionChoice{
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
	t.Parallel()

	call := chat.FunctionCall{
		Name:      "my-function-call",
		Arguments: []byte(`{"foo": "bar"}`),
	}

	r := chat.CompletionResponse{
		Choices: []chat.CompletionChoice{
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
	t.Parallel()

	compresp := chat.CompletionResponse{
		Choices: []chat.CompletionChoice{
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

	svc := service.New(openai.DefaultBaseURL, testKey, &doer)
	c := (*chat.Service)(svc)

	resp, err := c.CreateCompletion(
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
	t.Parallel()

	msg := "ack"
	r := &http.Response{}
	r.StatusCode = http.StatusOK
	sse := fmt.Sprintf(`data: {"choices": [{"index": 0, "delta": {"role": "user", "content": "%s"}}]}
	
[DONE]

`, msg)
	r.Body = httptesting.NewTestBody(strings.NewReader(sse))
	doer := httptesting.NewTestDoer(r, nil)
	testKey := "api-key"

	svc := service.New(openai.DefaultBaseURL, testKey, &doer)
	c := (*chat.Service)(svc)

	stream, err := c.CreateStreamingCompletion(
		context.Background(),
		"gpt-3.5-turbo",
		[]chat.Message{},
	)

	require.NoError(t, err)

	obj, err := stream.Next()
	require.NoError(t, err)
	assert.Equal(t, &chat.StreamingCompletionObject{
		Choices: []chat.StreamingCompletionChoice{
			{Index: 0, Delta: chat.StreamingCompletionDelta{Role: "user", Content: &msg}},
		},
	}, obj)
}
