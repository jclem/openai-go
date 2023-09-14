package openai_test

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"testing"

	"github.com/jclem/openai-go"
	"github.com/jclem/openai-go/internal/httptesting"
	"github.com/jclem/openai-go/pkg/chat"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestClient_Chat_CreateCompletion(t *testing.T) {
	compresp := chat.ChatCompletionResponse{
		Choices: []chat.ChatCompletionChoice{
			{
				Message: chat.NewMessage("user", chat.WithMessageContent("ack")),
			},
		},
	}
	bodyb, err := json.Marshal(compresp)
	require.NoError(t, err)

	resp := &http.Response{}
	resp.StatusCode = http.StatusOK
	resp.Body = httptesting.NewTestBody(bytes.NewReader(bodyb))
	doer := httptesting.NewTestDoer(resp, nil)

	c := openai.NewClient(openai.WithDoer(&doer))

	comp, err := c.Chat.CreateCompletion(
		context.Background(),
		"gpt-3.5-turbo",
		[]chat.Message{chat.NewMessage("user", chat.WithMessageContent("Hello, world."))},
	)
	require.NoError(t, err)

	assert.Equal(t, &compresp, comp)
}
