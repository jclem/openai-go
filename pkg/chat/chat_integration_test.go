//go:build integration
// +build integration

package chat_test

import (
	"context"
	"os"
	"testing"

	"github.com/jclem/openai-go/pkg/chat"
	"github.com/stretchr/testify/require"
)

var key = os.Getenv("OPENAI_API_KEY")

func TestCreateChatCompletion(t *testing.T) {
	client := chat.NewHTTPClient(chat.WithKey(key))

	messages := []chat.Message{chat.NewMessage("user", chat.WithMessageContent("Hello, world."))}
	resp, err := client.CreateChatCompletion(context.Background(), "gpt-3.5-turbo", messages, chat.WithMaxTokens(16))
	require.NoError(t, err)
	require.NotEmpty(t, resp.Choices[0].Message.Content)
}

func TestCreateStreamingChatCompletion(t *testing.T) {
	client := chat.NewHTTPClient(chat.WithKey(key))

	messages := []chat.Message{chat.NewMessage("user", chat.WithMessageContent("Hello, world."))}
	resp, err := client.CreateStreamingChatCompletion(context.Background(), "gpt-3.5-turbo", messages, chat.WithMaxTokens(16))
	require.NoError(t, err)

	content := ""

	for {
		obj, err := resp.Next()
		require.NoError(t, err)

		if obj == nil {
			break
		}

		if obj.Choices[0].Delta.Content != nil {
			content += *(obj.Choices[0].Delta.Content)
		}
	}

	require.NotEmpty(t, content)
	require.NoError(t, resp.Close())
}
