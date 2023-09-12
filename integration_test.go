//go:build integration
// +build integration

package openai_test

import (
	"context"
	"encoding/json"
	"os"
	"testing"

	"github.com/jclem/openai-go"
	"github.com/stretchr/testify/require"
)

var key = os.Getenv("OPENAI_API_KEY")

func TestDoChatCompletion(t *testing.T) {
	client := openai.NewHTTPClient(openai.WithKey(key))

	messages := []openai.Message{openai.NewMessage("user", openai.WithMessageContent("Hello, world."))}
	resp, err := client.DoChatCompletion(context.Background(), "gpt-3.5-turbo", messages, openai.WithMaxTokens(16))
	require.NoError(t, err)

	var body struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}

	err = json.NewDecoder(resp.Body).Decode(&body)
	require.NoError(t, err)

	err = resp.Body.Close()
	require.NoError(t, err)
	require.NotEmpty(t, body.Choices[0].Message.Content)
}

func TestCreateChatCompletion(t *testing.T) {
	client := openai.NewHTTPClient(openai.WithKey(key))

	messages := []openai.Message{openai.NewMessage("user", openai.WithMessageContent("Hello, world."))}
	resp, err := client.CreateChatCompletion(context.Background(), "gpt-3.5-turbo", messages, openai.WithMaxTokens(16))
	require.NoError(t, err)
	require.NotEmpty(t, resp.Choices[0].Message.Content)
}

func TestCreateStreamingChatCompletion(t *testing.T) {
	client := openai.NewHTTPClient(openai.WithKey(key))

	messages := []openai.Message{openai.NewMessage("user", openai.WithMessageContent("Hello, world."))}
	resp, err := client.CreateStreamingChatCompletion(context.Background(), "gpt-3.5-turbo", messages, openai.WithMaxTokens(16))
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
