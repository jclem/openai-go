//go:build integration
// +build integration

package embeddings_test

import (
	"os"
	"testing"

	"github.com/jclem/openai-go/pkg/embeddings"
	"github.com/stretchr/testify/require"
)

var key = os.Getenv("OPENAI_API_KEY")

const input = "Hello, world."

func TestCreateEmbeddings(t *testing.T) {
	client := embeddings.NewHTTPClient(embeddings.WithKey(key))
	resp, err := client.CreateEmbeddings("text-embedding-ada-002", []string{input})
	require.NoError(t, err)
	require.NotEmpty(t, resp.Data[0].Embedding)
}
