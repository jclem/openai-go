//go:build integration
// +build integration

package embeddings_test

import (
	"context"
	"net/http"
	"os"
	"testing"

	"github.com/jclem/openai-go"
	"github.com/jclem/openai-go/internal/service"
	"github.com/jclem/openai-go/pkg/embeddings"
	"github.com/stretchr/testify/require"
)

var key = os.Getenv("OPENAI_API_KEY")

const input = "Hello, world."

func TestCreateEmbeddings(t *testing.T) {
	svc := service.New(openai.DefaultBaseURL, key, http.DefaultClient)
	c := (*embeddings.EmbeddingsService)(svc)
	resp, err := c.Create(context.Background(), "text-embedding-ada-002", []string{input})
	require.NoError(t, err)
	require.NotEmpty(t, resp.Data[0].Embedding)
}
