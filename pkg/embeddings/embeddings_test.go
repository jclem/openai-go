package embeddings_test

import (
	"bytes"
	"encoding/json"
	"net/http"
	"testing"

	"github.com/jclem/openai-go/internal/httptesting"
	"github.com/jclem/openai-go/pkg/embeddings"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestHTTPClient_CreateEmbeddings(t *testing.T) {
	embresp := embeddings.EmbeddingsResponse{
		Object: "list",
		Data: []embeddings.Embedding{{
			Index:     0,
			Object:    "embedding",
			Embedding: []float64{0.0, 0.1, 0.2},
		}},
	}
	bodyb, err := json.Marshal(embresp)
	require.NoError(t, err)

	r := &http.Response{}
	r.StatusCode = http.StatusOK
	r.Body = httptesting.NewTestBody(bytes.NewReader(bodyb))
	doer := httptesting.NewTestDoer(r, nil)
	testKey := "api-key"

	c := embeddings.NewHTTPClient(
		embeddings.WithHTTPDoer(&doer),
		embeddings.WithKey(testKey),
	)

	resp, err := c.CreateEmbeddings(
		"ada",
		[]string{"hello"},
		embeddings.WithUser("user"),
	)

	require.NoError(t, err)

	assert.Equal(t, &embresp, resp)
}
