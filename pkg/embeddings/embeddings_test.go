package embeddings_test

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"testing"

	"github.com/jclem/openai-go"
	"github.com/jclem/openai-go/internal/httptesting"
	"github.com/jclem/openai-go/internal/service"
	"github.com/jclem/openai-go/pkg/embeddings"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestHTTPClient_CreateEmbeddings(t *testing.T) {
	t.Parallel()

	embresp := embeddings.Response{
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

	svc := service.New(openai.DefaultBaseURL, testKey, &doer)
	c := (*embeddings.Service)(svc)

	resp, err := c.Create(
		context.Background(),
		"ada",
		[]string{"hello"},
		embeddings.WithUser("user"),
	)

	require.NoError(t, err)

	assert.Equal(t, &embresp, resp)
}
