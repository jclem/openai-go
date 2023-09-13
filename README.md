
# OpenAI Go Client

**This is an OpenAI client for Go I wrote for personal use. Do not expect it to
necessarily be up to date quickly.**

[![Go Reference](https://pkg.go.dev/badge/github.com/jclem/openai-go.svg)](https://pkg.go.dev/github.com/jclem/openai-go)

## Usage

### Chat

#### Creating a client

```go
import "github.com/jclem/openai-go/pkg/chat"

client := chat.NewHTTPClient(chat.WithKey(yourApiKey))

// Optionally, provide a custom HTTP "Do"-er.
client = chat.NewHTTPClient(
	chat.WithKey(yourApiKey),
	chat.WithHTTPDoer(http.DefaultClient),
)
```

#### Making a completion request

Use `CreateChatCompletion` to create a chat completion call, and get back a
parsed response.

```go
comp, err := client.CreateChatCompletion(
	context.Background(),
	"gpt-4",
	[]chat.Message{chat.NewMessage("user", chat.WithMessageContent("Hello, world"))},
	chat.WithTemperature(0.6),
)

// Various methods exist to easily read the completion.
content, ok := comp.GetContentAt(0)
```

#### Making a streaming request

Use `CreateStreamingChatCompletion` to create a streaming chat completion call,
and get back a streaming response.

```go
stream, err := client.CreateStreamingChatCompletion(
	context.Background(),
	"gpt-4",
	[]chat.Message{chat.NewMessage("user", chat.WithMessageContent("Hello, world"))},
	chat.WithTemperature(0.6),
)

// Call `stream.Next()` to get the next stream completion object. It'll return
// `nil, nil` when done.
for {
	chunk, err := stream.Next()
	if err != nil {
		// Handle error.
	}

	if chunk == nil {
		break
	}

	// Various methods exist to easily read the stream chunk.
	content, ok := chunk.GetContentAt(0)
	if ok {
		fmt.Printf(content)
	}
}

// The caller must close the stream.
if closeErr := stream.Close(); closeErr != nil {
	// Handle error.
}
```

### Embeddings

#### Creating a client

```go
import "github.com/jclem/openai-go/pkg/embeddings"

client := embeddings.NewHTTPClient(embeddings.WithKey(yourApiKey))

// Optionally, provide a custom HTTP "Do"-er.
client = embeddings.NewHTTPClient(
	embeddings.WithKey(yourApiKey),
	embeddings.WithHTTPDoer(http.DefaultClient),
)
```

#### Creating embeddings

Use `CreateEmbeddings` to create embeddings, and get back a parsed response.

```go
resp, err := client.CreateEmbeddings(
	context.Background(),
	"text-embedding-ada-002",
	[]string{"HEllo, world."},
)

var embedding []float64 = resp.Data[0].Embedding
```