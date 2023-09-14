
# OpenAI Go Client

**This is an OpenAI client for Go I wrote for personal use. Do not expect it to
necessarily be up to date quickly.**

[![Go Reference](https://pkg.go.dev/badge/github.com/jclem/openai-go.svg)](https://pkg.go.dev/github.com/jclem/openai-go)

## Usage

### Creating a client

```go
import "github.com/jclem/openai-go"

client := openai.NewClient(openai.WithKey(yourAPIKey))

// Optionally, provide a custom HTTP "Do"-er.
client = openai.NewClient(
	openai.WithKey(yourApiKey),
	openai.WithDoer(http.DefaultClient),
)
```

### Making a completion request

Use the client's chat service to create a completion call.

```go
import "github.com/jclem/openai-go/pkg/chat"

comp, err := client.Chat.CreateCompletion(
	context.Background(),
	"gpt-4",
	[]chat.Message{chat.NewMessage("user", chat.WithMessageContent("Hello, world"))},
	chat.WithTemperature(0.6),
)

// Various methods exist to easily read the completion.
content, ok := comp.GetContentAt(0)
```

### Making a streaming completion request

Use the client's chat service to create a streaming completion call.

```go
import "github.com/jclem/openai-go/pkg/chat"

stream, err := client.Chat.CreateStreamingCompletion(
	context.Background(),
	"gpt-4",
	[]chat.Message{chat.NewMessage("user", chat.WithMessageContent("Hello, world"))},
	chat.WithTemperature(0.6),
)

// Call `stream.Next()` to get the next stream completion object. It'll return
// `nil, nil` when done.
for {
	chunk, err := stream.Next()

	if errors.Is(err, chat.ErrStreamDone) {
		break
	}

	if err != nil {
		// Handle error.
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

### Creating embeddings

Use `CreateEmbeddings` to create embeddings, and get back a parsed response.

```go
resp, err := client.Embeddings.Create(
	context.Background(),
	"text-embedding-ada-002",
	[]string{"HEllo, world."},
)

var embedding []float64 = resp.Data[0].Embedding
```