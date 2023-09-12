# OpenAI Go Client

**This is an OpenAI client for Go I wrote for personal use. Do not expect it to
necessarily be up to date quickly.**

## Usage

### Creating a client

```go
import "github.com/jclem/openai-go"

client := openai.NewHTTPClient(openai.WithKey(yourApiKey))

// Optionally, provide a custom HTTP "Do"-er.
client = openai.NewHTTPClient(
	openai.WithKey(yourApiKey),
	openai.WithHTTPDoer(http.DefaultClient),
)
```

### Making a low-level completion call

Use `DoChatCompletion` to create a chat completion call, and get back an
`*http.Response`.

```go
httpresp, err := client.DoChatCompletion(
	context.Background(),
	"gpt-4",
	[]openai.Message{openai.NewMessage("user", openai.WithMessageContent("Hello, world"))},
	openai.WithTemperature(0.6),
)
```

### Making a completion request

Use `CreateChatCompletion` to create a chat completion call, and get back a
parsed response.

```go
comp, err := client.CreateChatCompletion(
	context.Background(),
	"gpt-4",
	[]openai.Message{openai.NewMessage("user", openai.WithMessageContent("Hello, world"))},
	openai.WithTemperature(0.6),
)

// Various methods exist to easily read the completion.
content, ok := comp.GetContentAt(0)
```

### Making a streaming request

Use `CreateStreamingChatCompletion` to create a streaming chat completion call,
and get back a streaming response.

```go
stream, err := client.CreateStreamingChatCompletion(
	context.Background(),
	"gpt-4",
	[]openai.Message{openai.NewMessage("user", openai.WithMessageContent("Hello, world"))},
	openai.WithTemperature(0.6),
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