# Usage Guide

## Quick Start

### 1. Install the Router

```bash
git clone https://github.com/nomyo-ai/nomyo-router.git
cd nomyo-router
python3 -m venv .venv/router
source .venv/router/bin/activate
pip3 install -r requirements.txt
```

### 2. Configure Endpoints

Edit `config.yaml`:

```yaml
endpoints:
  - http://localhost:11434

max_concurrent_connections: 2
```

### 3. Run the Router

```bash
uvicorn router:app --host 0.0.0.0 --port 12434
```

### 4. Use the Router

Configure your frontend to point to `http://localhost:12434` instead of your Ollama instance.

## API Endpoints

### Ollama-Compatible Endpoints

The router provides all standard Ollama API endpoints:

| Endpoint        | Method | Description           |
| --------------- | ------ | --------------------- |
| `/api/generate` | POST   | Generate text         |
| `/api/chat`     | POST   | Chat completions      |
| `/api/embed`    | POST   | Embeddings            |
| `/api/tags`     | GET    | List available models |
| `/api/ps`       | GET    | List loaded models    |
| `/api/show`     | POST   | Show model details    |
| `/api/pull`     | POST   | Pull a model          |
| `/api/push`     | POST   | Push a model          |
| `/api/create`   | POST   | Create a model        |
| `/api/copy`     | POST   | Copy a model          |
| `/api/delete`   | DELETE | Delete a model        |

### OpenAI-Compatible Endpoints

For OpenAI API compatibility:

| Endpoint               | Method | Description      |
| ---------------------- | ------ | ---------------- |
| `/v1/chat/completions` | POST   | Chat completions |
| `/v1/completions`      | POST   | Text completions |
| `/v1/embeddings`       | POST   | Embeddings       |
| `/v1/models`           | GET    | List models      |

### Monitoring Endpoints

| Endpoint                           | Method | Description                              |
| ---------------------------------- | ------ | ---------------------------------------- |
| `/api/usage`                       | GET    | Current connection counts                |
| `/api/token_counts`                | GET    | Token usage statistics                   |
| `/api/stats`                       | POST   | Detailed model statistics                |
| `/api/aggregate_time_series_days`  | POST   | Aggregate time series data into daily    |
| `/api/version`                     | GET    | Ollama version info                      |
| `/api/config`                      | GET    | Endpoint configuration                   |
| `/api/usage-stream`                | GET    | Real-time usage updates (SSE)            |
| `/health`                          | GET    | Health check                             |

## Making Requests

### Basic Chat Request

```bash
curl http://localhost:12434/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3:latest",
    "messages": [
      {"role": "user", "content": "Hello, how are you?"}
    ],
    "stream": false
  }'
```

### Streaming Response

```bash
curl http://localhost:12434/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3:latest",
    "messages": [
      {"role": "user", "content": "Tell me a story"}
    ],
    "stream": true
  }'
```

### OpenAI API Format

```bash
curl http://localhost:12434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o-nano",
    "messages": [
      {"role": "user", "content": "Hello"}
    ]
  }'
```

## Advanced Features

### Multiple Opinions Ensemble (MOE)

Prefix your model name with `moe-` to enable the MOE system:

```bash
curl http://localhost:12434/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "moe-llama3:latest",
    "messages": [
      {"role": "user", "content": "Explain quantum computing"}
    ]
  }'
```

The MOE system:

1. Generates 3 responses from different endpoints
2. Generates 3 critiques of those responses
3. Selects the best response
4. Generates a final refined response

### Token Tracking

The router automatically tracks token usage:

```bash
curl http://localhost:12434/api/token_counts
```

Response:

```json
{
  "total_tokens": 1542,
  "breakdown": [
    {
      "endpoint": "http://localhost:11434",
      "model": "llama3",
      "input_tokens": 120,
      "output_tokens": 1422,
      "total_tokens": 1542
    }
  ]
}
```

### Real-time Monitoring

Use Server-Sent Events to monitor usage in real-time:

```bash
curl http://localhost:12434/api/usage-stream
```

## Integration Examples

### Python Client

```python
import requests

url = "http://localhost:12434/api/chat"
data = {
    "model": "llama3",
    "messages": [{"role": "user", "content": "What is AI?"}],
    "stream": False
}

response = requests.post(url, json=data)
print(response.json())
```

### JavaScript Client

```javascript
const response = await fetch('http://localhost:12434/api/chat', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    model: 'llama3',
    messages: [{ role: 'user', content: 'Hello!' }],
    stream: false
  })
});

const data = await response.json();
console.log(data);
```

### Streaming with JavaScript

```javascript
const eventSource = new EventSource('http://localhost:12434/api/usage-stream');

eventSource.onmessage = (event) => {
  const usage = JSON.parse(event.data);
  console.log('Current usage:', usage);
};
```

## Python Ollama Client

```python
from ollama import Client

# Configure the client to use the router
client = Client(host='http://localhost:12434')

# Chat with a model
response = client.chat(
    model='llama3:latest',
    messages=[
        {'role': 'user', 'content': 'Explain quantum computing'}
    ]
)
print(response['message']['content'])

# Generate text
response = client.generate(
    model='llama3:latest',
    prompt='Write a short poem about AI'
)
print(response['response'])

# List available models
models = client.list()['models']
print(f"Available models: {[m['name'] for m in models]}")
```

### Python OpenAI Client

```python
from openai import OpenAI

# Configure the client to use the router
client = OpenAI(
    base_url='http://localhost:12434/v1',
    api_key='not-needed'  # API key is not required for local usage
)

# Chat completions
response = client.chat.completions.create(
    model='gpt-4o-nano',
    messages=[
        {'role': 'user', 'content': 'What is the meaning of life?'}
    ]
)
print(response.choices[0].message.content)

# Text completions
response = client.completions.create(
    model='llama3:latest',
    prompt='Once upon a time'
)
print(response.choices[0].text)

# Embeddings
response = client.embeddings.create(
    model='llama3:latest',
    input='The quick brown fox jumps over the lazy dog'
)
print(f"Embedding length: {len(response.data[0].embedding)}")

# List models
response = client.models.list()
print(f"Available models: {[m.id for m in response.data]}")
```

## Best Practices

### 1. Model Selection

- Use the same model name across all endpoints
- For Ollama, append `:latest` or a specific version tag
- For OpenAI endpoints, use the model name without version

### 2. Connection Management

- Set `max_concurrent_connections` appropriately for your hardware
- Monitor `/api/usage` to ensure endpoints aren't overloaded
- Consider using the MOE system for critical queries

### 3. Error Handling

- Check the `/health` endpoint regularly
- Implement retry logic with exponential backoff
- Monitor error rates and connection failures

### 4. Performance

- Keep frequently used models loaded on multiple endpoints
- Use streaming for large responses
- Monitor token usage to optimize costs

## Troubleshooting

### Common Issues

**Problem**: Model not found

- **Solution**: Ensure the model is pulled on at least one endpoint
- **Check**: `curl http://localhost:12434/api/tags`

**Problem**: Connection refused

- **Solution**: Verify Ollama endpoints are running and accessible
- **Check**: `curl http://localhost:12434/health`

**Problem**: High latency

- **Solution**: Check endpoint health and connection counts
- **Check**: `curl http://localhost:12434/api/usage`

**Problem**: Token tracking not working

- **Solution**: Ensure database path is writable
- **Check**: `ls -la token_counts.db`

## Examples

See the [examples](examples/) directory for complete integration examples.
