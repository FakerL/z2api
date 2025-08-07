# Z.AI Proxy

A simple OpenAI-compatible API proxy for Z.AI, designed for deployment on Cloudflare Workers or Docker.

## Features

- **OpenAI-compatible API** - Works with any OpenAI client library
- **Token pool load balancing** - Automatically manages multiple tokens for better performance
- **Simple authentication** - No credentials needed, tokens fetched automatically
- **Streaming support** - Full support for streaming chat completions
- **Docker ready** - Easy deployment with Docker containers

## Quick Start with Docker

### 1. Build and Run

```bash
# Build the Docker image
docker build -t z-ai-proxy .

# Run with default settings (5 token pool)
docker run -p 3000:3000 z-ai-proxy

# Run with custom token pool size
docker run -p 3000:3000 -e TOKEN_POOL_SIZE=10 z-ai-proxy
```

### 2. Using Docker Compose

```bash
# Start the service
docker-compose up -d

# View logs
docker-compose logs -f

# Stop the service
docker-compose down
```

## Development Setup

### Local Development

```bash
# No dependencies needed - uses only Node.js built-ins

# Run tests
TOKEN_POOL_SIZE=2 node test.js

# Start development server
TOKEN_POOL_SIZE=5 node server.js

# Start with different port
PORT=8080 TOKEN_POOL_SIZE=3 node server.js
```

### Cloudflare Workers Deployment

```bash
# Deploy the same index.js to Cloudflare Workers
# Set TOKEN_POOL_SIZE environment variable in Workers dashboard
# Deploy index.js as your worker script
```

## Configuration

The service is configured via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `TOKEN_POOL_SIZE` | `5` | Number of tokens to maintain in the pool |
| `API_KEY` | `sk-z2api-key-2024` | API key for client authentication |
| `SHOW_THINK_TAGS` | `false` | Whether to show thinking tags in responses |
| `PORT` | `3000` | Port to run the service on |

## API Usage

The proxy provides OpenAI-compatible endpoints:

### Chat Completions

```bash
curl -X POST http://localhost:3000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-z2api-key-2024" \
  -d '{
    "model": "glm-4.5-air",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": false
  }'
```

### Available Models

```bash
curl http://localhost:3000/v1/models \
  -H "Authorization: Bearer sk-z2api-key-2024"
```

### Supported Models

- `glm-4.5` - Full model
- `glm-4.5-air` - Lightweight model

## How It Works

1. **Token Pool Initialization**: On startup, the service fetches N tokens (configurable) from Z.AI
2. **Load Balancing**: Requests are distributed across tokens using least-recently-used algorithm  
3. **Automatic Refresh**: Failed tokens are automatically refreshed in the background
4. **OpenAI Compatibility**: Request/response format matches OpenAI API exactly

## Architecture

```
Client Request → API Key Auth → Token Pool → Z.AI API → Response Processing → Client
```

The service maintains a pool of Z.AI tokens and automatically:
- Fetches fresh tokens when needed
- Tracks token usage and failures
- Provides load balancing across the token pool
- Handles both streaming and non-streaming responses

## Testing

The service includes comprehensive tests:

```bash
# Run all tests
node test.js

# Test with different pool sizes
TOKEN_POOL_SIZE=10 node test.js
TOKEN_POOL_SIZE=1 node test.js
TOKEN_POOL_SIZE=0 node test.js  # Should fail gracefully
```

## License

MIT
