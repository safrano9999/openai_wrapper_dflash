# DFlash MLX Server

OpenAI-compatible API server for [DFlash](https://github.com/z-lab/dflash) speculative decoding on Apple Silicon via MLX.

Wraps `dflash.model_mlx.stream_generate` behind a `/v1/chat/completions` endpoint with a request queue, so any OpenAI-compatible client works out of the box.

## Requirements

- Apple Silicon Mac (M1 or newer)
- Python 3.11+
- ~3.5 GB free unified memory

## Setup

```bash
git clone https://github.com/z-lab/dflash.git
cd dflash
python3.11 -m venv .venv-mlx
source .venv-mlx/bin/activate
pip install ".[mlx]"
pip install fastapi uvicorn
```

## First run — model download

On first start the server downloads two models from HuggingFace automatically:

| Model | Size | Source |
|-------|------|--------|
| Target (Qwen3.5-4B 4bit) | ~3 GB | `mlx-community/Qwen3.5-4B-4bit` |
| DFlash draft | ~1 GB | `z-lab/Qwen3.5-4B-DFlash` |

Both are cached locally after the first download and **will not be downloaded again** on subsequent starts.

> **Note:** The target model requires the full MLX safetensors package including all JSON config files (`config.json`, `tokenizer.json`, etc.). A plain GGUF file is not sufficient — mlx-lm needs the complete model directory with metadata.

From the second start onward the server loads directly from cache and is ready in seconds.

## Start

```bash
source .venv-mlx/bin/activate
python server.py --port 8080
```

On startup you will see a smoke test confirming the model loaded correctly:

```
──────────────────────────────────────────────────
  Target model : Qwen3.5-4B-4bit
  Draft model  : z-lab/Qwen3.5-4B-DFlash
  Target cached: yes
  Draft  cached: yes
──────────────────────────────────────────────────
  Loading target model...  Done (4.2s)
  Loading draft model...   Done (1.1s)
  Running smoke test...
  [OK] output: '1+1=2'
  [OK] 47.3 tok/s | peak mem: 2.81 GB
──────────────────────────────────────────────────
  Server ready on port below
──────────────────────────────────────────────────
```

## Usage

```bash
curl -s -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hello!"}],"stream":false}'
```

Streaming:

```bash
curl -s -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hello!"}],"stream":true}'
```

### Request parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `messages` | required | OpenAI-format message array |
| `stream` | `true` | Stream tokens as SSE |
| `max_tokens` | `512` | Maximum tokens to generate |
| `temperature` | `0.0` | Sampling temperature |
| `block_size` | auto | DFlash draft block size |

### Other endpoints

- `GET /v1/models` — list available models
- `GET /health` — server status + current queue depth

## Queue

Concurrent requests are queued and processed one at a time. The console shows queue depth on each request. The `X-Queue-Depth` header in each response shows how many requests are waiting.
