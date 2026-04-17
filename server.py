"""
DFlash MLX Server — OpenAI-compatible API with request queue
Usage: python server.py [--port 8080]
"""
import argparse
import asyncio
import json
import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

import mlx.core as mx
from huggingface_hub import snapshot_download
from huggingface_hub.utils import LocalEntryNotFoundError
from dflash.model_mlx import load, load_draft, stream_generate

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH      = "mlx-community/Qwen3.5-4B-4bit"
DRAFT_ID        = "z-lab/Qwen3.5-4B-DFlash"
MODEL_ID        = "qwen3.5-4b-dflash"
MAX_PROMPT_TOKENS = 2048   # hard limit — larger prompts cause Metal GPU timeout on 8GB
DEFAULT_BLOCK_SIZE = 4     # smaller = less GPU work per step, more stable


def _is_cached(repo_id: str) -> bool:
    try:
        snapshot_download(repo_id, local_files_only=True)
        return True
    except (LocalEntryNotFoundError, Exception):
        return False

# ── Global state ──────────────────────────────────────────────────────────────
model     = None
draft     = None
tokenizer = None
queue: asyncio.Queue = None


def _smoke_test():
    """Quick sanity check — generates 5 tokens and prints result + memory."""
    print("  Running smoke test...", flush=True)
    t0 = time.perf_counter()
    output = ""
    last = None
    for resp in stream_generate(model, draft, tokenizer, "1+1=", max_tokens=5, temperature=0.0):
        output += resp.text
        last = resp
    elapsed = time.perf_counter() - t0
    mem_gb = mx.get_peak_memory() / 1e9

    if last is None:
        print("  [FAIL] smoke test produced no output")
        return False

    print(f"  [OK] output: '1+1={output.strip()}'")
    print(f"  [OK] {last.generation_tps:.1f} tok/s | peak mem: {mem_gb:.2f} GB | {elapsed:.1f}s")
    return True


# ── Lifespan: load models once at startup ─────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, draft, tokenizer, queue

    print(f"\n{'─'*50}")
    print(f"  Target model : {MODEL_PATH.split('/')[-1]}")
    print(f"  Draft model  : {DRAFT_ID}")
    print(f"{'─'*50}")

    cached_model = _is_cached(MODEL_PATH) if "/" in MODEL_PATH and not MODEL_PATH.startswith("/") else True
    cached_draft = _is_cached(DRAFT_ID)
    print(f"  Target cached: {'yes' if cached_model else 'no — will download'}")
    print(f"  Draft  cached: {'yes' if cached_draft else 'no — will download'}")
    print(f"{'─'*50}")

    print("  Loading target model...", flush=True)
    t0 = time.perf_counter()
    model, tokenizer = load(MODEL_PATH)
    print(f"  Done ({time.perf_counter()-t0:.1f}s)")

    print("  Loading draft model...", flush=True)
    t0 = time.perf_counter()
    draft = load_draft(DRAFT_ID)
    print(f"  Done ({time.perf_counter()-t0:.1f}s)")

    ok = _smoke_test()
    print(f"{'─'*50}")
    if not ok:
        print("  WARNING: smoke test failed — server starts anyway")
    else:
        print(f"  Server ready on port below")
    print(f"{'─'*50}\n")

    queue = asyncio.Queue()
    asyncio.create_task(queue_worker())
    yield


app = FastAPI(title="DFlash MLX Server", lifespan=lifespan)


# ── Queue worker: one inference at a time ─────────────────────────────────────
async def queue_worker():
    while True:
        job = await queue.get()
        prompt, max_tokens, temperature, block_size, chunk_queue = job
        qsize = queue.qsize()
        print(f"[queue] processing | {qsize} still waiting")
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                _run_inference,
                prompt, max_tokens, temperature, block_size, chunk_queue,
            )
        except Exception as e:
            await chunk_queue.put(("error", str(e)))
        finally:
            queue.task_done()


def _run_inference(prompt, max_tokens, temperature, block_size, chunk_queue):
    """Runs synchronously in a thread pool executor."""
    loop = asyncio.new_event_loop()
    try:
        for resp in stream_generate(
            model, draft, tokenizer, prompt,
            block_size=block_size,
            max_tokens=max_tokens,
            temperature=temperature,
        ):
            loop.run_until_complete(chunk_queue.put(("chunk", resp)))
        loop.run_until_complete(chunk_queue.put(("done", None)))
    finally:
        loop.close()


# ── Request / Response schemas ────────────────────────────────────────────────
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: Optional[str] = MODEL_ID
    messages: list[Message]
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.0
    stream: Optional[bool] = True
    block_size: Optional[int] = DEFAULT_BLOCK_SIZE  # DFlash-specific


# ── Helpers ───────────────────────────────────────────────────────────────────
def apply_chat_template(messages: list[Message]) -> str:
    parts = []
    for m in messages:
        if m.role == "system":
            parts.append(f"<|im_start|>system\n{m.content}<|im_end|>")
        elif m.role == "user":
            parts.append(f"<|im_start|>user\n{m.content}<|im_end|>")
        elif m.role == "assistant":
            parts.append(f"<|im_start|>assistant\n{m.content}<|im_end|>")
    parts.append("<|im_start|>assistant\n")
    return "\n".join(parts)


def sse_chunk(delta: str, request_id: str, finish_reason=None) -> str:
    payload = {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": MODEL_ID,
        "choices": [{
            "index": 0,
            "delta": {"content": delta} if delta else {},
            "finish_reason": finish_reason,
        }],
    }
    return f"data: {json.dumps(payload)}\n\n"


async def stream_response(
    prompt: str, max_tokens: int, temperature: float,
    block_size: Optional[int], request_id: str
) -> AsyncGenerator[str, None]:
    chunk_queue: asyncio.Queue = asyncio.Queue()
    qsize_before = queue.qsize()
    if qsize_before > 0:
        print(f"[queue] enqueued | {qsize_before} ahead in queue")
    await queue.put((prompt, max_tokens, temperature, block_size, chunk_queue))

    while True:
        kind, data = await chunk_queue.get()
        if kind == "error":
            yield f"data: {{\"error\": \"{data}\"}}\n\n"
            break
        if kind == "done":
            yield sse_chunk("", request_id, finish_reason="stop")
            yield "data: [DONE]\n\n"
            break
        if kind == "chunk":
            if data.text:
                yield sse_chunk(data.text, request_id)


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [{
            "id": MODEL_ID,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "local",
        }]
    }


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    prompt = apply_chat_template(req.messages)
    prompt_tokens = len(prompt) // 4
    request_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"

    if prompt_tokens > MAX_PROMPT_TOKENS:
        raise HTTPException(
            status_code=400,
            detail=f"Prompt too long ({prompt_tokens} tokens estimated). Limit is {MAX_PROMPT_TOKENS} to avoid GPU timeout."
        )

    print(f"[{request_id}] prompt tokens≈{prompt_tokens} | queue depth: {queue.qsize()}")

    if req.stream:
        return StreamingResponse(
            stream_response(prompt, req.max_tokens, req.temperature, req.block_size, request_id),
            media_type="text/event-stream",
            headers={"X-Queue-Depth": str(queue.qsize())},
        )
    else:
        # Non-streaming: collect all chunks
        full_text = ""
        async for chunk in stream_response(prompt, req.max_tokens, req.temperature, req.block_size, request_id):
            if chunk.startswith("data: {"):
                try:
                    data = json.loads(chunk[6:])
                    full_text += data["choices"][0]["delta"].get("content", "")
                except Exception:
                    pass
        return {
            "id": request_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": MODEL_ID,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": full_text},
                "finish_reason": "stop",
            }],
        }


@app.get("/health")
def health():
    return {"status": "ok", "queue_depth": queue.qsize() if queue else 0}


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    args = parser.parse_args()

    print(f"Starting DFlash MLX Server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
