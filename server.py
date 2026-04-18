"""
DFlash MLX Server — OpenAI-compatible API with request queue
Usage: python server.py [--port 8080] [--config config.toml]
"""
import argparse
import asyncio
import json
import time
import tomllib
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

import mlx.core as mx
from huggingface_hub import snapshot_download
from huggingface_hub.utils import LocalEntryNotFoundError
from dflash.model_mlx import load, load_draft, stream_generate


def _load_config(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    with open(p, "rb") as f:
        return tomllib.load(f)


# Config is parsed at startup after argparse runs; placeholders here.
MODEL_PATH = DRAFT_ID = MODEL_ID = None
CONTEXT_LENGTH = MAX_PROMPT_TOKENS = DEFAULT_BLOCK_SIZE = None


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
            print(f"  [ERROR] {e}", flush=True)
            await chunk_queue.put(("error", str(e)))
        finally:
            queue.task_done()


def _run_inference(prompt, max_tokens, temperature, block_size, chunk_queue):
    """Runs synchronously in a thread pool executor."""
    loop = asyncio.new_event_loop()
    t0 = time.perf_counter()
    token_count = 0
    last_print = t0
    try:
        for resp in stream_generate(
            model, draft, tokenizer, prompt,
            block_size=block_size,
            max_tokens=max_tokens,
            temperature=temperature,
        ):
            loop.run_until_complete(chunk_queue.put(("chunk", resp)))
            token_count += 1
            now = time.perf_counter()
            if now - last_print >= 2.0:
                tps = token_count / (now - t0)
                print(f"  ... {token_count} tokens | {tps:.1f} tok/s", flush=True)
                last_print = now
        elapsed = time.perf_counter() - t0
        tps = token_count / elapsed if elapsed > 0 else 0
        print(f"  done: {token_count} tokens | {tps:.1f} tok/s | {elapsed:.1f}s", flush=True)
        loop.run_until_complete(chunk_queue.put(("done", None)))
    finally:
        loop.close()


import re as _re

def _clean(text: str) -> str:
    text = _re.sub(r"<think>.*?</think>", "", text, flags=_re.DOTALL)
    text = text.replace("<|im_end|>", "").strip()
    return text


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

    buffer = ""
    in_think = False
    while True:
        kind, data = await chunk_queue.get()
        if kind == "error":
            yield f"data: {{\"error\": \"{data}\"}}\n\n"
            break
        if kind == "done":
            tail = _clean(buffer)
            if tail:
                yield sse_chunk(tail, request_id)
            yield sse_chunk("", request_id, finish_reason="stop")
            yield "data: [DONE]\n\n"
            break
        if kind == "chunk" and data.text:
            buffer += data.text
            if "<think>" in buffer:
                in_think = True
            if in_think:
                if "</think>" in buffer:
                    buffer = buffer.split("</think>", 1)[1]
                    in_think = False
                continue
            out = buffer.replace("<|im_end|>", "")
            if out and not out.endswith("<"):
                yield sse_chunk(out, request_id)
                buffer = ""


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
            "context_length": CONTEXT_LENGTH,
        }]
    }


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    prompt = apply_chat_template(req.messages)
    prompt_tokens = len(prompt) // 4
    request_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"

    if MAX_PROMPT_TOKENS is not None and prompt_tokens > MAX_PROMPT_TOKENS:
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
                "message": {"role": "assistant", "content": _clean(full_text)},
                "finish_reason": "stop",
            }],
        }


@app.get("/health")
def health():
    return {"status": "ok", "queue_depth": queue.qsize() if queue else 0}


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--host", type=str, default=None)
    args = parser.parse_args()

    cfg = _load_config("config.toml")

    MODEL_PATH         = cfg.get("model", {}).get("path",           "mlx-community/Qwen3.5-4B-4bit")
    DRAFT_ID           = cfg.get("model", {}).get("draft_id",       "z-lab/Qwen3.5-4B-DFlash")
    MODEL_ID           = cfg.get("model", {}).get("id",             "qwen3.5-4b-dflash")
    CONTEXT_LENGTH     = cfg.get("model", {}).get("context_length", 32768)
    MAX_PROMPT_TOKENS  = cfg.get("inference", {}).get("max_prompt_tokens",  None)
    DEFAULT_BLOCK_SIZE = cfg.get("inference", {}).get("default_block_size", 4)
    host = args.host or cfg.get("server", {}).get("host", "127.0.0.1")
    port = args.port or cfg.get("server", {}).get("port", 8080)

    import sys
    _mod = sys.modules[__name__]
    for _k, _v in [("MODEL_PATH", MODEL_PATH), ("DRAFT_ID", DRAFT_ID), ("MODEL_ID", MODEL_ID),
                   ("CONTEXT_LENGTH", CONTEXT_LENGTH), ("MAX_PROMPT_TOKENS", MAX_PROMPT_TOKENS),
                   ("DEFAULT_BLOCK_SIZE", DEFAULT_BLOCK_SIZE)]:
        setattr(_mod, _k, _v)

    print(f"Starting DFlash MLX Server on {host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="warning")
