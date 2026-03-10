"""
Web UI server for NL-to-MQL inference.

Serves a sleek dark-mode interface and exposes the SmolLM3-3B
inference endpoint via FastAPI.

Usage:
    .venv/bin/python app.py
    # Then open http://localhost:8000
"""

import json
import logging
import time
from pathlib import Path
from typing import Callable

import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from NLtoMQL_SLM import (
    DEFAULT_ADAPTER_DIR,
    create_prompt,
    generate_mql,
    get_device_and_dtype,
)

app = FastAPI(title="NL → MQL", version="2.0")

# ── Logging Setup ────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("api")

@app.middleware("http")
async def log_requests(request: Request, call_next: Callable):
    start_time = time.perf_counter()
    
    # Log incoming request
    client_ip = request.client.host if request.client else "unknown"
    logger.info(f"Incoming request: {request.method} {request.url.path} from {client_ip}")
    
    response = await call_next(request)
    
    # Log outgoing response
    process_time = (time.perf_counter() - start_time) * 1000
    logger.info(
        f"Completed request: {request.method} {request.url.path} - "
        f"Status: {response.status_code} - "
        f"Time: {process_time:.2f}ms"
    )
    
    return response


# ── Request / Response models ────────────────────────────────────────────
class InferRequest(BaseModel):
    nl_query: str
    temperature: float = 0.2
    max_new_tokens: int = 1000


class InferResponse(BaseModel):
    mql: str
    device: str
    elapsed_seconds: float


# ── API endpoint ─────────────────────────────────────────────────────────
@app.post("/api/infer", response_model=InferResponse)
async def infer(req: InferRequest):
    if not req.nl_query.strip():
        raise HTTPException(status_code=400, detail="nl_query cannot be empty")

    device, dtype = get_device_and_dtype()
    t0 = time.perf_counter()

    try:
        mql = generate_mql(
            nl_query=req.nl_query,
            adapter_dir=DEFAULT_ADAPTER_DIR,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
        )
    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail="Model adapter not found. Run training first: "
            ".venv/bin/python NLtoMQL_SLM.py train --epochs 3",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    elapsed = time.perf_counter() - t0
    return InferResponse(mql=mql, device=device, elapsed_seconds=round(elapsed, 2))


# ── Health check ─────────────────────────────────────────────────────────
@app.get("/api/health")
async def health():
    device, dtype = get_device_and_dtype()
    adapter_exists = Path(DEFAULT_ADAPTER_DIR).exists()
    return {
        "status": "ok",
        "device": device,
        "dtype": str(dtype),
        "adapter_ready": adapter_exists,
        "model": "HuggingFaceTB/SmolLM3-3B",
    }


# ── Serve the UI ─────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = Path(__file__).parent / "static" / "index.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


# Mount static assets
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


if __name__ == "__main__":
    import uvicorn

    print("\n🚀 NL → MQL Web UI starting at http://localhost:8000\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
