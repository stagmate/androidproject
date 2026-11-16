#!/usr/bin/env python
"""Loopback-only HTTP tool server (OpenAI function-calling format) for Yankee AI.

    python server/http_tool_server.py            # serves on http://127.0.0.1:8765

    GET  /v1/tools                 -> tool list in OpenAI "tools" format (paste into the agent)
    POST /v1/tools/{name}          -> {"arguments": {...}}  => tool result JSON
    GET  /health

The server refuses to bind to anything but loopback, and the process runs in airgap mode, so the
retrieval layer cannot send data off the machine even if a tool argument tried to make it.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import uvicorn  # noqa: E402
from fastapi import FastAPI, HTTPException  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from safeharbor.tools import TOOL_SCHEMAS, SafeHarborTools  # noqa: E402

app = FastAPI(title="SafeHarbor-RAG tool server", version="0.1.0")
_tools: SafeHarborTools | None = None


class Invoke(BaseModel):
    arguments: dict = {}


def tools() -> SafeHarborTools:
    global _tools
    if _tools is None:
        _tools = SafeHarborTools()
    return _tools


@app.get("/health")
def health():
    return {"ok": True, **tools().index_stats()}


@app.get("/v1/tools")
def list_tools():
    return {"tools": [{"type": "function", "function": t} for t in TOOL_SCHEMAS]}


@app.post("/v1/tools/{name}")
def invoke(name: str, body: Invoke):
    try:
        return tools().call(name, body.arguments)
    except (ValueError, TypeError) as e:
        raise HTTPException(400, str(e))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()
    if a.host not in ("127.0.0.1", "localhost", "::1"):
        sys.exit("refusing to bind to a non-loopback interface: this server is local-only by design")
    tools()  # load models before the airgap closes and before accepting requests
    uvicorn.run(app, host=a.host, port=a.port, log_level="info")
