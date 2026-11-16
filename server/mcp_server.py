#!/usr/bin/env python
"""MCP (Model Context Protocol) server exposing the SafeHarbor Chroma DB to Yankee AI.

Transport is stdio: Yankee AI (or any MCP-capable agent host) launches this process locally and
talks to it over pipes, so there is no listening port at all. Register it in the agent host's MCP
configuration, e.g.

    {
      "mcpServers": {
        "safeharbor": {
          "command": "python",
          "args": ["/abs/path/to/server/mcp_server.py"]
        }
      }
    }
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mcp.server.fastmcp import FastMCP  # noqa: E402

from safeharbor.tools import SafeHarborTools  # noqa: E402

tools = SafeHarborTools()
mcp = FastMCP("safeharbor-chroma")


@mcp.tool()
def search_medical_knowledge(query: str, k: int = 5, section: str | None = None, exact_terms: bool = False) -> str:
    """Search the local NIH-derived medical knowledge base. Optional section filter: symptoms,
    treatment, causes, inheritance, genetic changes, exams and tests, outlook, prevention, ...
    Set exact_terms=true when the query hinges on gene symbols, drug names or codes."""
    return json.dumps(tools.search_medical_knowledge(query, k, section, exact_terms))


@mcp.tool()
def search_patient_notes(query: str, k: int = 5, condition: str | None = None) -> str:
    """Search de-identified local clinical notes (PHI replaced by placeholders like [PERSON_1])."""
    return json.dumps(tools.search_patient_notes(query, k, condition))


@mcp.tool()
def get_document(doc_id: str) -> str:
    """Fetch a full knowledge article or patient note by doc_id."""
    return json.dumps(tools.get_document(doc_id))


@mcp.tool()
def index_stats() -> str:
    """Describe the local vector database and its privacy mode."""
    return json.dumps(tools.index_stats())


if __name__ == "__main__":
    mcp.run()  # stdio
