#!/usr/bin/env python
"""Reference local agent loop: shows the Yankee AI <-> Chroma DB orchestration end to end.

Two modes:

* ``--llm ollama``  : a real tool-calling loop against a local LLM served by Ollama on
                      127.0.0.1:11434 (e.g. ``ollama pull llama3.1:8b``). The LLM decides which
                      SafeHarbor tools to call; the loop executes them and feeds results back.
                      This is the same protocol Yankee AI uses with the MCP / HTTP tool servers.
* ``--llm none``    : (default, zero-dependency) a deterministic planner that runs the
                      "patient brief" workflow so the demo is reproducible on any laptop.

Everything runs under airgap(): if any step tried to reach the internet it would raise.

    python agent/local_agent.py "Brief me on the atrial fibrillation patients and current treatment guidance"
    python agent/local_agent.py --llm ollama --model llama3.1:8b "What causes cystic fibrosis?"
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import textwrap
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from safeharbor.phi_vault import PHIVault  # noqa: E402
from safeharbor.privacy import airgap_enabled  # noqa: E402
from safeharbor.tools import TOOL_SCHEMAS, SafeHarborTools  # noqa: E402

SYSTEM = ("You are a clinical assistant running entirely on a local machine. Answer only from tool "
          "results, cite doc_ids in [brackets], keep patient placeholders such as [PERSON_1] as-is, "
          "and never guess identifiers.")


def log(step: str, msg: str):
    print(f"\033[36m[{step}]\033[0m {msg}", flush=True)


def _norm(t: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", t.lower()) if w not in {"disease", "the", "of", "s"}}


def _same_topic(condition: str, topic: str | None) -> bool:
    a, b = _norm(condition), _norm(topic or "")
    return bool(a) and a <= b


# --------------------------------------------------------------------------- deterministic planner
def planner_run(tools: SafeHarborTools, question: str, reidentify: bool = False) -> str:
    log("plan", "1) find relevant patient notes  2) extract their conditions  "
                "3) pull treatment + complication guidance per condition  4) compose cited brief")
    lines: list[str] = []
    notes = tools.search_patient_notes(question, k=3)
    log("tool", f"search_patient_notes -> {len(notes['results'])} hits in {notes['latency_ms']} ms")
    conditions: dict[str, list[str]] = {}
    for r in notes["results"]:
        m = re.search(r"history of (.+?) and (.+?)\.", r["text"])
        conds = [c.strip() for c in m.groups()] if m else [r["topic"]]
        for c in conds:
            conditions.setdefault(c, []).append(r["doc_id"])
        lines.append(f"- Note [{r['doc_id']}]: {textwrap.shorten(r['text'], 220)}")

    lines.append("\nGuidance from local medical knowledge base:")
    for cond, ids in list(conditions.items())[:4]:
        for section in ("treatment", "complications"):
            kb = tools.search_medical_knowledge(f"{section} of {cond}", k=5)
            # Grounding check: only cite evidence about *this* condition. Nearest neighbours of a
            # missing section are often other diseases' pages, which must never be cited.
            on_topic = [h for h in kb["results"] if _same_topic(cond, h["topic"])]
            pick = next((h for h in on_topic if h["section"] == section), None)
            log("tool", f"search_medical_knowledge('{section} of {cond}') -> "
                        f"{len(kb['results'])} hits, {len(on_topic)} on-topic, "
                        f"{'cited ' + pick['doc_id'] if pick else 'NOT COVERED'} in {kb['latency_ms']} ms")
            if pick:
                lines.append(f"- {cond} - {section} [{pick['doc_id']}]: {textwrap.shorten(pick['text'], 260)}")
            else:
                lines.append(f"- {cond} - {section}: not covered by the local knowledge base (no citation).")

    answer = "Patient brief (de-identified):\n" + "\n".join(lines)
    if reidentify:
        vault = PHIVault(tools.cfg["paths"]["phi_vault"])
        for nid in {i for ids in conditions.values() for i in ids}:
            answer = re.sub(rf"(Note \[{nid}\]: )(.*)", lambda m, nid=nid: m.group(1) + vault.reidentify(nid, m.group(2)), answer)
        log("vault", "re-identified locally for the authorised viewer (never sent to the LLM)")
    return answer


# --------------------------------------------------------------------------- Ollama tool loop
def ollama_chat(model: str, messages: list, host: str) -> dict:
    body = json.dumps({"model": model, "messages": messages, "stream": False,
                       "tools": [{"type": "function", "function": t} for t in TOOL_SCHEMAS]}).encode()
    req = urllib.request.Request(f"{host}/api/chat", body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())["message"]


def llm_run(tools: SafeHarborTools, question: str, model: str, host: str, max_steps: int = 6) -> str:
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}]
    for _ in range(max_steps):
        msg = ollama_chat(model, messages, host)
        messages.append(msg)
        calls = msg.get("tool_calls") or []
        if not calls:
            return msg.get("content", "")
        for c in calls:
            fn = c["function"]
            args = fn.get("arguments") or {}
            if isinstance(args, str):
                args = json.loads(args)
            result = tools.call(fn["name"], args)
            log("tool", f"{fn['name']}({json.dumps(args)}) -> {len(result.get('results', []))} results")
            messages.append({"role": "tool", "content": json.dumps(result)[:6000]})
    return "(stopped after max_steps)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--llm", choices=["none", "ollama"], default="none")
    ap.add_argument("--model", default="llama3.1:8b")
    ap.add_argument("--ollama-host", default="http://127.0.0.1:11434")
    ap.add_argument("--reidentify", action="store_true", help="restore PHI for local display only")
    a = ap.parse_args()

    tools = SafeHarborTools(airgap=True)
    log("privacy", f"airgap={'ON' if airgap_enabled() else 'OFF'}  stats={json.dumps(tools.index_stats()['collections'])}")
    if a.llm == "ollama":
        out = llm_run(tools, a.question, a.model, a.ollama_host)
    else:
        out = planner_run(tools, a.question, a.reidentify)
    print("\n" + out)


if __name__ == "__main__":
    main()
