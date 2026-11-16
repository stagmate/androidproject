"""Privacy controls: telemetry off, an outbound-network kill switch, and PHI redaction.

Three layers, each checkable in tests:

1. ``disable_telemetry()``  - turns off Chroma / HF / tokenizers phone-home before import.
2. ``airgap()``             - monkey-patches ``socket`` so any non-loopback connection raises
                              ``EgressBlocked``. Used at query time: once models are on disk the
                              whole retrieval + agent loop runs with the network physically refused.
3. ``PHIRedactor``          - deterministic de-identification of clinical free text *before* it is
                              chunked or embedded. Identifiers are replaced by stable pseudonyms
                              (``[PATIENT_1]``, ``[DATE_3]``) and the mapping goes to a local SQLite
                              vault (``phi_vault.py``) that the LLM never sees.
"""
from __future__ import annotations

import ipaddress
import os
import re
import socket
from contextlib import contextmanager
from dataclasses import dataclass, field


def disable_telemetry() -> None:
    os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")  # chromadb
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def go_offline() -> None:
    """Force Hugging Face libraries to use only the local cache."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"


# ---------------------------------------------------------------------------
# Egress guard
# ---------------------------------------------------------------------------
class EgressBlocked(ConnectionRefusedError):
    """Raised when code tries to open a non-loopback network connection in airgap mode."""


_original_connect = socket.socket.connect
_original_connect_ex = socket.socket.connect_ex
_original_getaddrinfo = socket.getaddrinfo
_airgap_on = False


def _is_local(host) -> bool:
    if host is None:
        return True
    if isinstance(host, bytes):
        host = host.decode()
    host = str(host)
    if host in ("localhost", "") or host.endswith(".localhost"):
        return True
    try:
        ip = ipaddress.ip_address(host.split("%")[0])
        return ip.is_loopback or ip.is_unspecified
    except ValueError:
        return False


def _guard_addr(address):
    if isinstance(address, tuple) and address:
        if not _is_local(address[0]):
            raise EgressBlocked(f"airgap: outbound connection to {address[0]!r} blocked")
    # AF_UNIX paths (str/bytes) are local by definition.


def _guarded_connect(self, address):
    _guard_addr(address)
    return _original_connect(self, address)


def _guarded_connect_ex(self, address):
    _guard_addr(address)
    return _original_connect_ex(self, address)


def _guarded_getaddrinfo(host, *args, **kwargs):
    if not _is_local(host):
        raise EgressBlocked(f"airgap: DNS lookup for {host!r} blocked")
    return _original_getaddrinfo(host, *args, **kwargs)


def enable_airgap() -> None:
    global _airgap_on
    disable_telemetry()
    go_offline()
    socket.socket.connect = _guarded_connect
    socket.socket.connect_ex = _guarded_connect_ex
    socket.getaddrinfo = _guarded_getaddrinfo
    _airgap_on = True


def disable_airgap() -> None:
    global _airgap_on
    socket.socket.connect = _original_connect
    socket.socket.connect_ex = _original_connect_ex
    socket.getaddrinfo = _original_getaddrinfo
    _airgap_on = False


def airgap_enabled() -> bool:
    return _airgap_on


@contextmanager
def airgap():
    was_on = _airgap_on
    enable_airgap()
    try:
        yield
    finally:
        if not was_on:
            disable_airgap()


# ---------------------------------------------------------------------------
# PHI redaction
# ---------------------------------------------------------------------------
# Order matters: more specific patterns first so e.g. an MRN is not eaten by PHONE.
_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
    ("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("MRN", re.compile(r"\b(?:MRN|Medical Record(?: Number)?)[:#\s]*([A-Z]{0,3}\d{6,10})\b", re.I)),
    ("PHONE", re.compile(r"(?:\+?1[\s.-]?)?\(?\b\d{3}\)?[\s.-]?\d{3}[\s.-]\d{4}\b")),
    ("DATE", re.compile(
        r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2}|"
        r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.? \d{1,2},? \d{4})\b")),
    ("ZIP", re.compile(r"\b\d{5}(?:-\d{4})?\b")),
    ("ADDRESS", re.compile(
        r"\b\d{1,5} [A-Z][a-z]+(?: [A-Z][a-z]+)* (?:Street|St|Avenue|Ave|Road|Rd|Lane|Ln|Drive|Dr|Boulevard|Blvd|Court|Ct)\b\.?")),
]
# Names are found via context cues ("Patient: X Y", "Mr. X", "Dr. X") plus any name the caller
# already knows (e.g. from the structured header of an EHR export).
_NAME_CUE = re.compile(
    r"\b(?:(?:Patient|Pt|Name|Contact|Emergency contact|Son|Daughter|Wife|Husband|Mother|Father)\s*:\s*"
    r"|(?:Mr|Mrs|Ms|Miss|Dr)\.?\s+)"
    r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})"
)


@dataclass
class RedactionResult:
    text: str
    mapping: dict[str, str] = field(default_factory=dict)  # placeholder -> original

    @property
    def n_redacted(self) -> int:
        return len(self.mapping)


class PHIRedactor:
    """Deterministic, dependency-free de-identifier for the HIPAA Safe-Harbor identifier classes
    that appear in free-text notes (names, dates, phone, email, SSN, MRN, address, ZIP).

    Placeholders are stable *within a document*, so "[PATIENT_1]" refers to the same person
    everywhere in that note and retrieval/LLM reasoning still works on the redacted text.
    """

    def __init__(self, extra_names: list[str] | None = None):
        self.extra_names = [n for n in (extra_names or []) if n]

    def redact(self, text: str, known_names: list[str] | None = None) -> RedactionResult:
        mapping: dict[str, str] = {}
        reverse: dict[tuple[str, str], str] = {}
        counters: dict[str, int] = {}

        def placeholder(kind: str, value: str) -> str:
            key = (kind, value.strip().lower())
            if key not in reverse:
                counters[kind] = counters.get(kind, 0) + 1
                ph = f"[{kind}_{counters[kind]}]"
                reverse[key] = ph
                mapping[ph] = value.strip()
            return reverse[key]

        # 1. Names: collect from cues + known list, then replace full names and their parts.
        names = set(self.extra_names) | set(known_names or [])
        for m in _NAME_CUE.finditer(text):
            names.add(m.group(1))
        # Longest first so "Maria Lopez" wins over "Maria".
        name_parts: dict[str, str] = {}
        for full in sorted(names, key=len, reverse=True):
            ph = placeholder("PERSON", full)
            name_parts.setdefault(full, ph)
            parts = full.split()
            if len(parts) > 1:
                # A bare given/family name gets its own sub-placeholder ([PERSON_1.2] = 2nd part of
                # person 1) so the vault can restore the exact surface form.
                for i, part in enumerate(parts, 1):
                    if len(part) > 2 and part not in name_parts:
                        sub = f"{ph[:-1]}.{i}]"
                        mapping[sub] = part
                        name_parts[part] = sub
        for name in sorted(name_parts, key=len, reverse=True):
            text = re.sub(rf"\b{re.escape(name)}\b", name_parts[name], text)

        # 2. Structured identifiers.
        for kind, pat in _PATTERNS:
            def _sub(m, kind=kind):
                value = m.group(1) if (kind == "MRN" and m.lastindex) else m.group(0)
                ph = placeholder(kind, value)
                return m.group(0).replace(value, ph) if kind == "MRN" else ph
            text = pat.sub(_sub, text)

        return RedactionResult(text=text, mapping=mapping)
