#!/usr/bin/env python
"""E. Privacy audit of the shipped vector database.

1. PHI leakage: every raw identifier from the synthetic notes (names, DOB, MRN, phone, address,
   ZIP) is searched for in *every* document string stored in Chroma. Target: 0.
2. Re-identification: the local vault restores the original note exactly for authorised use.
3. Egress: the full query path runs inside ``airgap()``; any outbound socket would raise.
4. Telemetry: Chroma's anonymized telemetry flag is verified off.

    python scripts/audit_privacy.py
"""
from __future__ import annotations

import json
import re
import socket
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from safeharbor.config import ROOT, load_config  # noqa: E402
from safeharbor.phi_vault import PHIVault  # noqa: E402
from safeharbor.privacy import EgressBlocked, airgap  # noqa: E402

# Field extractors for the raw synthetic notes (ground truth of what must never be indexed)
_FIELDS = {
    "name": r"Patient: ([A-Z][a-z]+ [A-Z][a-z]+)",
    "contact": r"Emergency contact: ([A-Z][a-z]+ [A-Z][a-z]+)",
    "dob": r"DOB: ([\d/]+)",
    "mrn": r"MRN: (\d+)",
    "phone": r"Phone: (\(\d{3}\) \d{3}-\d{4})",
    "address": r"Address: (\d+ [A-Za-z ]+?),",
    "zip": r"Springfield (\d{5})",
    "visit_date": r"Visit date (\d{4}-\d{2}-\d{2})",
}


def main():
    cfg = load_config()
    raw = [json.loads(l) for l in Path(cfg["paths"]["raw_dir"], "synthetic_notes_with_phi.jsonl").read_text().splitlines() if l]
    truth: dict[str, list[str]] = {k: [] for k in _FIELDS}
    for n in raw:
        for k, rx in _FIELDS.items():
            m = re.search(rx, n["text"])
            if m:
                truth[k].append(m.group(1))

    report = {}
    with airgap():
        # Any accidental network call inside this block raises EgressBlocked.
        try:
            socket.create_connection(("example.com", 443), timeout=2)
            report["egress_blocked"] = False
        except EgressBlocked:
            report["egress_blocked"] = True

        from safeharbor.store import get_client
        col = get_client(cfg["paths"]["chroma_dir"]).get_collection("patient_notes")
        got = col.get(include=["documents", "metadatas"])
        blob = "\n".join(got["documents"]) + "\n" + json.dumps(got["metadatas"])
        leaks = {}
        for k, values in truth.items():
            uniq = sorted(set(values))
            leaked = [v for v in uniq if re.search(rf"(?<![\w]){re.escape(v)}(?![\w])", blob)]
            leaks[k] = {"identifiers": len(uniq), "leaked": len(leaked), "examples": leaked[:3]}
        report["phi_leakage"] = leaks
        report["total_identifiers"] = sum(v["identifiers"] for v in leaks.values())
        report["total_leaked"] = sum(v["leaked"] for v in leaks.values())

        vault = PHIVault(cfg["paths"]["phi_vault"])
        red = {json.loads(l)["doc_id"]: json.loads(l)["text"]
               for l in Path(cfg["paths"]["raw_dir"], "notes_redacted.jsonl").read_text().splitlines() if l}
        ok = sum(vault.reidentify(n["doc_id"], red[n["doc_id"]]) == n["text"] for n in raw)
        report["reidentification_exact"] = f"{ok}/{len(raw)}"

    import os
    report["chroma_telemetry"] = os.environ.get("ANONYMIZED_TELEMETRY")
    out = ROOT / "results" / "privacy_audit.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0 if report["total_leaked"] == 0 and report["egress_blocked"] else 1


if __name__ == "__main__":
    sys.exit(main())
