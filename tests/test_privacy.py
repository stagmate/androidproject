import socket
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from safeharbor.data import generate_notes  # noqa: E402
from safeharbor.phi_vault import PHIVault  # noqa: E402
from safeharbor.privacy import EgressBlocked, PHIRedactor, airgap  # noqa: E402


def test_airgap_blocks_outbound_but_allows_loopback():
    with airgap():
        with pytest.raises(EgressBlocked):
            socket.create_connection(("93.184.216.34", 443), timeout=1)
        with pytest.raises(EgressBlocked):
            socket.getaddrinfo("huggingface.co", 443)
        srv = socket.socket(); srv.bind(("127.0.0.1", 0)); srv.listen(1)
        c = socket.create_connection(srv.getsockname(), timeout=1)  # loopback is fine
        c.close(); srv.close()
    socket.getaddrinfo("localhost", 80)  # restored afterwards


def test_redaction_removes_all_identifiers_and_roundtrips(tmp_path):
    notes = generate_notes(50, seed=1)
    red = PHIRedactor()
    vault = PHIVault(tmp_path / "v.sqlite")
    for n in notes:
        r = red.redact(n["text"], n["_known_names"])
        for name in n["_known_names"]:
            for part in name.split():
                assert part not in r.text, (part, r.text)
        for v in r.mapping.values():
            assert v not in r.text
        vault.store(n["doc_id"], r.mapping)
        assert vault.reidentify(n["doc_id"], r.text) == n["text"]


def test_redaction_keeps_clinical_content():
    text = "Patient: Ana Ruiz. DOB: 01/02/1950. On metformin 500 mg twice daily; HbA1c 8.1%."
    r = PHIRedactor().redact(text)
    assert "metformin 500 mg" in r.text and "HbA1c 8.1%" in r.text
    assert "Ana" not in r.text and "01/02/1950" not in r.text
