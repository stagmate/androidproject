"""Local re-identification vault.

The vector DB and the LLM only ever see pseudonymised text. The placeholder -> real value map
lives in a separate SQLite file so that an authorised clinician's UI can re-identify a result
*after* the agent has answered, on the same machine. Deleting this file irreversibly
anonymises the index.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path


class PHIVault:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path)
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS phi (doc_id TEXT, placeholder TEXT, value TEXT, "
            "PRIMARY KEY (doc_id, placeholder))"
        )

    def store(self, doc_id: str, mapping: dict[str, str]) -> None:
        self.db.executemany(
            "INSERT OR REPLACE INTO phi VALUES (?, ?, ?)",
            [(doc_id, k, v) for k, v in mapping.items()],
        )
        self.db.commit()

    def reidentify(self, doc_id: str, text: str) -> str:
        rows = self.db.execute("SELECT placeholder, value FROM phi WHERE doc_id = ?", (doc_id,))
        for ph, value in sorted(rows, key=lambda r: -len(r[0])):
            text = text.replace(ph, value)
        return text

    def count(self) -> int:
        return self.db.execute("SELECT COUNT(*) FROM phi").fetchone()[0]

    def close(self) -> None:
        self.db.close()
