"""Small end-to-end test: in-memory Chroma + real embedder + hybrid retrieval."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from safeharbor.chunking import iter_chunks, make_token_counter  # noqa: E402
from safeharbor.store import create_collection, get_client, ingest  # noqa: E402
from safeharbor.retriever import Retriever, route_section  # noqa: E402

DOCS = [
    {"doc_id": "d1", "topic": "Cystic Fibrosis", "section": "genetic changes",
     "text": "Mutations in the CFTR gene cause cystic fibrosis. The CFTR protein is a chloride channel."},
    {"doc_id": "d2", "topic": "Asthma", "section": "treatment",
     "text": "Inhaled corticosteroids are the main long-term control medicine. Quick-relief inhalers open airways."},
    {"doc_id": "d3", "topic": "Migraine", "section": "symptoms",
     "text": "Migraine causes throbbing headache, nausea, and sensitivity to light and sound."},
]


@pytest.fixture(scope="module")
def retr():
    from safeharbor.embeddings import LocalEmbedder
    emb = LocalEmbedder("BAAI/bge-small-en-v1.5")
    ids, texts, metas = map(list, zip(*iter_chunks(DOCS, make_token_counter(emb.tokenizer), 64, 1)))
    col = create_collection(get_client(None), "test-kb", {"max_neighbors": 16, "ef_construction": 50, "ef_search": 20})
    ingest(col, ids, texts, metas, emb.embed_documents(texts))
    return Retriever(col, emb)


@pytest.mark.parametrize("mode", ["dense", "bm25", "hybrid"])
def test_modes_find_exact_gene_symbol(retr, mode):
    assert retr.search("which gene is CFTR", k=1, mode=mode)[0].doc_id == "d1"


def test_paraphrase_dense(retr):
    assert retr.search("pounding head pain and light bothers me", k=1, mode="hybrid")[0].doc_id == "d3"


def test_metadata_filter(retr):
    hits = retr.search("medicine", k=3, mode="hybrid", where={"section": "treatment"})
    assert [h.doc_id for h in hits] == ["d2"]


def test_router():
    assert route_section("what are the symptoms of lupus") == "symptoms"
    assert route_section("is it inherited") == "inheritance"
