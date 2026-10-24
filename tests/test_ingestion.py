from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from langchain_core.documents import Document

from park_rag import ingest
from park_rag.config import Settings
from park_rag.store import upsert_chunks


def test_rerun_upsert_overwrites_same_record_ids():
    class MemoryIndex:
        def __init__(self):
            self.records = {}

        def upsert(self, *, vectors, namespace):
            for vector in vectors:
                self.records[namespace, vector["id"]] = vector

    index, embeddings = MemoryIndex(), Mock()
    embeddings.embed_documents.return_value = [[0.1, 0.2]]
    chunks = [Document(page_content="A passage", metadata={"chunk_id": "stable-0000", "title": "Pets"})]
    upsert_chunks(index, embeddings, chunks, "yell-snapshot")
    upsert_chunks(index, embeddings, chunks, "yell-snapshot")
    assert len(index.records) == 1


def test_failed_publication_keeps_active_pointer(tmp_path, monkeypatch):
    monkeypatch.setattr(ingest, "ROOT", tmp_path)
    (tmp_path / "data").mkdir()
    active = tmp_path / "data" / "active.json"
    active.write_text('{"previous":"corpus"}')
    monkeypatch.setattr(ingest, "read_sources", lambda _: ["url"])
    monkeypatch.setattr(ingest, "collect", lambda _: [Document(
        page_content="A passage", metadata={"document_id": "stable", "title": "Pets",
                                             "source_url": "url", "collected_at": "now"})])
    monkeypatch.setattr(ingest, "connect_index", lambda *a, **k: SimpleNamespace())
    monkeypatch.setattr(ingest, "make_embeddings", lambda _: Mock())
    monkeypatch.setattr(ingest, "upsert_chunks", lambda *a: None)
    def fail_visibility(*args):
        raise TimeoutError("not visible")
    monkeypatch.setattr(ingest, "wait_for_records", fail_visibility)
    with pytest.raises(TimeoutError):
        ingest.run(Settings(openai_api_key="test", pinecone_api_key="test"), tmp_path / "sources.json")
    assert active.read_text() == '{"previous":"corpus"}'
