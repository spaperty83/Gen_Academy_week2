from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from langchain_core.documents import Document

from park_rag import ingest
from park_rag.config import Settings
from park_rag.store import LocalIndex, build_index, fuse


def records():
    return [{"chunk_id": code, "park_code": code, "park_name": name, "title": "Pets",
             "source_url": f"https://www.nps.gov/{code}/planyourvisit/pets.htm",
             "collected_at": "now", "text": "Pets hiking trails guidance", "vector": vector}
            for code, name, vector in [("yell", "Yellowstone", [1., 0.]),
                                       ("shen", "Shenandoah", [0., 1.]),
                                       ("grsm", "Great Smoky Mountains", [0.6, 0.8])]]


def test_dense_and_bm25_filter_before_ranking():
    remote = Mock()
    remote.query.return_value = SimpleNamespace(matches=[SimpleNamespace(id="shen")])
    index = LocalIndex(records(), remote, "active")
    assert index.dense([1, 0], 2, ["shen"]) == [1]
    assert index.sparse("Shenandoah", 2, ["shen"]) == [1]
    assert remote.query.call_args.kwargs["filter"] == {"park_code": {"$in": ["shen"]}}
    assert remote.query.call_args.kwargs["namespace"] == "active"


def test_fusion_deduplicates_and_checks_allowlist():
    remote = Mock()
    remote.query.return_value = SimpleNamespace(matches=[SimpleNamespace(id="shen")])
    index = LocalIndex(records(), remote, "active")
    evidence = fuse(index, [1, 0], [1, 2], [{"source_url": records()[1]["source_url"]}], 5)
    assert len(evidence) == 1
    assert evidence[0]["park_code"] == "shen"


def test_invalid_embeddings_never_publish_index(tmp_path):
    embeddings = Mock()
    embeddings.embed_documents.return_value = [[float("nan"), 0]]
    path = tmp_path / "index.json"
    with pytest.raises(ValueError):
        build_index(embeddings, [Document(page_content="pets", metadata={"title": "Pets", "chunk_id": "one"})],
                    2, path, index=Mock(), namespace="active")
    assert not path.exists()


def test_failed_publication_keeps_active_pointer(tmp_path, monkeypatch):
    monkeypatch.setattr(ingest, "ROOT", tmp_path)
    (tmp_path / "data").mkdir()
    active = tmp_path / "data" / "active.json"
    active.write_text('{"previous":"corpus"}')
    monkeypatch.setattr(ingest, "read_sources", lambda _: ["url"])
    monkeypatch.setattr(ingest, "collect", lambda _: [Document(
        page_content="A passage", metadata={"document_id": "stable", "title": "Pets", "park_code": "shen",
                                             "source_url": "url", "collected_at": "now"})])
    monkeypatch.setattr(ingest, "make_embeddings", lambda _: Mock())
    monkeypatch.setattr(ingest, "connect_index", lambda *a, **k: Mock())
    def fail(*args, **kwargs):
        raise TimeoutError("failed")
    monkeypatch.setattr(ingest, "build_index", fail)
    with pytest.raises(TimeoutError):
        ingest.run(Settings(openai_api_key="test", pinecone_api_key="test"), tmp_path / "sources.json")
    assert active.read_text() == '{"previous":"corpus"}'


def test_vectors_uploaded_and_only_text_saved_locally(tmp_path):
    import json
    embeddings, remote = Mock(), Mock()
    embeddings.embed_documents.return_value = [[3., 4.]]
    remote.fetch.return_value = SimpleNamespace(vectors={"one": {}})
    chunk = Document(page_content="Pets", metadata={"chunk_id": "one", "title": "Pets", "park_code": "shen"})
    path = tmp_path / "corpus.json"
    build_index(embeddings, [chunk], 2, path, index=remote, namespace="snapshot")
    assert remote.upsert.call_args.kwargs["vectors"][0]["values"] == [0.6, 0.8]
    assert remote.upsert.call_args.kwargs["namespace"] == "snapshot"
    assert "vector" not in json.loads(path.read_text())["records"][0]


def test_index_configuration_checked(monkeypatch):
    from park_rag import store
    remote = Mock()
    remote.has_index.return_value = True
    remote.describe_index.return_value = SimpleNamespace(dimension=1024, metric="cosine")
    monkeypatch.setattr(store, "Pinecone", lambda **kwargs: remote)
    with pytest.raises(ValueError, match="1536"):
        store.connect_index(Settings())


def test_failed_remote_verification_does_not_publish_local_corpus(tmp_path, monkeypatch):
    from park_rag import store
    embeddings, remote = Mock(), Mock()
    embeddings.embed_documents.return_value = [[1., 0.]]
    def fail(*args):
        raise TimeoutError("not visible")
    monkeypatch.setattr(store, "wait_for_records", fail)
    path = tmp_path / "corpus.json"
    with pytest.raises(TimeoutError):
        build_index(embeddings, [Document(page_content="Pets", metadata={"chunk_id": "one", "title": "Pets"})],
                    2, path, index=remote, namespace="new")
    assert not path.exists()
