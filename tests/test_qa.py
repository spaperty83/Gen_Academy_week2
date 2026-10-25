import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from park_rag.config import Settings
from park_rag.qa import FALLBACK, Claim, EvidenceAnswer, ParkAssistant, load_active, validate_answer

URL = "https://www.nps.gov/yell/planyourvisit/pets.htm"
EVIDENCE = [{"source_id": "S1", "title": "Pets", "url": URL,
             "collected_at": "2026-10-23T12:00:00+00:00", "text": "Pets must be leashed."}]


def test_invalid_citation_falls_back():
    output = EvidenceAnswer(status="supported", claims=[Claim(text="A claim", source_ids=["S99"])], gap="")
    result = validate_answer(output, EVIDENCE)
    assert result.text == FALLBACK
    assert result.sources == []


def test_partial_answer_retains_cited_sources_and_gap():
    output = EvidenceAnswer(status="partial", claims=[Claim(text="Pets must be leashed.", source_ids=["S1"])],
                            gap="The documents do not describe hotel pet fees.")
    result = validate_answer(output, EVIDENCE)
    assert "[S1]" in result.text
    assert "hotel pet fees" in result.text
    assert result.sources == EVIDENCE


def test_unsupported_output_never_displays_model_claims():
    output = EvidenceAnswer(status="unsupported", claims=[Claim(text="Invented", source_ids=["S1"])], gap="")
    assert validate_answer(output, EVIDENCE).text == FALLBACK


def assistant_with_empty_index():
    assistant = ParkAssistant.__new__(ParkAssistant)
    assistant.settings = Settings()
    assistant.snapshot = {"namespace": "yell-active", "sources": [{"source_url": URL}]}
    assistant.index, assistant.embeddings, assistant.chain = Mock(), Mock(), Mock()
    assistant.index.sparse.return_value = []
    assistant.index.dense.return_value = []
    assistant.graph = assistant.build_graph()
    return assistant


def test_empty_retrieval_skips_generation():
    assistant = assistant_with_empty_index()
    assert assistant.ask("A question?").text == FALLBACK
    assistant.chain.invoke.assert_not_called()


def test_question_limits():
    assistant = assistant_with_empty_index()
    with pytest.raises(ValueError):
        assistant.ask(" ")
    with pytest.raises(ValueError):
        assistant.ask("a" * 2001)


def test_settings_mismatch_requires_reingestion(tmp_path, monkeypatch):
    from park_rag import qa
    monkeypatch.setattr(qa, "ROOT", tmp_path)
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "active.json").write_text(json.dumps({"index_name": "another-index"}))
    with pytest.raises(ValueError, match="Re-ingest"):
        load_active(Settings())


def test_keys_not_in_settings_repr():
    settings = Settings(openai_api_key="private-one")
    assert "private" not in repr(settings)


def test_graph_retrieves_and_generates_cited_answer():
    from park_rag.store import LocalIndex
    assistant = ParkAssistant.__new__(ParkAssistant)
    assistant.settings = Settings(top_k=2)
    assistant.snapshot = {"sources": [{"source_url": URL}]}
    remote = Mock()
    remote.query.return_value = SimpleNamespace(matches=[SimpleNamespace(id="one")])
    assistant.index = LocalIndex([{"chunk_id": "one", "park_code": "yell", "park_name": "Yellowstone",
                                  "title": "Pets", "source_url": URL, "collected_at": "now",
                                  "text": "Pets must be leashed.", "vector": [1., 0.]}], remote, "active")
    assistant.embeddings = Mock()
    assistant.embeddings.embed_query.return_value = [1., 0.]
    assistant.chain = Mock()
    assistant.chain.invoke.return_value = EvidenceAnswer(status="supported", claims=[
        Claim(text="Pets must be leashed.", source_ids=["S1"])], gap="")
    assistant.graph = assistant.build_graph()
    result = assistant.ask("Yellowstone pets?")
    assert result.status == "supported"
    assert result.sources[0]["park_code"] == "yell"
    assert assistant.chain.invoke.call_count == 1


def test_question_routes_multiple_parks():
    from park_rag.qa import question_parks
    assert question_parks("Compare Shenandoah and Great Smoky Mountains") == ["shen", "grsm"]
    assert question_parks("Pet rules?") == []
