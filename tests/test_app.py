from streamlit.testing.v1 import AppTest

from park_rag.config import ROOT, Settings


def test_app_startup_without_credentials(monkeypatch):
    monkeypatch.setattr(Settings, "load", classmethod(lambda cls: Settings()))
    app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=15)
    assert not app.exception
    assert app.title[0].value == "National parks NPS assistant"
    assert "OPENAI_API_KEY" in app.warning[0].value


def test_app_renders_answer_links_and_collection_dates(monkeypatch):
    from park_rag import qa
    settings = Settings(openai_api_key="test", pinecone_api_key="test")
    snapshot = {"page_count": 1, "chunk_count": 1, "sources": [{
        "title": "Pets", "source_url": "https://www.nps.gov/yell/planyourvisit/pets.htm",
        "collected_at": "2026-10-23T12:00:00+00:00",
    }]}
    source = {"source_id": "S1", "title": "Pets", "url": snapshot["sources"][0]["source_url"],
              "collected_at": snapshot["sources"][0]["collected_at"], "text": "Pets must be leashed."}
    class FakeAssistant:
        def __init__(self, settings, snapshot):
            pass

        def ask(self, question):
            return qa.Answer("supported", "Pets must be leashed. [S1]", [source])
    monkeypatch.setattr(Settings, "load", classmethod(lambda cls: settings))
    monkeypatch.setattr(qa, "load_active", lambda _: snapshot)
    monkeypatch.setattr(qa, "ParkAssistant", FakeAssistant)
    app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=15)
    app.chat_input[0].set_value("What are pet rules?").run(timeout=15)
    assert not app.exception
    assert any("Pets must be leashed. [S1]" in t.value for t in app.text)
    assert any("2026-10-23" in c.value for c in app.caption)
    assert len(app.get("link_button")) == 2
