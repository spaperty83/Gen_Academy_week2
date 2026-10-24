import json

import pytest

from park_rag.corpus import canonical_url, corpus_version, extract_page, read_sources, split_pages

URL = "https://www.nps.gov/yell/planyourvisit/pets.htm"


def page(timestamp="2026-10-23T12:00:00+00:00"):
    return extract_page(
        '<div id="main"><h1>Pets</h1><nav>Menu secret</nav>'
        '<p>' + "Pets must be leashed. " * 30 + '</p>'
        '<div id="ParkFooter">Park footer boilerplate</div><script>bad()</script></div>',
        URL, timestamp,
    )


@pytest.mark.parametrize("url", [
    "http://www.nps.gov/yell/planyourvisit/pets.htm",
    "https://www.nps.gov.evil.example/yell/pets.htm",
    "https://www.nps.gov/shen/planyourvisit/pets.htm",
    "https://www.nps.gov/yell/pets.htm?redirect=example.com",
    "https://www.nps.gov/yell/pets.htm#fragment",
])
def test_allowlist_rejects_noncanonical_sources(url):
    with pytest.raises(ValueError):
        canonical_url(url)


def test_cleaning_and_metadata():
    result = page()
    assert "Menu secret" not in result.page_content
    assert "Park footer" not in result.page_content
    assert "bad()" not in result.page_content
    assert result.metadata["title"] == "Pets"
    assert result.metadata["source_url"] == URL
    chunks = split_pages([result], 200, 30)
    assert len(chunks) > 1
    assert all(c.metadata["collected_at"] == result.metadata["collected_at"] for c in chunks)
    assert all(c.metadata["source_url"] == URL for c in chunks)


def test_unchanged_content_has_stable_ids_and_namespace():
    first = split_pages([page()], 200, 30)
    second = split_pages([page("2026-10-24T12:00:00+00:00")], 200, 30)
    assert [c.metadata["chunk_id"] for c in first] == [c.metadata["chunk_id"] for c in second]
    assert corpus_version(first, "model-a", 1536) == corpus_version(second, "model-a", 1536)
    second[0].page_content += "changed"
    assert corpus_version(first, "model-a", 1536) != corpus_version(second, "model-a", 1536)
    assert corpus_version(first, "model-a", 1536) != corpus_version(first, "model-b", 1536)


def test_duplicate_manifest_rejected(tmp_path):
    path = tmp_path / "sources.json"
    path.write_text(json.dumps({"park_code": "yell", "sources": [{"url": URL}] * 2}))
    with pytest.raises(ValueError):
        read_sources(path)


def test_empty_extraction_rejected():
    with pytest.raises(ValueError):
        extract_page('<main><h1>Pets</h1><p>Empty</p></main>', URL, 'now')
