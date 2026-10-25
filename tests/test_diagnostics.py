from unittest.mock import Mock

import httpx
import pytest
from langchain_core.documents import Document

from park_rag.diagnostics import describe_failure, step
from park_rag.store import build_index


def test_http_failure_does_not_expose_secret_or_body():
    request = httpx.Request("GET", "https://example.com/?key=secret-token")
    exc = httpx.HTTPStatusError("secret-token in response", request=request,
                                response=httpx.Response(403, request=request))
    exc.ingestion_stage = "collecting NPS pages"
    message = describe_failure(exc)
    assert "collecting NPS pages" in message
    assert "HTTP 403" in message
    assert "secret-token" not in message
    assert "example.com" not in message


def test_innermost_failure_stage_is_preserved():
    with pytest.raises(TimeoutError) as captured, step("outer"), step("generating embeddings"):
        raise TimeoutError("sensitive provider payload")
    message = describe_failure(captured.value)
    assert "generating embeddings" in message
    assert "sensitive" not in message


def test_quota_code_is_actionable_without_raw_response():
    exc = RuntimeError("secret response body")
    exc.status_code = 429
    exc.code = "insufficient_quota"
    message = describe_failure(exc)
    assert "insufficient_quota" in message
    assert "billing" in message
    assert "secret response body" not in message


def test_embedding_failure_is_distinguished_from_upload(tmp_path):
    embeddings, index = Mock(), Mock()
    embeddings.embed_documents.side_effect = RuntimeError("secret provider response")
    chunks = [Document(page_content="A passage", metadata={"title": "Pets", "chunk_id": "one"})]
    with pytest.raises(RuntimeError) as captured:
        build_index(embeddings, chunks, 2, tmp_path / "index.json", index=index, namespace="test")
    assert "generating embeddings" in describe_failure(captured.value)
    index.upsert.assert_not_called()
