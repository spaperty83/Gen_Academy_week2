# Implementation verification

Recorded October 23, 2026.

## Passed checks

- `uv sync --locked`: successful; dependencies and the local package installed from the lockfile.
- `uv run ruff check .`: all checks passed.
- `uv run pytest -q`: 21 tests passed, including Streamlit startup without keys and supported-answer rendering with a fake service.
- `uv run python -m park_rag.ingest --preview`: fetched all three configured public NPS pages and produced 18 chunks without model or Pinecone calls.
- Inspected preview passages, then corrected extraction to remove NPS's `ParkFooter` container and hidden media helper text.
- Verified final preview titles (Safety, Pets, Hiking), source URLs, collection timestamps and absence of footer text.
- Checked installed LangChain model constructors and JSON-schema binding without network calls.
- Checked installed Pinecone create/index/query/fetch signatures and readiness-response compatibility against the implementation.

## API sources

Official LangChain docs for OpenAI embeddings, ChatOpenAI structured output, and recursive character splitting; official Pinecone docs for index creation, upserts, metadata filters and fetch; official Streamlit chat documentation. Links appear in the README.

Installed core versions: Python 3.11.1, uv 0.13.0, Streamlit 1.66.1, langchain-core 1.7.0, langchain-openai 1.7.2, langchain-text-splitters 1.1.3, Pinecone 10.1.0. Exact transitive versions are in uv.lock.

## Not yet verified

No credentialed model or Pinecone operations were performed. Embedding generation, index creation, actual upsert/query behavior and generated-answer quality need a local credentialed run using the README steps. Offline tests use mock services; they are not evidence of production connectivity or faithfulness. No measured faithfulness, retrieval quality or latency results are claimed.

The preview is a real collection; it is not a published Pinecone corpus. `data/active.json` will be created only by successful non-preview ingestion. Generated data is ignored by Git.
