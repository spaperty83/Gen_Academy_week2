# National parks RAG assistant

Week 2 supports **Yellowstone**, **Shenandoah**, and **Great Smoky Mountains** using an explicit allowlist of nine official NPS pages covering safety, pets, and hiking.

Retrieval combines **BM25** lexical search and **Pinecone dense cosine search**. A LangGraph graph runs the two retrievers in parallel, joins their rankings using reciprocal rank fusion (RRF, constant 60), and generates a cited answer. Pinecone stores and searches dense embeddings; BM25 uses a local text snapshot. LangGraph coordinates retrieval; OpenAI supplies embeddings and answer generation.

## Setup and run

From the Week2 directory in PowerShell:

```powershell
uv sync --locked
# Only for new setups; preserve your existing .env:
# Copy-Item .env.example .env
uv run python -m park_rag.ingest --preview
uv run python -m park_rag.ingest
uv run streamlit run app.py
```

Set `OPENAI_API_KEY` and `PINECONE_API_KEY` locally in `.env`. Configure `PINECONE_INDEX`, `PINECONE_CLOUD` and `PINECONE_REGION` as shown in `.env.example`; existing environment variables override `.env`. The preview downloads and splits public pages without model calls. Full ingestion calls the embedding API; questions call the embedding and chat APIs and may incur provider charges.

**Re-ingest once after this migration.** Old local-only snapshots cannot be used with the Pinecone backend. Ingestion upserts vectors and metadata into a versioned Pinecone namespace, verifies record visibility, and writes passage text and metadata to `data/indexes/<version>.json`, then atomically publishes `data/active.json`. Failed embedding, Pinecone upload/verification, or local writes preserve the previous active pointer. Keep the active pointer and its matching index file with the app. Do not run concurrent ingestion processes. Historical local index files are retained; no remote Pinecone indexes are deleted.

## Retrieval and evidence

- Only the configured NPS URLs are collected. Links are not crawled. Collection checks robots.txt, spaces requests and rejects redirects and short/missing content.
- Each passage retains its park code/name, source URL, title, collection time and stable chunk ID. Chunks default to 1800 characters with 250 characters overlap.
- Both retrievers search the same versioned corpus, taking up to `TOP_K * 4` candidates each. BM25 uses lowercase word tokens. Dense search sends normalized OpenAI query embeddings to Pinecone, using the active namespace and park metadata filters. RRF deduplicates and returns at most `TOP_K` passages.
- Explicit park names in a question restrict both retrievers to those parks; questions with multiple named parks search those parks together. Questions without a recognized name search all parks. Name matching is deterministic, not model-based; use full park names to avoid ambiguity.
- The prompt keeps park rules separate. Answers use structured claims with validated evidence IDs. Missing evidence and invalid citations fall back to an unsupported answer. Citation validation does not prove claim entailment.
- These are dated snapshots. Stored evidence cannot verify live conditions, closures or today's availability. Each question is independent.

Pinecone is the dense vector database. The local JSON file contains passage text and metadata for BM25 and citation rendering, without dense vectors, and is loaded for each submitted question. BM25 and dense scores are rankings, not confidence probabilities. Re-ingest whenever the embedding model/dimension, source manifest, or chunking changes.

## Example questions

- Can I bring my dog hiking in Shenandoah?
- What hiking safety guidance does Great Smoky Mountains provide?
- Compare pet rules in Yellowstone and Great Smoky Mountains.
- Is Skyline Drive open right now? (requires live information; unsupported by snapshots)

## Project structure

`config/sources.json` owns the allowlist; `corpus.py` collects and splits pages; `store.py` builds the Pinecone dense index and supplies BM25, Pinecone search and fusion; `ingest.py` publishes snapshots; `qa.py` builds the LangGraph workflow and validates cited answers; `app.py` renders the Streamlit interface.

## Validation

```powershell
uv run pytest -q
uv run ruff check .
```

Offline tests use mocked providers to check park metadata/filtering, dense ranking, BM25 matching, fusion deduplication/allowlisting, failed publication, citation handling, graph empty-evidence behavior and Streamlit UI. They do not measure real model faithfulness or retrieval quality. Full embedding and generated-answer validation requires a credentialed run.

References: [LangGraph graph API](https://reference.langchain.com/python/langgraph/graph), [BM25 implementation](https://github.com/dorianbrown/rank_bm25), [OpenAI embeddings integration](https://docs.langchain.com/oss/python/integrations/embeddings/openai).

Pinecone ingestion creates a missing serverless index and requires cosine metric and the configured embedding dimension. Changed corpus content uses a new namespace; historical namespaces remain for rollback. This does not delete previous remote data.
