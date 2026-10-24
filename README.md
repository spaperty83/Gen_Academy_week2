# Yellowstone National Park RAG assistant

A small Python and Streamlit application answering questions from three explicitly configured public National Park Service pages. LangChain supplies document splitting, embeddings, prompts, and structured generation. Pinecone stores dense vectors. Ingestion runs separately from the question-answering app.

Stored content is a dated snapshot, not a live park-conditions service. Answers include supporting NPS links and passage excerpts. Missing evidence produces an explicit unsupported answer. No LangGraph or agents are needed for this version.

## Project structure

```text
app.py                       Streamlit interface
config/sources.json          Explicit Yellowstone URL allowlist
src/park_rag/
  config.py                  Environment settings and validation
  corpus.py                  Collection, cleaning, splitting, stable IDs
  ingest.py                  Offline preview and publication command
  store.py                   Embedding and Pinecone adapters
  qa.py                      Retrieval, evidence prompt, citation validation
tests/                      Offline behavior and UI tests
 data/                       Generated local preview/active snapshot; ignored by Git
.env.example                 Credential placeholders and model/index settings
pyproject.toml               Dependencies and tool configuration
uv.lock                      Exact resolved versions
```

## Stage 1: install and configure

Use Python 3.11 or newer and [uv](https://docs.astral.sh/uv/getting-started/installation/). In PowerShell:

```powershell
cd C:\Sunita\GIT\GenAcademyProject\Week2
uv sync --locked
Copy-Item .env.example .env
```

Edit `.env` locally. Supply `OPENAI_API_KEY` and `PINECONE_API_KEY`; never send keys in chat or commit them. Existing environment variables take precedence over `.env`. An NPS API key is unnecessary for public HTML collection.

The default model provider is OpenAI through `langchain-openai`: `text-embedding-3-small` with 1536 dimensions, and `gpt-4.1-mini` for structured answers. Change `CHAT_MODEL` to a model supporting native JSON-schema structured output. Embedding configuration uses the OpenAI `dimensions` argument and should use a compatible `text-embedding-3` model. Re-ingest after an embedding change, even if dimensions are unchanged. Other providers require adapting `make_embeddings` and the chat constructor; this version does not claim arbitrary endpoint compatibility.

Ingestion creates the configured Pinecone serverless index if it does not exist. Defaults are AWS/us-east-1; adjust cloud/region to your account. An existing index must be dense with the configured dimension and cosine similarity. Use a dedicated project index and an API key that can create/write it. Provider calls and storage may incur account charges.

Verify the environment without any API calls:

```powershell
uv run python -c "from park_rag.config import Settings; print('Imports OK')"
uv run pytest -q
uv run ruff check .
```

## Stage 2: inspect collection before indexing

The manifest contains safety, pets, and hiking pages at `https://www.nps.gov/yell/`. Only the listed URLs are downloaded; links are not crawled. Redirects are rejected for review. The collector checks robots.txt, spaces requests, and fails on HTTP errors, missing main content, or suspiciously short extractions.

```powershell
uv run python -m park_rag.ingest --preview
```

No model/Pinecone keys are needed for preview. Inspect `data/preview.json`: check that there are three pages, nonempty passages, correct page titles/URLs, UTC collection dates, and no park footer. Source text may describe conditions at publication time; that does not make the snapshot current.

Chunks initially use **1800 characters** and **250 characters** of overlap, split preferentially at text boundaries. These settings are not token counts. Titles and source metadata accompany every chunk, and titles are included in embedding input. Chunk quality needs evaluation before tuning.

## Stage 3: embed and publish

After configuring keys locally:

```powershell
uv run python -m park_rag.ingest
```

The command downloads the allowlist, cleans and splits the pages, embeds passages, and upserts Pinecone records in batches. The document ID is a hash of the canonical URL; the chunk ID is the document ID plus its ordinal. The corpus namespace hashes the complete chunk content and embedding configuration. Unchanged re-ingestion overwrites the same IDs in the same namespace instead of creating duplicate records.

If content or chunk boundaries change, a new namespace isolates the new snapshot, so obsolete chunks cannot appear in current retrieval. Previous namespaces remain in Pinecone for rollback; this minimal version does not automatically delete them. They are retained history, not active duplicate search results, and may use storage. A complete run is checked with Pinecone fetch before atomically replacing `data/active.json`. An interrupted run leaves the previous active pointer in place. Pinecone is eventually consistent; if visibility verification times out, wait and rerun ingestion.

Keep `data/active.json` with the running app: it binds the index, namespace, embedding configuration, approved sources and collection timestamps. Do not run concurrent ingestion commands. Re-ingest rather than editing it manually. The app reads the pointer on each Streamlit rerun; it never collects pages during a question.

Verification with real services:

1. Confirm ingestion reports three pages and a positive chunk count.
2. Rerun ingestion. If page content has not changed, confirm the reported namespace is unchanged and the namespace's record count does not increase in Pinecone.
3. Start the app and confirm the sidebar displays the three source collection times.

## Stage 4: run question answering

```powershell
uv run streamlit run app.py
```

Open the localhost URL printed by Streamlit. Example checks:

- `Can I bring my dog on a Yellowstone hiking trail?` ? answer should cite the pets page and show supporting text.
- `Does day hiking require a permit?` ? answer should cite hiking evidence.
- `What are Yellowstone's rules for staying near thermal features?` ? answer should cite safety evidence.
- `What is the Wi-Fi password at my hotel?` ? available documents should not support an answer.
- `Is the road to Old Faithful open right now?` ? stored documents cannot verify live conditions.

Each question is independent; previous chat turns are not sent to the model. Retrieval queries only the active namespace with a Yellowstone filter, then supplies up to `TOP_K` passages. No live search or built-in model tools are enabled. The model must return supported claims with source IDs, partial coverage with a gap, or an unsupported result. Unknown or missing citation IDs cause a fallback; only cited source links are shown. Citation existence is checked in code, but claim entailment remains model-based and must be reviewed in evaluation. A similarity score is not treated as confidence.

## Stage 5: verification and remaining work

```powershell
uv run pytest -q
uv run ruff check .
```

Tests cover allowlist enforcement, cleaning, metadata preservation, repeat ingestion IDs/namespaces, snapshot isolation, publication failure, retrieval filters, empty evidence, invalid citations, partial answers, settings mismatch, and Streamlit startup without credentials. Mocked services test pipeline behavior without charging accounts. These do not prove real model faithfulness or real Pinecone connectivity.

The initial public-page preview and local package API checks can be run without credentials. End-to-end embedding, Pinecone writes/queries, generated answers, faithfulness and latency need a credentialed run. See `docs/verification.md` for the recorded implementation checks. Do not report quality targets as achieved without measurements.

For the assignment submission, retain a fixed evaluation set with expected evidence, record actual answer/retrieval failures, document prompts and iterations, prepare the required project documentation and demo, and publish project assets. This minimal version supplies the implementation foundation, not completed evaluation results or a submitted project.

## Troubleshooting

Ingestion prints each step as it runs. If it fails, the final message identifies the failed step,
exception type, and HTTP status or recognized provider code when available. For example,
`generating embeddings (RateLimitError, HTTP 429, insufficient_quota)` points to model API quota,
while `connecting to Pinecone` points to index setup or access. Raw provider responses and credentials
are not printed. A failed run does not switch the active corpus pointer. Use the reported step to
choose the relevant check below; the presence of an older preview file alone does not identify the failure.

- Missing credentials: set them locally in `.env`; restart Streamlit after changing environment values.
- Wrong index dimensions/metric: use a correctly configured new index and re-ingest.
- No active corpus/configuration mismatch: run ingestion successfully using the same settings as the app.
- Collection failed: check the configured URL, network and NPS access policy. The app retains its prior active corpus.
- Provider failure: check permissions, quota and model access in the provider dashboards. User-facing errors intentionally omit raw provider responses to avoid disclosing credentials.

## Extension points

`corpus` and `ingest` own offline data preparation; `store` owns provider access; `qa.retrieve` and `ParkAssistant.ask` own the evidence flow; `app.py` renders results. Later work can wrap these functions in LangGraph nodes, add live-search tools, persistent state, bounded error recovery, and approval steps. Keep live tool outputs separate from the dated document evidence.

## API references checked during implementation

- [LangChain OpenAI embeddings](https://docs.langchain.com/oss/python/integrations/embeddings/openai): `embed_documents`, `embed_query`, model/dimension configuration.
- [LangChain ChatOpenAI](https://docs.langchain.com/oss/python/integrations/chat/openai): `with_structured_output`, JSON-schema mode and invocation.
- [LangChain recursive text splitting](https://docs.langchain.com/oss/python/integrations/splitters/recursive_text_splitter): character-based splitting.
- [Pinecone index creation](https://docs.pinecone.io/guides/index-data/create-an-index): serverless index specification.
- [Pinecone upserts](https://docs.pinecone.io/guides/index-data/upsert-data): stable record IDs, metadata and namespaces.
- [Pinecone metadata filtering](https://docs.pinecone.io/guides/search/filter-by-metadata): vector queries and metadata filters.
- [Streamlit chat UI](https://docs.streamlit.io/develop/tutorials/chat-and-llm-apps/build-conversational-apps): chat input and message containers.

Exact installed versions are committed in `uv.lock`; `uv sync --locked` reproduces that resolution.
