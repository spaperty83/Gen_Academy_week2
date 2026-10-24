"""Small Pinecone adapter; no model or UI orchestration here."""
import time

from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec

from .diagnostics import IndexConfigurationError, step


def make_embeddings(settings):
    return OpenAIEmbeddings(api_key=settings.openai_api_key, model=settings.embedding_model,
                            dimensions=settings.embedding_dimension, max_retries=1,
                            request_timeout=30)


def connect_index(settings, *, create=False):
    pc = Pinecone(api_key=settings.pinecone_api_key, timeout=30)
    if not pc.has_index(settings.index_name):
        if not create:
            raise ValueError("Pinecone index is missing. Run ingestion first.")
        pc.create_index(name=settings.index_name, dimension=settings.embedding_dimension,
                        metric="cosine", spec=ServerlessSpec(cloud=settings.cloud,
                                                             region=settings.region))
    deadline = time.monotonic() + 60
    while True:
        info = pc.describe_index(settings.index_name)
        if info.dimension != settings.embedding_dimension:
            raise IndexConfigurationError(
                f"Pinecone index has {info.dimension} dimensions; "
                f"EMBEDDING_DIMENSION is {settings.embedding_dimension}. "
                "Use a matching index, or set an unused PINECONE_INDEX name and let ingestion create it."
            )
        if info.metric != "cosine":
            raise IndexConfigurationError("Pinecone index must use cosine similarity. Use a cosine index.")
        if info.status["ready"]:
            return pc.Index(host=info.host)
        if time.monotonic() > deadline:
            raise TimeoutError("Index is not ready yet. Rerun ingestion later.")
        time.sleep(2)


def upsert_chunks(index, embeddings, chunks, namespace):
    for start in range(0, len(chunks), 32):
        batch = chunks[start:start + 32]
        with step("generating embeddings"):
            vectors = embeddings.embed_documents([
                f"{c.metadata['title']}\n{c.page_content}" for c in batch
            ])
        if len(vectors) != len(batch):
            raise ValueError("Embedding count does not match chunks.")
        with step("uploading passages to Pinecone"):
            index.upsert(namespace=namespace, vectors=[
                {"id": c.metadata["chunk_id"], "values": vector,
                 "metadata": {**c.metadata, "text": c.page_content}}
                for c, vector in zip(batch, vectors, strict=True)
            ])


def wait_for_records(index, chunks, namespace):
    pending = {c.metadata["chunk_id"] for c in chunks}
    deadline = time.monotonic() + 30
    while pending:
        for start in range(0, len(pending), 100):
            ids = sorted(pending)[start:start + 100]
            result = index.fetch(ids=ids, namespace=namespace)
            # Fetch visibility is checked before publishing a new local active pointer.
            if result.vectors:
                pending = pending - set(result.vectors)
                break
        if not pending:
            return
        if time.monotonic() >= deadline:
            raise TimeoutError("Records are not visible yet; active corpus was not switched. Rerun ingestion.")
        time.sleep(2)
