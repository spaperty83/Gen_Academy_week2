"""Pinecone dense vectors and local BM25 over the same published corpus."""
import math
import re
import time

from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
from rank_bm25 import BM25Okapi

from .corpus import save_json
from .diagnostics import IndexConfigurationError, step


def make_embeddings(settings):
    return OpenAIEmbeddings(api_key=settings.openai_api_key, model=settings.embedding_model,
                            dimensions=settings.embedding_dimension, max_retries=1, request_timeout=30)


def tokens(text):
    return re.findall(r"\w+", text.lower())


def unit(vector):
    if not vector or not all(math.isfinite(v) for v in vector):
        raise ValueError("Embedding must contain finite values.")
    norm = math.sqrt(sum(v * v for v in vector))
    if not norm:
        raise ValueError("Embedding must be nonzero.")
    return [v / norm for v in vector]


def connect_index(settings, *, create=False):
    client = Pinecone(api_key=settings.pinecone_api_key, timeout=30)
    if not client.has_index(settings.index_name):
        if not create:
            raise ValueError("Pinecone index is missing. Run ingestion first.")
        client.create_index(name=settings.index_name, dimension=settings.embedding_dimension,
                            metric="cosine", spec=ServerlessSpec(cloud=settings.cloud, region=settings.region))
    deadline = time.monotonic() + 60
    while True:
        info = client.describe_index(settings.index_name)
        if info.dimension != settings.embedding_dimension or info.metric != "cosine":
            raise IndexConfigurationError(
                f"Pinecone index must have {settings.embedding_dimension} dimensions and cosine metric; "
                f"found {info.dimension} dimensions and {info.metric}. Use a matching PINECONE_INDEX."
            )
        if info.status["ready"]:
            return client.Index(host=info.host)
        if time.monotonic() >= deadline:
            raise TimeoutError("Pinecone index is not ready.")
        time.sleep(2)


def wait_for_records(index, ids, namespace):
    pending = set(ids)
    deadline = time.monotonic() + 60
    while pending:
        batch = sorted(pending)[:100]
        result = index.fetch(ids=batch, namespace=namespace)
        pending.difference_update(result.vectors or {})
        if pending:
            if time.monotonic() >= deadline:
                raise TimeoutError("Pinecone records are not visible. Active snapshot was not switched.")
            time.sleep(2)


def build_index(embeddings, chunks, dimension, path, *, index, namespace):
    records = []
    for start in range(0, len(chunks), 32):
        batch = chunks[start:start + 32]
        with step("generating embeddings"):
            vectors = embeddings.embed_documents([
                f"{c.metadata.get('park_name', '')}\n{c.metadata['title']}\n{c.page_content}" for c in batch
            ])
        if len(vectors) != len(batch):
            raise ValueError("Embedding count does not match chunks.")
        remote = []
        for chunk, vector in zip(batch, vectors, strict=True):
            if len(vector) != dimension:
                raise ValueError("Embedding dimension does not match configuration.")
            record = {**chunk.metadata, "text": chunk.page_content}
            remote.append({"id": chunk.metadata["chunk_id"], "values": unit(vector), "metadata": record})
            records.append(record)
        with step("uploading passages to Pinecone"):
            index.upsert(namespace=namespace, vectors=remote)
    if not records:
        raise ValueError("Cannot publish an empty corpus.")
    with step("verifying Pinecone records"):
        wait_for_records(index, [r["chunk_id"] for r in records], namespace)
    save_json(path, {"records": records})


class LocalIndex:
    def __init__(self, records, index, namespace):
        self.records = records
        self.index, self.namespace = index, namespace
        self.positions = {r["chunk_id"]: i for i, r in enumerate(records)}
        self.bm25 = BM25Okapi([tokens(f"{r.get('park_name', '')} {r['title']} {r['text']}") for r in records])

    def eligible(self, park_codes):
        return [i for i, r in enumerate(self.records) if not park_codes or r['park_code'] in park_codes]

    def sparse(self, question, top_k, park_codes):
        scores = self.bm25.get_scores(tokens(question))
        return sorted((i for i in self.eligible(park_codes) if scores[i] > 0),
                      key=lambda i: (-scores[i], i))[:top_k]

    def dense(self, vector, top_k, park_codes):
        query_filter = {"park_code": {"$in": park_codes}} if park_codes else None
        result = self.index.query(vector=unit(vector), namespace=self.namespace, top_k=top_k,
                                  include_metadata=False, filter=query_filter)
        eligible = set(self.eligible(park_codes))
        return [self.positions[m.id] for m in result.matches or []
                if m.id in self.positions and self.positions[m.id] in eligible]



def fuse(index, sparse, dense, sources, top_k):
    scores = {}
    for ranking in (sparse, dense):
        for rank, i in enumerate(ranking, 1):
            scores[i] = scores.get(i, 0) + 1 / (60 + rank)
    approved = {s['source_url'] for s in sources}
    evidence = []
    for i in sorted(scores, key=lambda i: (-scores[i], i)):
        r = index.records[i]
        if r['source_url'] not in approved:
            continue
        evidence.append({"source_id": f"S{len(evidence)+1}", "chunk_id": r['chunk_id'],
                         "title": r['title'], "url": r['source_url'], "park_code": r['park_code'],
                         "park_name": r.get('park_name', ''), "collected_at": r['collected_at'], "text": r['text']})
        if len(evidence) == top_k:
            break
    return evidence
