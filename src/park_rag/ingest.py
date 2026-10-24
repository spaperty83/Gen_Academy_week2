"""Offline command: collect, preview, or publish a complete corpus."""
import argparse
import sys
from pathlib import Path

from .config import ROOT, Settings
from .corpus import collect, corpus_version, read_sources, save_json, split_pages
from .diagnostics import describe_failure, step
from .store import connect_index, make_embeddings, upsert_chunks, wait_for_records


def run(settings, sources, *, preview=False):
    if not preview:
        with step("checking credentials"):
            settings.require_credentials()
    with step("reading source manifest"):
        urls = read_sources(sources)
    with step("collecting NPS pages"):
        pages = collect(urls)
    with step("splitting documents"):
        chunks = split_pages(pages, settings.chunk_size, settings.chunk_overlap)
    version = corpus_version(chunks, settings.embedding_model, settings.embedding_dimension)
    snapshot = {
        "park_code": "yell", "namespace": f"yell-{version}", "corpus_version": version,
        "index_name": settings.index_name, "embedding_model": settings.embedding_model,
        "embedding_dimension": settings.embedding_dimension,
        "page_count": len(pages), "chunk_count": len(chunks),
        "sources": [p.metadata for p in pages],
    }
    save_json(ROOT / "data" / "preview.json", {
        **snapshot, "chunks": [{**c.metadata, "text": c.page_content} for c in chunks]
    })
    if preview:
        print(f"Preview: {len(pages)} pages, {len(chunks)} chunks. Inspect data/preview.json.")
        return snapshot
    with step("connecting to Pinecone"):
        index = connect_index(settings, create=True)
    with step("configuring embedding model"):
        embeddings = make_embeddings(settings)
    with step("embedding and storing passages"):
        upsert_chunks(index, embeddings, chunks, snapshot["namespace"])
    with step("verifying Pinecone records"):
        wait_for_records(index, chunks, snapshot["namespace"])
    with step("publishing active corpus"):
        save_json(ROOT / "data" / "active.json", snapshot)
    print(f"Published {len(chunks)} chunks from {len(pages)} pages. Namespace: {snapshot['namespace']}")
    return snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path,
                        default=ROOT / "config" / "sources.json")
    parser.add_argument("--preview", action="store_true", help="Collect and split without API keys or writes to Pinecone")
    args = parser.parse_args()
    try:
        with step("loading settings"):
            settings = Settings.load()
        run(settings, args.sources, preview=args.preview)
    except Exception as exc:  # noqa: BLE001 -- sanitize provider errors at the user-facing boundary
        print(describe_failure(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
