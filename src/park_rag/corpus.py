"""Allowlisted NPS collection, cleaning, deterministic document/chunk IDs."""
import hashlib
import json
import re
import time
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import unquote, urlsplit
from urllib.robotparser import RobotFileParser

import httpx
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

PARKS = {"yell": "Yellowstone", "shen": "Shenandoah", "grsm": "Great Smoky Mountains"}
USER_AGENT = "NationalParksLearningRAG/0.2"


def canonical_url(url: str) -> str:
    parts = urlsplit(url)
    if (parts.scheme != "https" or parts.netloc.lower() != "www.nps.gov"
            or parts.path.split("/")[1] not in PARKS or not parts.path.endswith(".htm")
            or parts.query or parts.fragment
            or any(p in {".", ".."} for p in unquote(parts.path).split("/"))):
        raise ValueError("Sources must be explicit NPS .htm pages for yell, shen or grsm.")
    return "https://www.nps.gov" + parts.path


def read_sources(path: Path) -> list[str]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    urls = [canonical_url(row["url"]) for row in manifest["sources"]]
    if not urls or len(urls) != len(set(urls)):
        raise ValueError("Source manifest must contain a nonempty list of unique URLs.")
    return urls


def extract_page(html: str, url: str, collected_at: str) -> Document:
    soup = BeautifulSoup(html, "html.parser")
    main = soup.select_one("#main") or soup.find("main") or soup.select_one("[role=main]")
    if main is None:
        raise ValueError(f"No main content region: {url}")
    heading = main.find("h1") or soup.find("h1")
    title = heading.get_text(" ", strip=True) if heading else ""
    if not title:
        raise ValueError(f"No page title: {url}")
    for unwanted in main.select(
        "script, style, nav, footer, header, form, iframe, button, noscript, "
        ".modal, .breadcrumb, .PageActions, .ContentHeader__pageActions, "
        "#ParkFooter, .ParkFooter, [role=contentinfo], .mejs-container, .visually-hidden"
    ):
        unwanted.decompose()
    lines = [re.sub(r"\s+", " ", line).strip() for line in main.get_text("\n").splitlines()]
    text = "\n".join(line for line in lines if line)
    if len(text) < 150:
        raise ValueError(f"Suspiciously short extracted page: {url}")
    return Document(page_content=text, metadata={
        "document_id": hashlib.sha256(url.encode()).hexdigest()[:24],
        "title": title, "source_url": canonical_url(url), "park_code": urlsplit(url).path.split("/")[1],
        "park_name": PARKS[urlsplit(url).path.split("/")[1]],
        "collected_at": collected_at,
    })


def collect(urls: list[str]) -> list[Document]:
    # Do not crawl links. Check robots policy once, then request only configured URLs.
    with httpx.Client(timeout=30, follow_redirects=False,
                      headers={"User-Agent": USER_AGENT}) as client:
        policy = client.get("https://www.nps.gov/robots.txt")
        policy.raise_for_status()
        robots = RobotFileParser()
        robots.parse(policy.text.splitlines())
        pages = []
        for url in urls:
            canonical_url(url)
            if not robots.can_fetch(USER_AGENT, url):
                raise ValueError(f"robots.txt disallows collection: {url}")
            time.sleep(1)
            response = client.get(url)
            response.raise_for_status()
            if "text/html" not in response.headers.get("content-type", ""):
                raise ValueError(f"Expected HTML: {url}")
            pages.append(extract_page(response.text, url, datetime.now(UTC).isoformat()))
        return pages


def split_pages(pages: list[Document], size: int, overlap: int) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap,
                                             separators=["\n\n", "\n", ". ", " ", ""])
    chunks = []
    for page in pages:
        for ordinal, text in enumerate(splitter.split_text(page.page_content)):
            metadata = dict(page.metadata)
            metadata["chunk_id"] = f"{metadata['document_id']}-{ordinal:04d}"
            chunks.append(Document(page_content=text, metadata=metadata))
    return chunks


def corpus_version(chunks: list[Document], embedding_model: str, dimension: int) -> str:
    # Timestamps excluded: unchanged pages and settings yield identical namespace and IDs.
    payload = {"schema": 1, "embedding_model": embedding_model, "dimension": dimension,
               "chunks": sorted((c.metadata["chunk_id"], c.metadata["title"], c.page_content)
                                for c in chunks)}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:24]


def save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    temp.replace(path)
