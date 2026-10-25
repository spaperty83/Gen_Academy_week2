"""Configuration shared by offline ingestion and online Q&A."""
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    openai_api_key: str = field(default="", repr=False)
    pinecone_api_key: str = field(default="", repr=False)
    index_name: str = "national-parks-rag"
    cloud: str = "aws"
    region: str = "us-east-1"
    chat_model: str = "gpt-4.1-mini"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimension: int = 1536
    top_k: int = 5
    chunk_size: int = 1800
    chunk_overlap: int = 250

    def __post_init__(self):
        if self.embedding_dimension < 1 or not 1 <= self.top_k <= 20:
            raise ValueError("Embedding dimension must be positive and TOP_K between 1 and 20.")
        if not 200 <= self.chunk_size <= 6000 or not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError("CHUNK_SIZE must be 200..6000 and overlap smaller than the chunk.")

    @classmethod
    def load(cls):
        load_dotenv(ROOT / ".env", override=False)
        return cls(
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            pinecone_api_key=os.getenv("PINECONE_API_KEY", ""),
            index_name=os.getenv("PINECONE_INDEX", "national-parks-rag"),
            cloud=os.getenv("PINECONE_CLOUD", "aws"),
            region=os.getenv("PINECONE_REGION", "us-east-1"),
            chat_model=os.getenv("CHAT_MODEL", "gpt-4.1-mini"),
            embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
            embedding_dimension=int(os.getenv("EMBEDDING_DIMENSION", "1536")),
            top_k=int(os.getenv("TOP_K", "5")),
            chunk_size=int(os.getenv("CHUNK_SIZE", "1800")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "250")),
        )

    def require_credentials(self):
        missing = [name for name, value in (
            ("OPENAI_API_KEY", self.openai_api_key),
            ("PINECONE_API_KEY", self.pinecone_api_key),
        ) if not value or value.startswith("your-")]
        if missing:
            raise ValueError("Set " + ", ".join(missing) + " locally in .env or your environment.")
