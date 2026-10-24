"""Report ingestion failures without printing credentials or provider response bodies."""
from contextlib import contextmanager


class IndexConfigurationError(ValueError):
    """Application-generated index mismatch; safe to display without provider payloads."""


@contextmanager
def step(name):
    print(f"Ingestion: {name}...", flush=True)
    try:
        yield
    except Exception as exc:
        if not getattr(exc, "ingestion_stage", None):
            exc.ingestion_stage = name
        raise


def describe_failure(exc):
    stage = getattr(exc, "ingestion_stage", "initialization")
    details = [type(exc).__name__]
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    if not isinstance(status, int):
        status = getattr(getattr(exc, "response", None), "status_code", None)
    if isinstance(status, int):
        details.append(f"HTTP {status}")
    code = getattr(exc, "code", None)
    known_codes = {"insufficient_quota", "rate_limit_exceeded", "invalid_api_key", "model_not_found"}
    if isinstance(code, str) and code in known_codes:
        details.append(code)

    if isinstance(exc, IndexConfigurationError):
        hint = str(exc)
    elif code == "insufficient_quota":
        hint = "Check the model provider's API billing and available quota for this project."
    elif status == 401:
        hint = "Check the API key for the service at this step in your local .env."
    elif status == 403:
        hint = "Check service permissions and access restrictions."
    elif status == 429:
        hint = "Check service quota and rate limits before retrying."
    elif status == 404:
        hint = "Check the configured model, index, or source URL for this step."
    elif stage in {"loading settings", "checking credentials"}:
        hint = "Check .env variable names, required keys, and numeric settings against .env.example."
    elif stage == "collecting NPS pages":
        hint = "Check NPS connectivity, source URLs, robots.txt access, and HTML extraction."
    elif stage == "connecting to Pinecone":
        hint = "Check index name, cloud/region, permissions, cosine metric, and embedding dimension."
    elif stage == "generating embeddings":
        hint = "Check embedding model access, dimension support, provider connectivity, and quota."
    elif stage == "verifying Pinecone records":
        hint = "Pinecone records may not be visible yet. Wait briefly and rerun ingestion."
    else:
        hint = "Check the settings and service connectivity for this step."
    return (f"Ingestion failed during {stage} ({', '.join(details)}). {hint}\n"
            "The active corpus pointer was not switched. Provider error bodies and secrets are hidden.")
