"""Central configuration for SYNAPTIX."""

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class Settings:
    ollama_base_url: str = os.getenv(
        "OLLAMA_BASE_URL",
        "http://127.0.0.1:11434",
    )

    model: str = os.getenv(
        "SYNAPTIX_MODEL",
        "qwen3.6:35b",
    )

    ollama_timeout: int = int(
        os.getenv("OLLAMA_TIMEOUT", "300")
    )

    max_search_results: int = int(
        os.getenv("MAX_SEARCH_RESULTS", "6")
    )

    max_research_sources: int = int(
        os.getenv("MAX_RESEARCH_SOURCES", "5")
    )

    search_workers: int = max(
        1,
        int(os.getenv("SEARCH_WORKERS", "6")),
    )

    context_window: int = int(
        os.getenv("SYNAPTIX_CONTEXT", "32768")
    )


SETTINGS = Settings()