"""Source processing utilities."""

from urllib.parse import (
    urlsplit,
    urlunsplit,
)


def normalize_url(url: str) -> str:

    url = (url or "").strip()

    if not url:
        return ""

    parts = urlsplit(url)

    scheme = (
        parts.scheme.lower()
        or "https"
    )

    netloc = parts.netloc.lower()

    path = parts.path.rstrip("/")

    return urlunsplit(
        (
            scheme,
            netloc,
            path,
            parts.query,
            "",
        )
    )


def source_key(item):

    return (
        normalize_url(
            item.get("url", "")
        )
        or
        str(
            item.get("title", "")
        )
        .strip()
        .lower()
    )


def rank_source(item):

    title = str(
        item.get("title", "")
    )

    snippet = str(
        item.get("snippet", "")
    )

    url = normalize_url(
        item.get("url", "")
    )

    score = 0

    if url.startswith("https://"):
        score += 2

    if title:
        score += 1

    if snippet:
        score += 1

    return score