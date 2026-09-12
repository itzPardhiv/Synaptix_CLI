"""Conversation memory management."""

import json

from pathlib import Path

from typing import (
    List,
    Dict,
)


def load(
    path: str = "synaptix_memory.json"
) -> List[Dict]:

    file = Path(path)

    if not file.exists():
        return []

    try:

        data = json.loads(
            file.read_text(
                encoding="utf-8"
            )
        )

        if isinstance(data, list):
            return data

        return []

    except (
        OSError,
        ValueError,
    ):
        return []


def save(
    messages: List[Dict],
    path: str = "synaptix_memory.json",
    limit: int = 100,
) -> None:

    file = Path(path)

    file.write_text(
        json.dumps(
            messages[-limit:],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def trim(
    messages: List[Dict],
    max_messages: int = 24,
) -> List[Dict]:

    return messages[
        -max(2, int(max_messages)):
    ]