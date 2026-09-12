"""Fault-tolerant research orchestration."""

from concurrent.futures import (
    ThreadPoolExecutor,
    as_completed,
)

from typing import (
    Callable,
    Iterable,
    List,
    Dict,
    Any,
    Optional,
)

from .config import SETTINGS


def _safe_workers(
    count: int,
    cap: Optional[int] = None,
) -> int:

    value = max(
        1,
        int(count),
    )

    if cap is not None:

        value = min(
            value,
            max(1, int(cap)),
        )

    return value


def parallel_map(
    fn: Callable,
    items: Iterable,
    workers: int = None,
) -> List[Any]:

    items = list(items)

    # VERY IMPORTANT:
    # Don't create ThreadPoolExecutor at all
    # when there is nothing to process.

    if not items:
        return []

    workers = _safe_workers(
        workers or SETTINGS.search_workers,
        len(items),
    )

    output = [None] * len(items)

    with ThreadPoolExecutor(
        max_workers=workers
    ) as pool:

        futures = {
            pool.submit(fn, item): index
            for index, item in enumerate(items)
        }

        for future in as_completed(futures):

            index = futures[future]

            try:
                output[index] = future.result()

            except Exception:
                # One broken source must not
                # destroy the entire research job.
                output[index] = None

    return [
        item
        for item in output
        if item is not None
    ]


def deduplicate(
    results: Iterable[Dict[str, Any]]
) -> List[Dict[str, Any]]:

    seen = set()

    output = []

    for item in results or []:

        url = (
            str(item.get("url", ""))
            .strip()
            .lower()
            .rstrip("/")
        )

        key = (
            url
            or
            str(
                item.get("title", "")
            )
            .strip()
            .lower()
        )

        if not key:
            continue

        if key in seen:
            continue

        seen.add(key)

        output.append(item)

    return output