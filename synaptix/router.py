"""SYNAPTIX intelligent query router."""

import re

from dataclasses import dataclass


@dataclass(frozen=True)
class Route:

    kind: str

    needs_web: bool

    deep_research: bool

    complexity: str


CURRENT = re.compile(
    r"\b("
    r"latest|"
    r"today|"
    r"tonight|"
    r"current|"
    r"currently|"
    r"right now|"
    r"this week|"
    r"this month|"
    r"this year|"
    r"as of now|"
    r"as of today|"
    r"recent|"
    r"recently|"
    r"2026|"
    r"newest|"
    r"updated"
    r")\b",
    re.I,
)


DEEP = re.compile(
    r"\b("
    r"research|"
    r"deep dive|"
    r"investigate|"
    r"comprehensive|"
    r"in[- ]depth|"
    r"evaluate|"
    r"analysis|"
    r"survey|"
    r"literature"
    r")\b",
    re.I,
)


COMPLEX = re.compile(
    r"\b("
    r"architect|"
    r"architecture|"
    r"debug|"
    r"debugging|"
    r"design|"
    r"implement|"
    r"optimize|"
    r"algorithm|"
    r"system|"
    r"trade[- ]off|"
    r"step[- ]by[- ]step|"
    r"prove|"
    r"derive"
    r")\b",
    re.I,
)


def classify(query: str) -> Route:

    query = (query or "").strip()

    if not query:
        return Route(
            "empty",
            False,
            False,
            "low",
        )

    needs_web = bool(
        CURRENT.search(query)
        or DEEP.search(query)
    )

    deep = bool(
        DEEP.search(query)
    )

    if deep:

        return Route(
            "research",
            True,
            True,
            "high",
        )

    if needs_web:

        return Route(
            "current",
            True,
            False,
            "medium",
        )

    if COMPLEX.search(query):

        return Route(
            "complex",
            False,
            False,
            "high",
        )

    return Route(
        "normal",
        False,
        False,
        "low",
    )