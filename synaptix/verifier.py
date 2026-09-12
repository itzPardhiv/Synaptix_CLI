"""Evidence verification primitives."""

import re

from dataclasses import dataclass

from typing import (
    Iterable,
    Dict,
)


@dataclass(frozen=True)
class Verification:

    supported: bool

    confidence: float

    reason: str


def verify_claim(
    claim: str,
    sources: Iterable[Dict],
) -> Verification:

    claim = (
        claim or ""
    ).strip()

    evidence_parts = []

    for source in sources or []:

        evidence_parts.append(
            " ".join(
                [
                    str(
                        source.get(
                            "title",
                            "",
                        )
                    ),

                    str(
                        source.get(
                            "snippet",
                            "",
                        )
                    ),

                    str(
                        source.get(
                            "content",
                            "",
                        )
                    ),
                ]
            )
        )

    evidence = (
        " ".join(evidence_parts)
        .lower()
    )

    if not claim:

        return Verification(
            False,
            0.0,
            "empty claim",
        )

    tokens = [
        token
        for token in re.findall(
            r"[a-z0-9]{4,}",
            claim.lower(),
        )
    ]

    if not tokens:

        return Verification(
            False,
            0.0,
            "no verifiable terms",
        )

    unique_tokens = set(tokens)

    hits = sum(
        1
        for token in unique_tokens
        if token in evidence
    )

    ratio = (
        hits
        /
        max(
            1,
            len(unique_tokens),
        )
    )

    if ratio >= 0.25:

        return Verification(
            True,
            round(ratio, 2),
            "evidence overlap detected",
        )

    return Verification(
        False,
        round(ratio, 2),
        "insufficient evidence overlap",
    )