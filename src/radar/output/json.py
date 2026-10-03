"""Machine-readable JSON envelopes for collect-only output (pure: schema only).

No Markdown here, no printing: callers print the returned strings.
"""

from __future__ import annotations

import json as _json
from typing import Any

from radar.schema.papers import CollectedWork


def work_to_json(work: CollectedWork) -> dict[str, Any]:
    """Bounded JSON view of one collected work (abstract capped at 2000)."""
    return {
        "openalex_id": work.openalex_id,
        "title": work.title,
        "abstract": work.abstract[:2000],
        "publication_year": work.publication_year,
        "doi": work.doi,
        "primary_url": work.primary_url,
        "cited_by_count": work.cited_by_count,
        "score": work.score,
        "matched_queries": work.matched_queries,
        "query_kinds": work.query_kinds,
    }


def collect_envelope(works: list[CollectedWork]) -> str:
    """JSON array of bounded candidate views (no LLM involved)."""
    return _json.dumps([work_to_json(w) for w in works], indent=2)


def refresh_envelope(summary: dict[str, Any]) -> str:
    """JSON summary of a metadata-only refresh (coverage + delta)."""
    return _json.dumps(summary, indent=2)


def snapshot_inspection_envelope(
    *,
    snapshot: str,
    collected_at_utc: str,
    coverage: dict[str, Any],
    discovery: dict[str, Any],
    disclosure: str,
    selected: int,
    works: list[CollectedWork],
) -> str:
    """JSON inspection of a cached snapshot: full-pool counts + topN slice."""
    return _json.dumps(
        {
            "snapshot": snapshot,
            "collected_at_utc": collected_at_utc,
            "coverage": coverage,
            "discovery": discovery,
            "disclosure": disclosure,
            "selected": selected,
            "works": [work_to_json(w) for w in works],
        },
        indent=2,
    )


__all__ = [
    "collect_envelope",
    "refresh_envelope",
    "snapshot_inspection_envelope",
    "work_to_json",
]
