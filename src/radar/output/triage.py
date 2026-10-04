"""Triage coverage summaries (pure: no services, no I/O).

Distinguishes screening counts (considered/scored/unknown/failure with
model/rubric provenance) from synthesis counts (selected/analyzed/
opportunities). Unknown covers missing abstracts, oversize texts, and
deadlines; failure covers service errors. No printing: callers print.
"""

from __future__ import annotations

from typing import Any
from collections import Counter

UNKNOWN_STATUSES = frozenset({"missing_abstract", "oversized", "deadline"})


def summarize_batch(batch: Any) -> dict[str, Any]:
    """Count a screening batch by status bucket with provenance."""
    results = list(getattr(batch, "results", []) or [])
    scored = sum(1 for r in results if getattr(r, "status", None) == "scored")
    unknown = sum(1 for r in results if getattr(r, "status", None) in UNKNOWN_STATUSES)
    failed = sum(1 for r in results if getattr(r, "status", None) == "failed")
    return {
        "considered": len(results),
        "scored": scored,
        "unknown": unknown,
        "failed": failed,
        "model_id": getattr(batch, "model_id", ""),
        "rubric_version": getattr(batch, "rubric_version", ""),
        "backend": getattr(batch, "backend", "clef"),
        "probability_kind": getattr(batch, "probability_kind", "native_noul"),
        "fallback_reason": getattr(batch, "fallback_reason", None),
        "failure_categories": dict(sorted(Counter(
            str(getattr(r.failure_kind, "value", r.failure_kind))
            for r in results if getattr(r, "failure_kind", None) is not None
        ).items())),
    }


def triage_coverage_line(
    *,
    pool_total: int,
    summary: dict[str, Any],
    selected: int,
    opportunities: int,
) -> str:
    """One honest coverage line: screening buckets vs synthesis counts."""
    return (
        f"radar: coverage pool={pool_total} "
        f"considered={summary['considered']} scored={summary['scored']} "
        f"unknown={summary['unknown']} failed={summary['failed']} "
        f"selected={selected} analyzed={selected} "
        f"opportunities={opportunities} "
        f"model={summary['model_id']} rubric={summary['rubric_version']} "
        f"backend={summary['backend']} probabilities={summary['probability_kind']} "
        f"fallback={summary['fallback_reason'] or 'none'} "
        "(bounded discovery sample, not all of OpenAlex)"
    )


__all__ = ["UNKNOWN_STATUSES", "summarize_batch", "triage_coverage_line"]
