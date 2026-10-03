"""Deterministic scoring and selection (pure: schema + config only).

No network, no LLM, no I/O. Scoring is keyword overlap with small
deterministic tie-breakers; selection is rank-then-slice so later query
branches still contribute when the output bound is small.
"""

from __future__ import annotations

import datetime as _dt

from radar.config.runtime import MAX_CANDIDATES_IN_PROMPT, MAX_KEYWORDS
from radar.schema.papers import CollectedWork


def cheap_score(work: CollectedWork, keywords: list[str]) -> float:
    """Deterministic keyword-overlap score in [0, ~1.3]."""
    hay = f"{work.title}\n{work.abstract}".lower()
    kws = [k.lower().strip() for k in keywords if isinstance(k, str) and k.strip()]
    if not kws or not hay.strip():
        base = 0.0
    else:
        hits = sum(1 for k in kws[:MAX_KEYWORDS] if k in hay)
        base = hits / max(1, len(kws[:MAX_KEYWORDS]))
    # Tiny deterministic tie-breakers (no randomness, no network).
    cite_bonus = min(0.2, (work.cited_by_count or 0) / 5000.0)
    recency_bonus = 0.0
    if work.publication_year:
        try:
            age = _dt.date.today().year - int(work.publication_year)
            recency_bonus = max(0.0, min(0.1, 0.1 - 0.01 * max(0, age)))
        except (ValueError, TypeError):
            recency_bonus = 0.0
    return round(base + cite_bonus + recency_bonus, 6)


def rank_works(
    works: list[CollectedWork], keywords: list[str] | None = None
) -> list[CollectedWork]:
    """Score in place and return works ordered by score desc, ID asc."""
    scoring_kws = keywords or []
    for work in works:
        work.score = cheap_score(work, scoring_kws)
    return sorted(works, key=lambda w: (-w.score, w.openalex_id))


def select_topn(works: list[CollectedWork], n: int) -> list[CollectedWork]:
    """Rank cached works reproducibly (score desc, OpenAlex ID asc); slice topN."""
    ranked = sorted(works, key=lambda w: (-w.score, w.openalex_id))
    return ranked[: max(0, int(n))]


def effective_candidate_limit(max_candidates: int) -> int:
    """Clamp *max_candidates* to the prompt bound (1..MAX_CANDIDATES_IN_PROMPT)."""
    try:
        n = int(max_candidates)
    except (TypeError, ValueError):
        return MAX_CANDIDATES_IN_PROMPT
    return max(1, min(n, MAX_CANDIDATES_IN_PROMPT))


def bound_candidates(
    candidates: list[CollectedWork], max_candidates: int = MAX_CANDIDATES_IN_PROMPT
) -> list[CollectedWork]:
    """Return the deterministic bounded slice supplied to the LLM."""
    return list(candidates[: effective_candidate_limit(max_candidates)])


__all__ = [
    "bound_candidates",
    "cheap_score",
    "effective_candidate_limit",
    "rank_works",
    "select_topn",
]
