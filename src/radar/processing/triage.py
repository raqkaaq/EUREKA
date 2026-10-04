"""Deterministic triage selection policy (pure: schema only).

No network, no LLM, no I/O. Current ``clef-importance-v1`` batches rank
qualifying scored works by primary ``research_importance`` DESC, then
``cross_domain_potential`` DESC, then stable OpenAlex-ID, requiring
``research_importance >= 0.5``. Unknowns are never reserved or backfilled:
they stay recorded and unassessed for later investigation. Historical and
legacy batches keep the old maximum-probability rank with one reserved
unknown slot and no hard filters.
"""

from __future__ import annotations

from radar.schema.papers import CollectedWork
from radar.schema.triage import IMPORTANCE_RUBRIC_VERSION, TriageBatch, TriageResult


def _relevance(result) -> float:
    probs = [result.ai_ml_relevance, result.cross_domain_potential]
    return max(p for p in probs if p is not None)


def _select_importance(
    works: list[CollectedWork], by_id: dict[str, TriageResult], count: int
) -> list[CollectedWork]:
    """Importance selection: thresholded primary rank, no unknown backfill."""
    scored: list[tuple[float, float, str, CollectedWork]] = []
    for work in works:
        result = by_id.get(work.openalex_id)
        if result is not None and result.status == "scored":
            primary = result.research_importance
            secondary = result.cross_domain_potential
            if primary is not None and secondary is not None and primary >= 0.5:
                scored.append((primary, secondary, work.openalex_id, work))
    scored.sort(key=lambda item: (-item[0], -item[1], item[2]))
    return [work for _, _, _, work in scored[:count]]


def select_candidates(
    works: list[CollectedWork], batch: TriageBatch, n: int
) -> list[CollectedWork]:
    """Select ``n`` works: importance-ranked or legacy max-ranked."""
    try:
        count = int(n)
    except (TypeError, ValueError):
        return []
    if count <= 0 or not works:
        return []
    by_id = {result.work_id: result for result in batch.results}
    if batch.rubric_version == IMPORTANCE_RUBRIC_VERSION:
        return _select_importance(works, by_id, count)
    scored: list[tuple[float, str, CollectedWork]] = []
    unknowns: list[CollectedWork] = []
    for work in works:
        result = by_id.get(work.openalex_id)
        if result is not None and result.status == "scored":
            scored.append((_relevance(result), work.openalex_id, work))
        else:
            unknowns.append(work)
    scored.sort(key=lambda item: (-item[0], item[1]))
    if unknowns and count >= 2:
        reserve = unknowns[:1]
        rest_unknowns = unknowns[1:]
        picked = [work for _, _, work in scored[: count - 1]]
        picked.extend(reserve)
        picked.extend(rest_unknowns[: max(0, count - len(picked))])
        return picked[:count]
    picked = [work for _, _, work in scored[:count]]
    picked.extend(unknowns[: max(0, count - len(picked))])
    return picked[:count]
