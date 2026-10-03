"""Deterministic triage selection policy (pure: schema only).

No network, no LLM, no I/O. Scored works rank by the maximum of their
two screening probabilities with a stable OpenAlex-ID tie-break.
Exactly one unknown slot is reserved (first unknown in input heuristic
order) whenever unknowns exist and the shortlist holds two or more;
remaining slots fall back to heuristic order. No hard filters: every
input work stays eligible, so unknown works are never dropped.
"""

from __future__ import annotations

from radar.schema.papers import CollectedWork
from radar.schema.triage import TriageBatch


def _relevance(result) -> float:
    probs = [result.ai_ml_relevance, result.cross_domain_potential]
    return max(p for p in probs if p is not None)


def select_candidates(
    works: list[CollectedWork], batch: TriageBatch, n: int
) -> list[CollectedWork]:
    """Select ``n`` works: scored-ranked first, one unknown slot reserved."""
    try:
        count = int(n)
    except (TypeError, ValueError):
        return []
    if count <= 0 or not works:
        return []
    by_id = {result.work_id: result for result in batch.results}
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
