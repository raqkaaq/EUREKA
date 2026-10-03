"""Deterministic evidence resolution (pure: schema only).

Evidence links are resolved *after* the model run by mapping the model's
integer ``evidence`` indices against the candidate list that was embedded
in the prompt. Out-of-range / duplicate indices are dropped. The model
never writes URLs, so it cannot hallucinate them.
"""

from __future__ import annotations

import urllib.parse as _urlparse

from radar.schema.opportunities import (
    EvidenceLink,
    Opportunity,
    RadarDraft,
    RadarReport,
)
from radar.schema.papers import CollectedWork


def _link_for(index: int, candidates: list[CollectedWork]) -> EvidenceLink | None:
    if not isinstance(index, int) or isinstance(index, bool):
        return None
    if not (0 <= index < len(candidates)):
        return None
    work = candidates[index]
    url = work.primary_url or work.openalex_id
    parsed = _urlparse.urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        url = work.openalex_id
    return EvidenceLink(
        index=index,
        title=work.title or "(untitled)",
        url=url,
        openalex_id=work.openalex_id,
    )


def attach_evidence(draft: RadarDraft, candidates: list[CollectedWork]) -> RadarReport:
    """Attach deterministic evidence links to a model draft."""
    opportunities: list[Opportunity] = []
    for item in draft.opportunities:
        seen: set[int] = set()
        links: list[EvidenceLink] = []
        for raw in item.evidence:
            link = _link_for(raw, candidates)
            if link is None or link.index in seen:
                continue
            seen.add(link.index)
            links.append(link)
        opportunities.append(Opportunity(draft=item, evidence_links=links))
    ignore = [str(x).strip()[:300] for x in draft.ignore if str(x).strip()][:10]
    return RadarReport(
        opportunities=opportunities,
        ignore=ignore,
        next_move=draft.next_move.strip()[:2000],
    )


__all__ = ["attach_evidence"]
