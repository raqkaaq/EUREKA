"""Deterministic evidence resolution (pure: schema only).

Evidence links are resolved *after* the model run by mapping the model's
integer ``evidence`` indices against the candidate list that was embedded
in the prompt. Out-of-range / duplicate opportunity indices are dropped.
Learning dossier primary indices are never silently dropped: an
out-of-range or unsafe dossier source raises, so the final validator (not
silent filtering) owns the failure. The model never writes URLs, so it
cannot hallucinate them.
"""

from __future__ import annotations

from radar.processing.link_validation import is_http_link, is_openalex_work_link

from radar.schema.opportunities import (
    EvidenceLink,
    LearningRadarDraft,
    Opportunity,
    RadarDraft,
    RadarReport,
    ResolvedLearningDossier,
)
from radar.schema.papers import CollectedWork


def _link_for(index: int, candidates: list[CollectedWork]) -> EvidenceLink | None:
    if not isinstance(index, int) or isinstance(index, bool):
        return None
    if not (0 <= index < len(candidates)):
        return None
    work = candidates[index]
    if is_http_link(work.primary_url):
        url = work.primary_url
    elif is_openalex_work_link(work.openalex_id):
        url = work.openalex_id
    else:
        return None  # No safe source link: never invent or emit an unsafe fallback.
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
    dossiers: list[ResolvedLearningDossier] = []
    if isinstance(draft, LearningRadarDraft):
        seen_primary: set[int] = set()
        for dossier in draft.learning_dossiers:
            index = dossier.paper_index
            if index in seen_primary:
                raise ValueError(f"Duplicate primary dossier index: {index}")
            seen_primary.add(index)
            link = _link_for(index, candidates)
            if link is None:
                raise ValueError(
                    f"Learning dossier source index {index} has no safe candidate link; "
                    "invalid learning sources are never silently dropped."
                )
            if not candidates[index].abstract.strip():
                raise ValueError("An abstract-level learning dossier requires a supplied abstract.")
            dossiers.append(ResolvedLearningDossier(
                dossier=dossier, source=link, evidence_level="abstract"))
    elif getattr(draft, "learning_dossiers", None):
        raise ValueError("Unexpected learning dossiers on a non-learning draft.")
    ignore = [str(x).strip()[:300] for x in draft.ignore if str(x).strip()][:10]
    return RadarReport(
        opportunities=opportunities,
        ignore=ignore,
        next_move=draft.next_move.strip()[:2000],
        learning_dossiers=dossiers,
    )


__all__ = ["attach_evidence"]
