"""Deterministic evidence attachment + Markdown rendering.

Evidence links are attached *after* the model run by resolving the model's
integer ``evidence`` indices against the candidate list that was embedded
in the prompt. Out-of-range / duplicate indices are dropped. The model
never writes URLs, so it cannot hallucinate them.
"""

from __future__ import annotations

import urllib.parse as _urlparse

from radar.models import (
    CollectedWork,
    EvidenceLink,
    Opportunity,
    RadarDraft,
    RadarReport,
)


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


def _escape_link_label(value: str) -> str:
    """Escape untrusted text used inside a Markdown link label."""
    return value.replace("\\", "\\\\").replace("[", "\\[").replace("]", "\\]")


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


def render_markdown(report: RadarReport) -> str:
    """Render the report with The Wow / Investigate / Reproduce / Ignore / Next move."""
    lines: list[str] = ["# AI/ML Opportunity Radar", ""]
    if not report.opportunities:
        lines.append("No opportunities cleared the bar this run.")
        lines.append("")
    for n, opp in enumerate(report.opportunities, 1):
        d = opp.draft
        lines.append(f"## {n}. {d.title}")
        lines.append("")
        lines.append(f"**The Wow:** {d.wow or '—'}")
        lines.append("")
        lines.append(f"**Investigate:** {d.investigate or '—'}")
        lines.append("")
        lines.append(f"**Reproduce:** {d.reproduce or '—'}")
        lines.append("")
        if opp.evidence_links:
            lines.append("**Evidence:**")
            for link in opp.evidence_links:
                title = _escape_link_label(link.title)
                lines.append(f"- [{title}]({link.url}) ({link.openalex_id})")
        else:
            lines.append("**Evidence:** none (no valid candidate indices).")
        lines.append("")
    lines.append("## Ignore")
    lines.append("")
    if report.ignore:
        for item in report.ignore:
            lines.append(f"- {item}")
    else:
        lines.append("Nothing flagged to ignore.")
    lines.append("")
    lines.append("## Next move")
    lines.append("")
    lines.append(report.next_move or "—")
    lines.append("")
    return "\n".join(lines)
