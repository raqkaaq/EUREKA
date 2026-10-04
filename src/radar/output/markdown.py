"""Markdown report rendering plus plain-text status lines (pure: schema only).

No JSON formatting here: machine-readable envelopes live in
:mod:`radar.output.json`. No printing: callers print.
"""

from __future__ import annotations

import datetime as _dt

from radar.schema.opportunities import RadarReport


def _escape_link_label(value: str) -> str:
    """Escape untrusted text used inside a Markdown link label."""
    return value.replace("\\", "\\\\").replace("[", "\\[").replace("]", "\\]")


def _opportunity_lines(report: RadarReport) -> list[str]:
    """Keep the legacy opportunity section separate from primary study content."""
    lines: list[str] = []
    if not report.opportunities:
        lines.append("No additional research opportunities proposed." if report.learning_dossiers
                     else "No opportunities cleared the bar this run.")
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
    return lines


def render_markdown(report: RadarReport) -> str:
    """Lead with grounded learning; preserve opportunity-only report rendering."""
    title = "AI/ML Learning Radar" if report.learning_dossiers else "AI/ML Opportunity Radar"
    lines: list[str] = [f"# {title}", ""]
    for resolved in report.learning_dossiers:
        dossier = resolved.dossier
        source = resolved.source
        lines.append("## Learning dossier (primary, abstract-level only)")
        lines.append("")
        lines.append(f"**Core problem:** {dossier.core_problem or '—'}")
        lines.append("")
        lines.append(f"**Reported contribution:** {dossier.reported_contribution or '—'}")
        lines.append("")
        lines.append(f"**Reasoning:** {dossier.reasoning or '—'}")
        lines.append("")
        lines.append(f"**Significance (interpretation):** {dossier.significance or '—'}")
        lines.append("")
        lines.append("**Assumptions/limits:**")
        for item in dossier.assumptions_limits:
            lines.append(f"- {item}")
        lines.append("")
        lines.append("**Hypothesized connections (not reported findings):**")
        lines.append("")
        for item in dossier.connections:
            lines.append(f"- {item}")
        if not dossier.connections:
            lines.append("No defensible transfer proposed.")
        lines.append("")
        lines.append("**Study tasks:**")
        lines.append("")
        for task in dossier.study_tasks:
            lines.append(f"- [{task.kind}] {task.objective}")
            lines.append("")
            lines.append(f"  - Success criterion: {task.success_criterion}")
            lines.append(f"  - Missing evidence: {task.missing_evidence}")
        lines.append("")
        lines.append("**Open questions:**")
        for item in dossier.open_questions:
            lines.append(f"- {item}")
        lines.append("")
        title = _escape_link_label(source.title)
        lines.append(f"**Source (abstract-level evidence):** [{title}]({source.url}) "
                     f"({source.openalex_id}; evidence_level={resolved.evidence_level})")
        lines.append("")
    lines.extend(_opportunity_lines(report))
    lines.append("## Next move")
    lines.append("")
    lines.append(report.next_move or "—")
    lines.append("")
    return "\n".join(lines)


def coverage_line(
    pool_total: int, selected: int, opportunities: int
) -> str:
    """Honest one-line coverage: pool size vs LLM-read papers vs findings.

    ``selected``/``analyzed`` are the papers actually sent to the model
    (identical by construction); ``opportunities`` counts findings, never
    papers. The whole pool is never implied LLM-read.
    """
    return (
        f"radar: coverage selected={selected} analyzed={selected} "
        f"opportunities={opportunities} "
        f"(pool={pool_total}; bounded discovery sample, not all of OpenAlex)"
    )


def staleness_note(collected_at_utc: str) -> str:
    """Human-readable disclosure of a cached snapshot's age."""
    try:
        taken = _dt.datetime.fromisoformat(collected_at_utc)
        if taken.tzinfo is None:
            taken = taken.replace(tzinfo=_dt.timezone.utc)
        age = _dt.datetime.now(_dt.timezone.utc) - taken
        seconds = max(0, int(age.total_seconds()))
    except (ValueError, TypeError):
        return f"snapshot collected at {collected_at_utc} (age unknown)"
    if seconds < 90:
        age_note = f"{seconds}s old"
    elif seconds < 5400:
        age_note = f"{seconds // 60}m old"
    else:
        age_note = f"{seconds // 3600}h old"
    return f"snapshot collected at {collected_at_utc} ({age_note})"


__all__ = ["coverage_line", "render_markdown", "staleness_note"]
