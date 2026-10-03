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
