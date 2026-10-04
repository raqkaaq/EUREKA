"""Opportunity analysis agent: instructions, prompt budget, typed runs.

Owns the PydanticAI ``Agent[None, RadarDraft]`` wiring, the analysis
instructions (system + task footer), budget-aware prompt selection, per-run
``ModelSettings``/``UsageLimits``, the hard overall deadline, and output
validation. Owns no provider client (a session is passed in) and no
rendering (see :mod:`radar.output.markdown`).

Prompt-fit invariant: only complete candidate blocks are submitted, the
task footer always survives, and every paper counted as analyzed is
present in the submitted input. Callers derive coverage from the actual
included list, never the requested bound.
"""

from __future__ import annotations

import asyncio as _asyncio
from typing import TYPE_CHECKING

from radar.config.runtime import (
    ANALYSIS_MAX_TIMEOUT_S,
    ANALYSIS_MAX_TOKENS,
    ANALYSIS_REQUEST_LIMIT,
    ANALYSIS_REQUEST_TIMEOUT_S,
    ANALYSIS_RETRIES,
    ANALYSIS_TIMEOUT_S,
    DEFAULT_MAX_CANDIDATES,
    MAX_ABSTRACT_IN_PROMPT,
    MAX_ANALYSIS_OPPORTUNITIES,
    MAX_CANDIDATES_IN_PROMPT,
    MAX_PROMPT_CHARS,
    MAX_TITLE_IN_PROMPT,
    validate_analysis_timeout,
)
from radar.processing.ranking import bound_candidates
from radar.prompts.catalog import opportunity_analysis_prompt
from radar.provider import strata as _strata
from radar.schema.opportunities import RadarDraft
from radar.schema.papers import CollectedWork
from radar.schema.configuration import AnalysisPrompt

if TYPE_CHECKING:
    from pydantic_ai import Agent as _Agent
    from pydantic_ai.models import Model as _Model
    from pydantic_ai.settings import ModelSettings as _ModelSettings
    from pydantic_ai.usage import UsageLimits as _UsageLimits


def _normalize(text: str) -> str:
    return " ".join((text or "").split())


def _truncate(text: str, limit: int) -> str:
    text = _normalize(text)
    if len(text) > limit:
        return text[:limit].rstrip() + "…"
    return text


def _header(spec: AnalysisPrompt) -> str:
    return spec.candidate_header


def _candidate_block(index: int, work: CollectedWork, *, complete_abstracts: bool = False) -> str:
    title = _truncate(work.title or "(untitled)", MAX_TITLE_IN_PROMPT)
    if complete_abstracts:
        # Intact full normalized abstract: never truncate tails. Oversized
        # leading sources simply yield zero honest coverage in selection.
        abstract = _normalize(work.abstract) or "(no abstract)"
    else:
        abstract = _truncate(work.abstract or "(no abstract)", MAX_ABSTRACT_IN_PROMPT)
    coverage = "complete abstract" if complete_abstracts else "abstract excerpt"
    if not work.abstract.strip():
        coverage = "no abstract available"
    year = work.publication_year or "n/a"
    return (
        f"[{index}] {title} ({year}, cited_by={work.cited_by_count})\n"
        f"--- begin untrusted candidate {index} data ---\n"
        f"    Source coverage: {coverage}\n"
        f"    {abstract}\n"
        f"--- end untrusted candidate {index} data ---"
    )


def _footer(included: int, spec: AnalysisPrompt) -> str:
    if included <= 0:
        valid_range = "none (no candidates)"
    else:
        valid_range = f"0..{included - 1}"
    return spec.task_template.format(
        max_opportunities=spec.max_opportunities, valid_range=valid_range)


def select_for_prompt(
    candidates: list[CollectedWork],
    max_candidates: int = MAX_CANDIDATES_IN_PROMPT,
    *,
    prompt_spec: AnalysisPrompt | None = None,
    reserved_context_chars: int = 0,
    complete_abstracts: bool = False,
) -> list[CollectedWork]:
    """Budget-aware complete-block selection for the prompt.

    Takes the ranked topN slice, then keeps the longest leading run of
    whole candidate blocks that fits in ``MAX_PROMPT_CHARS`` with the
    header and the full task footer always reserved. Every returned paper
    is present verbatim in the submitted input. With ``complete_abstracts``
    the intact full normalized abstract sizes the block; tails are never
    truncated, so an oversized leading source yields zero honest coverage.
    """
    bounded = bound_candidates(candidates, max_candidates)
    spec = prompt_spec if prompt_spec is not None else opportunity_analysis_prompt()
    if not 0 <= reserved_context_chars <= MAX_PROMPT_CHARS:
        raise ValueError("invalid reserved prompt context budget")
    header = _header(spec)
    footer = _footer(len(bounded), spec)
    # Reserve the worst-case footer (index width only shrinks when fewer
    # papers are included, so fitting against the full footer is safe).
    # Instructions travel separately, but still consume the same prompt budget.
    used = len(spec.instructions) + len(header) + len(footer) + reserved_context_chars + 6
    included: list[CollectedWork] = []
    for index, work in enumerate(bounded):
        block = _candidate_block(index, work, complete_abstracts=complete_abstracts)
        if used + len(block) + 1 > MAX_PROMPT_CHARS:
            break
        included.append(work)
        used += len(block) + 1
    return included


def build_prompt(
    candidates: list[CollectedWork],
    max_candidates: int = MAX_CANDIDATES_IN_PROMPT,
    *,
    prompt_spec: AnalysisPrompt | None = None,
    context: str = "",
    complete_abstracts: bool = False,
) -> str:
    """Build the bounded analysis prompt (complete blocks + intact footer)."""
    spec = prompt_spec if prompt_spec is not None else opportunity_analysis_prompt()
    included = select_for_prompt(candidates, max_candidates, prompt_spec=spec,
                                 reserved_context_chars=len(context),
                                 complete_abstracts=complete_abstracts)
    parts = [_header(spec)]
    parts.extend(_candidate_block(i, work, complete_abstracts=complete_abstracts)
                 for i, work in enumerate(included))
    if context:
        parts.append(context)
    parts.append(_footer(len(included), spec))
    return "\n".join(parts)


def pdf_candidate_block(index: int, work: CollectedWork, reading) -> str:
    """Full-text candidate block from a completed PDF reading."""
    from radar.schema.documents import PDFReading as _PDFReading

    if not isinstance(reading, _PDFReading):
        raise ValueError("PDF block requires a completed PDFReading")
    title = _truncate(work.title or "(untitled)", MAX_TITLE_IN_PROMPT)
    year = work.publication_year or "n/a"
    notes_json = reading.notes.model_dump_json()
    quotes = "; ".join(
        f"p{item.page}: {item.quote!r}" for item in reading.notes.evidence) or "(no verified quotes)"
    return (
        f"[{index}] {title} ({year}, cited_by={work.cited_by_count})\n"
        f"--- begin untrusted candidate {index} data ---\n"
        f"    Source coverage: full extracted PDF text read via "
        f"{len(reading.chunks)} chunks ({reading.page_count} pages, "
        f"{reading.text_chars} chars)\n"
        f"    Extraction warning: {reading.extraction_warning}\n"
        f"    Final reading notes: {notes_json}\n"
        f"    Verified page quotes: {quotes}\n"
        f"--- end untrusted candidate {index} data ---"
    )


def select_for_pdf_prompt(
    candidates: list[CollectedWork],
    readings_by_id: dict[str, object],
    max_candidates: int = MAX_CANDIDATES_IN_PROMPT,
    *,
    prompt_spec: AnalysisPrompt | None = None,
    reserved_context_chars: int = 0,
    max_chars: int | None = None,
) -> list[CollectedWork]:
    """Common complete-paper cohort over full-text PDF blocks.

    Only candidates with a completed reading are eligible; every returned
    paper is present verbatim in the submitted input within ``max_chars``.
    """
    from radar.config.documents import PDF_RESEARCH_PROMPT_CHARS

    bound = PDF_RESEARCH_PROMPT_CHARS if max_chars is None else max_chars
    spec = prompt_spec if prompt_spec is not None else opportunity_analysis_prompt()
    if not 0 <= reserved_context_chars <= bound:
        raise ValueError("invalid reserved prompt context budget")
    bounded = bound_candidates(candidates, max_candidates)
    eligible = [w for w in bounded if w.openalex_id in readings_by_id]
    header = _header(spec)
    footer = _footer(len(eligible), spec)
    used = len(spec.instructions) + len(header) + len(footer) + reserved_context_chars + 6
    included: list[CollectedWork] = []
    for work in eligible:
        block = pdf_candidate_block(len(included), work, readings_by_id[work.openalex_id])
        if used + len(block) + 1 > bound:
            break
        included.append(work)
        used += len(block) + 1
    return included


def build_pdf_prompt(
    candidates: list[CollectedWork],
    readings_by_id: dict[str, object],
    max_candidates: int = MAX_CANDIDATES_IN_PROMPT,
    *,
    prompt_spec: AnalysisPrompt | None = None,
    context: str = "",
    max_chars: int | None = None,
) -> str:
    """Build the bounded full-text PDF prompt (complete blocks + footer)."""
    from radar.config.documents import PDF_RESEARCH_PROMPT_CHARS

    bound = PDF_RESEARCH_PROMPT_CHARS if max_chars is None else max_chars
    spec = prompt_spec if prompt_spec is not None else opportunity_analysis_prompt()
    included = select_for_pdf_prompt(
        candidates, readings_by_id, max_candidates, prompt_spec=spec,
        reserved_context_chars=len(context), max_chars=bound)
    parts = [_header(spec)]
    parts.extend(pdf_candidate_block(i, work, readings_by_id[work.openalex_id])
                 for i, work in enumerate(included))
    if context:
        parts.append(context)
    parts.append(_footer(len(included), spec))
    prompt = "\n".join(parts)
    if len(spec.instructions) + len(prompt) + 2 > bound:
        raise ValueError("PDF prompt exceeded its bound")
    return prompt


def run_bounds(
    disable_thinking: bool = False,
    max_tokens: int = ANALYSIS_MAX_TOKENS,
    request_timeout_s: float = ANALYSIS_REQUEST_TIMEOUT_S,
    request_limit: int = ANALYSIS_REQUEST_LIMIT,
    analysis_timeout_s: float = ANALYSIS_TIMEOUT_S,
) -> tuple["_ModelSettings", "_UsageLimits", int]:
    """Build the bounded per-run settings: ``(model_settings, usage_limits, retries)``.

    Pure seam: no network, no model. ``request_timeout_s`` must fit inside
    the remaining overall budget; the thinking-disable key is included only
    when explicitly opted in.
    """
    from radar.config.runtime import validate_max_tokens

    analysis_timeout = validate_analysis_timeout(analysis_timeout_s)
    tokens = validate_max_tokens(max_tokens)
    try:
        limit = int(request_limit)
    except (TypeError, ValueError) as exc:
        raise ValueError("request_limit must be 1 or 2") from exc
    if limit not in (1, 2):
        raise ValueError("request_limit must be 1 or 2")
    try:
        per_request = float(request_timeout_s)
    except (TypeError, ValueError) as exc:
        raise ValueError("request timeout must be within (0, overall deadline]s") from exc
    import math as _math

    if not _math.isfinite(per_request) or not (0 < per_request <= analysis_timeout):
        raise ValueError("request timeout must be within (0, overall deadline]s")
    settings: _ModelSettings = {"max_tokens": tokens, "timeout": per_request}  # type: ignore[typeddict-item]
    if disable_thinking:
        settings["extra_body"] = _strata.thinking_extra_body()  # type: ignore[typeddict-unknown-key]
    from pydantic_ai.usage import UsageLimits

    return settings, UsageLimits(request_limit=limit), ANALYSIS_RETRIES


def build_agent(model: "_Model") -> "_Agent[None, RadarDraft]":
    """Build the analysis agent around an explicitly provided model.

    The model (or session) always comes from the pipeline via the
    provider; the agent never resolves configuration itself, so no
    unowned client lifetime can leak here.
    """
    from pydantic_ai import Agent

    return Agent(model, output_type=RadarDraft, retries=ANALYSIS_RETRIES,
                 instructions=opportunity_analysis_prompt().instructions)


def _validation_categories(exc: BaseException) -> list[str]:
    """Summarize output-validation failures as ``loc: error-type`` entries.

    Only field paths and error types are recorded -- never offending values,
    prompts, or secrets -- so the summary is safe for errors and logs.
    """
    categories: list[str] = []
    node: BaseException | None = exc
    seen = 0
    while node is not None and seen < 8:
        if type(node).__name__ == "ValidationError":
            errors = getattr(node, "errors", None)
            if callable(errors):
                try:
                    for entry in errors(include_url=False):
                        if isinstance(entry, dict):
                            loc = ".".join(str(p) for p in entry.get("loc", ()))
                            categories.append(f"{loc or '?'}: {entry.get('type', '?')}")
                except Exception:
                    pass
        node = node.__cause__
        seen += 1
    return categories[:12]


def _actionable(exc: Exception, disable_thinking: bool) -> _strata.StrataError:
    text = str(exc)
    lowered = text.lower()
    if disable_thinking and (
        "400" in text or "bad request" in lowered or "extra_body" in lowered
    ):
        return _strata.StrataError(
            f"Strata server rejected the optional thinking-disable key "
            f"({type(exc).__name__}: {exc}). Rerun without --disable-thinking; "
            "that server-specific key is not supported by every backend."
        )
    categories = _validation_categories(exc)
    detail = (
        f" Output failed validation ({'; '.join(categories)})"
        if categories
        else ""
    )
    return _strata.StrataError(
        f"Strata inference failed ({type(exc).__name__}: {exc}).{detail} "
        "Check that your user-owned Strata server is serving "
        "OpenAI-compatible Chat Completions at the configured private-network "
        "endpoint from STRATA_BASE_URL (legacy FREETOKEN_BASE_URL)/--base-url and that "
        "STRATA_MODEL names a served model."
    )


async def analyze_candidates_async(
    candidates: list[CollectedWork],
    model: "_Model | None" = None,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
    analysis_timeout_s: float = ANALYSIS_TIMEOUT_S,
    max_tokens: int = ANALYSIS_MAX_TOKENS,
    request_limit: int = ANALYSIS_REQUEST_LIMIT,
    disable_thinking: bool = False,
    session: "_strata.StrataSession | None" = None,
) -> tuple[RadarDraft, str]:
    """Run the radar analysis under a hard overall deadline.

    The deadline cancels the async ``Agent.run`` itself (a blocked sync run
    could not be interrupted). When ``session`` is given, its model is used
    and its HTTP client is closed in the same event loop once the run
    settles (including on cancellation), so cleanup is deterministic.
    Raises :class:`strata.StrataError` on deadline breach or
    inference failure.
    """
    try:
        settings, limits, retries = run_bounds(
            disable_thinking=disable_thinking,
            max_tokens=max_tokens,
            request_timeout_s=min(ANALYSIS_REQUEST_TIMEOUT_S, float(analysis_timeout_s)),
            analysis_timeout_s=analysis_timeout_s,
            request_limit=request_limit,
        )
        prompt = build_prompt(candidates, max_candidates=max_candidates)
    except ValueError as exc:
        if session is not None:
            try:
                await session.http_client.aclose()
            except Exception:
                pass
        raise
    if session is not None:
        model = session.model
    if model is None:
        if session is not None:
            try:
                await session.http_client.aclose()
            except Exception:
                pass
        raise _strata.StrataError(
            "No model or session was provided to the analysis agent; "
            "the pipeline must supply one."
        )
    try:
        agent = build_agent(model)
        async with _asyncio.timeout(float(analysis_timeout_s)):
            result = await agent.run(
                prompt,
                model_settings=settings,  # type: ignore[arg-type]
                usage_limits=limits,
                retries=retries,
            )
    except (TimeoutError, _asyncio.CancelledError) as exc:
        raise _strata.StrataError(
            f"Strata analysis exceeded the overall {analysis_timeout_s}s "
            f"deadline ({type(exc).__name__}); no partial report was produced. "
            "Retry with fewer candidates, a smaller --max-tokens, or a larger "
            "--analysis-timeout."
        ) from exc
    except _strata.StrataError:
        raise
    except Exception as exc:
        raise _actionable(exc, disable_thinking) from exc
    finally:
        if session is not None:
            # Same-loop deterministic cleanup, including on cancellation.
            # A close failure must not mask the analysis outcome.
            try:
                await session.http_client.aclose()
            except Exception:
                pass
    output = result.output
    if not isinstance(output, RadarDraft):
        # Defensive: Agent(output_type=RadarDraft) must return RadarDraft;
        # coerce when a test double returns a mapping.
        try:
            output = RadarDraft.model_validate(output)
        except Exception as exc:
            raise _strata.StrataError(
                f"Model returned output that does not validate as RadarDraft: {exc}"
            ) from exc
    return output, prompt


def analyze_candidates(
    candidates: list[CollectedWork],
    model: "_Model | None" = None,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
    analysis_timeout_s: float = ANALYSIS_TIMEOUT_S,
    max_tokens: int = ANALYSIS_MAX_TOKENS,
    request_limit: int = ANALYSIS_REQUEST_LIMIT,
    disable_thinking: bool = False,
    session: "_strata.StrataSession | None" = None,
) -> tuple[RadarDraft, str]:
    """Run the radar analysis; return ``(draft, prompt)``.

    Synchronous wrapper around :func:`analyze_candidates_async` (fresh event
    loop per call, never nested). Raises :class:`strata.StrataError`
    when the local model endpoint is unreachable/misconfigured or the
    overall deadline is breached.
    """
    return _asyncio.run(
        analyze_candidates_async(
            candidates,
            model=model,
            max_candidates=max_candidates,
            analysis_timeout_s=analysis_timeout_s,
            max_tokens=max_tokens,
            request_limit=request_limit,
            disable_thinking=disable_thinking,
            session=session,
        )
    )


__all__ = [
    "analyze_candidates",
    "analyze_candidates_async",
    "build_agent",
    "build_pdf_prompt",
    "build_prompt",
    "pdf_candidate_block",
    "run_bounds",
    "select_for_pdf_prompt",
    "select_for_prompt",
]
