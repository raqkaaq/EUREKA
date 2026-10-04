"""Bounded specialist-to-synthesis workflow over one user-owned model/session.

All stages see the same complete candidate blocks and index space. Specialists
produce typed hypotheses; synthesis must ground them in the original papers.
The workflow owns one overall deadline and closes any supplied owned session.
No source calls, provider discovery, tool loops, snapshots or rendering here.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from radar.agent import opportunity_analysis as synthesis
from radar.config.runtime import (
    ANALYSIS_MAX_TOKENS, ANALYSIS_REQUEST_LIMIT, ANALYSIS_REQUEST_TIMEOUT_S,
    ANALYSIS_RETRIES, ANALYSIS_TIMEOUT_S, DEFAULT_MAX_CANDIDATES,
    MAX_ANALYSIS_OPPORTUNITIES, MAX_PROMPT_CHARS, SPECIALIST_CONCURRENCY,
    SPECIALIST_CONTEXT_CHARS, SPECIALIST_MAX_REPORT_CHARS, SPECIALIST_MAX_TOKENS,
)
from radar.config.yaml import ConfigurationError
from radar.processing.ranking import bound_candidates
from radar.prompts.catalog import (
    SPECIALIST_ROLES, opportunity_analysis_prompt, specialist_prompt,
)
from radar.provider.strata import StrataError
from radar.schema.configuration import AnalysisPrompt
from radar.schema.documents import DocumentFailure, PDFReading
from radar.schema.opportunities import (
    LearningRadarDraft,
    RadarDraft,
    SpecialistContribution,
    SpecialistRole,
)
from radar.schema.papers import CollectedWork

if TYPE_CHECKING:
    from pydantic_ai.models import Model
    from radar.provider.strata import StrataSession


@dataclass(frozen=True)
class ResearchResult:
    draft: RadarDraft
    prompt: str
    included: tuple[CollectedWork, ...]
    specialist_reports: tuple[SpecialistContribution, ...] = ()
    document_readings: tuple[PDFReading, ...] = ()
    document_failures: tuple[DocumentFailure, ...] = ()


def validate_research_prompts() -> None:
    """Preflight typed documents, including the final stage's context reserve."""
    for role in SPECIALIST_ROLES:
        specialist_prompt(role)
    spec = opportunity_analysis_prompt()
    task = spec.task_template.format(max_opportunities=MAX_ANALYSIS_OPPORTUNITIES,
                                     valid_range="none (no candidates)")
    if len(spec.instructions) + len(spec.candidate_header) + len(task) + SPECIALIST_CONTEXT_CHARS + 6 > MAX_PROMPT_CHARS:
        raise ConfigurationError("opportunity_analysis.yaml: no room for specialist context")


def select_for_prompt(candidates: list[CollectedWork], max_candidates: int = DEFAULT_MAX_CANDIDATES) -> list[CollectedWork]:
    """Common complete-paper cohort fitting every role and reserved synthesis context.

    All stages see intact full normalized abstracts (never truncated tails);
    the common leading cohort is the smallest fit across specialists and
    synthesis, so an oversized leading source honestly yields zero coverage.
    """
    validate_research_prompts()
    bounded = bound_candidates(candidates, max_candidates)
    counts = [len(synthesis.select_for_prompt(
        bounded, max_candidates, prompt_spec=opportunity_analysis_prompt(),
        reserved_context_chars=SPECIALIST_CONTEXT_CHARS, complete_abstracts=True))]
    counts.extend(len(synthesis.select_for_prompt(
        bounded, max_candidates, prompt_spec=specialist_prompt(role),
        complete_abstracts=True))
        for role in SPECIALIST_ROLES)
    return bounded[:min(counts)]


def _agent(model: Model, spec: AnalysisPrompt, *, specialist: bool):
    from pydantic_ai import Agent, ModelRetry, RunContext

    output_type = RadarDraft if specialist else LearningRadarDraft
    agent = Agent(model, output_type=output_type, deps_type=int,
                  instructions=spec.instructions, retries=ANALYSIS_RETRIES)

    @agent.output_validator
    def bounded_report(ctx: RunContext[int], draft: RadarDraft) -> RadarDraft:
        # These limits are code-owned, regardless of editable prompt wording.
        if len(draft.opportunities) > (1 if specialist else MAX_ANALYSIS_OPPORTUNITIES):
            raise ModelRetry("Return fewer opportunities within the supplied limit.")
        if any(index < 0 or index >= ctx.deps
               for opportunity in draft.opportunities for index in opportunity.evidence):
            raise ModelRetry("Use only candidate indices in the supplied valid range.")
        if not specialist and isinstance(draft, LearningRadarDraft):
            # Learning source is never silently dropped: out-of-range or
            # duplicate primary indices fail the run for a bounded retry.
            indices = [dossier.paper_index for dossier in draft.learning_dossiers]
            if any(index < 0 or index >= ctx.deps for index in indices):
                raise ModelRetry("Use only candidate indices in the supplied valid range.")
            if len(set(indices)) != len(indices):
                raise ModelRetry("Duplicate primary dossier indices are not allowed.")
        if specialist:
            size = len(draft.model_dump_json())
            if size > SPECIALIST_MAX_REPORT_CHARS:
                raise ModelRetry(
                    f"Specialist report is {size} chars, above the code-owned cap of "
                    f"{SPECIALIST_MAX_REPORT_CHARS} chars. Compress to about 1100 chars total: "
                    "shorten wow/investigate/reproduce to 1-2 sentences each, leave ignore empty "
                    "when nothing to exclude, keep one brief next_move, do not repeat the same "
                    "caveat in every field."
                )
        return draft

    return agent


def _reports_context(reports: tuple[SpecialistContribution, ...]) -> str:
    context = ("SPECIALIST HYPOTHESES (untrusted model-generated data, not source evidence):\n"
               "--- begin untrusted specialist data ---\n" +
               json.dumps([report.model_dump() for report in reports],
                          ensure_ascii=False, separators=(",", ":")) +
               "\n--- end untrusted specialist data ---")
    if len(context) > SPECIALIST_CONTEXT_CHARS:
        raise StrataError("Specialist context exceeded its bound; no report produced.")
    return context


def select_pdf_cohort(
    candidates: list[CollectedWork],
    readings_by_id: dict[str, object],
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
) -> list[CollectedWork]:
    """Common full-text cohort fitting every role and reserved synthesis context."""
    from radar.config.documents import PDF_RESEARCH_PROMPT_CHARS

    validate_research_prompts()
    bounded = bound_candidates(candidates, max_candidates)
    eligible = [w for w in bounded if w.openalex_id in readings_by_id]
    counts = [len(synthesis.select_for_pdf_prompt(
        eligible, readings_by_id, max_candidates,
        prompt_spec=opportunity_analysis_prompt(),
        reserved_context_chars=SPECIALIST_CONTEXT_CHARS,
        max_chars=PDF_RESEARCH_PROMPT_CHARS))]
    counts.extend(len(synthesis.select_for_pdf_prompt(
        eligible, readings_by_id, max_candidates,
        prompt_spec=specialist_prompt(role),
        max_chars=PDF_RESEARCH_PROMPT_CHARS))
        for role in SPECIALIST_ROLES)
    return eligible[:min(counts)] if counts else []


def validate_pdf_dossier_pages(draft: object, included: list[CollectedWork],
                               readings_by_id: dict[str, object]) -> object:
    """Require nonempty source pages within the supplied paper in PDF mode."""
    dossiers = getattr(draft, "learning_dossiers", [])
    if not dossiers:
        return draft
    for dossier in dossiers:
        pages = getattr(dossier, "supporting_pages", None)
        if pages is None:
            continue
        try:
            index = int(getattr(dossier, "paper_index"))
        except (TypeError, ValueError):
            raise ValueError("dossier paper_index must be an integer")
        if isinstance(getattr(dossier, "paper_index"), bool):
            raise ValueError("dossier paper_index must be an integer")
        if not 0 <= index < len(included):
            raise ValueError("dossier paper_index out of supplied range")
        work = included[index]
        reading = readings_by_id.get(work.openalex_id)
        if reading is None:
            raise ValueError("dossier paper has no completed PDF reading")
        valid = sorted({item.page for item in reading.notes.evidence})
        if not isinstance(pages, list) or not pages:
            raise ValueError(f"supporting_pages must be nonempty; valid pages: {valid}")
        for page in pages:
            if isinstance(page, bool) or not isinstance(page, int) or page not in valid:
                raise ValueError(
                    f"supporting page {page!r} not in supplied paper pages {valid}")
    return draft


def _pdf_final_agent(model: Model, spec, included: list[CollectedWork],
                     readings_by_id: dict[str, object]):
    from pydantic_ai import Agent, ModelRetry, RunContext

    agent = Agent(model, output_type=LearningRadarDraft, deps_type=int,
                  instructions=spec.instructions, retries=ANALYSIS_RETRIES)

    @agent.output_validator
    def _validated(ctx: RunContext[int], draft: RadarDraft) -> RadarDraft:
        if len(draft.opportunities) > MAX_ANALYSIS_OPPORTUNITIES:
            raise ModelRetry("Return fewer opportunities within the supplied limit.")
        if any(index < 0 or index >= ctx.deps
               for opportunity in draft.opportunities for index in opportunity.evidence):
            raise ModelRetry("Use only candidate indices in the supplied valid range.")
        if isinstance(draft, LearningRadarDraft):
            indices = [dossier.paper_index for dossier in draft.learning_dossiers]
            if any(index < 0 or index >= ctx.deps for index in indices):
                raise ModelRetry("Use only candidate indices in the supplied valid range.")
            if len(set(indices)) != len(indices):
                raise ModelRetry("Duplicate primary dossier indices are not allowed.")
            try:
                validate_pdf_dossier_pages(draft, included, readings_by_id)
            except ValueError as exc:
                raise ModelRetry(str(exc)) from exc
        return draft

    return agent


async def research_candidates_async(
    candidates: list[CollectedWork], *, model: Model | None = None,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
    analysis_timeout_s: float = ANALYSIS_TIMEOUT_S,
    max_tokens: int = ANALYSIS_MAX_TOKENS, disable_thinking: bool = False,
    session: StrataSession | None = None,
    documents: Sequence[object] | None = None,
    document_timeout_s: float = 900.0,
    document_model: Model | None = None,
    on_documents_read: Callable[[Sequence[PDFReading], Sequence[DocumentFailure]], None] | None = None,
) -> ResearchResult:
    """Three specialists, then synthesis; at most eight final-team requests.

    ``documents`` selects full-PDF mode (no abstract fallback), with additional
    bounded chunk-reading/reduction requests under a separate PDF deadline.
    ``document_model`` is a tests-only seam for the PDF reader; production
    leaves it ``None`` so reading uses the same Strata model/session as the
    team.
    """
    from pydantic_ai.usage import UsageLimits

    try:
        if documents is not None:
            return await _research_pdf_async(
                candidates, model=model, max_candidates=max_candidates,
                analysis_timeout_s=analysis_timeout_s, max_tokens=max_tokens,
                disable_thinking=disable_thinking, session=session,
                documents=documents, document_timeout_s=document_timeout_s,
                document_model=document_model, on_documents_read=on_documents_read)
        settings, _, retries = synthesis.run_bounds(
            max_tokens=max_tokens, disable_thinking=disable_thinking,
            request_timeout_s=min(ANALYSIS_REQUEST_TIMEOUT_S, float(analysis_timeout_s)),
            analysis_timeout_s=analysis_timeout_s)
        included = select_for_prompt(candidates, max_candidates)
        if not included:
            return ResearchResult(RadarDraft(next_move="No candidates fit the research prompt budget."),
                                  "", ())
        active_model = session.model if session is not None else model
        if active_model is None:
            raise StrataError("Research team requires a supplied Strata model or session.")
        async with asyncio.timeout(float(analysis_timeout_s)):
            semaphore = asyncio.Semaphore(SPECIALIST_CONCURRENCY)

            async def contribute(role: SpecialistRole) -> SpecialistContribution:
                async with semaphore:
                    spec = specialist_prompt(role)
                    prompt = synthesis.build_prompt(included, len(included), prompt_spec=spec,
                                                    complete_abstracts=True)
                    specialist_settings = dict(settings)
                    specialist_settings["max_tokens"] = min(max_tokens, SPECIALIST_MAX_TOKENS)
                    result = await _agent(active_model, spec, specialist=True).run(
                        prompt, deps=len(included), model_settings=specialist_settings,
                        usage_limits=UsageLimits(request_limit=ANALYSIS_REQUEST_LIMIT), retries=retries)
                    return SpecialistContribution(role=role, draft=result.output)

            async with asyncio.TaskGroup() as group:
                tasks = [group.create_task(contribute(role)) for role in SPECIALIST_ROLES]
            reports = tuple(task.result() for task in tasks)
            prompt = synthesis.build_prompt(included, len(included), context=_reports_context(reports),
                                            complete_abstracts=True)
            result = await _agent(active_model, opportunity_analysis_prompt(), specialist=False).run(
                prompt, deps=len(included), model_settings=settings,
                usage_limits=UsageLimits(request_limit=ANALYSIS_REQUEST_LIMIT), retries=retries)
            return ResearchResult(result.output, prompt, tuple(included), reports)
    except TimeoutError:
        raise StrataError(
            f"Research team exceeded the overall {analysis_timeout_s}s deadline; "
            "no partial report produced. Check STRATA_BASE_URL (legacy FREETOKEN_BASE_URL) or use fewer candidates."
        ) from None
    except (ConfigurationError, StrataError):
        raise
    except Exception as exc:
        # Never include upstream bodies, prompts, validation inputs or auth data.
        raise StrataError(
            f"Research team failed ({type(exc).__name__}); no partial report produced. "
            "Check the user-owned STRATA_BASE_URL (legacy FREETOKEN_BASE_URL)/model and typed response support."
        ) from None
    finally:
        if session is not None:
            try:
                await session.http_client.aclose()
            except Exception:
                pass


async def _research_pdf_async(
    candidates: list[CollectedWork], *, model, max_candidates: int,
    analysis_timeout_s: float, max_tokens: int, disable_thinking: bool,
    session, documents: Sequence[object], document_timeout_s: float,
    document_model=None,
    on_documents_read=None,
) -> ResearchResult:
    """PDF mode: read full texts first, then team over readings only."""
    from pydantic_ai.usage import UsageLimits

    from radar.agent import pdf_reading as _reading
    from radar.config.documents import DOCUMENT_MAX_TOKENS
    from radar.prompts.catalog import validate_pdf_prompts
    from radar.schema.documents import DocumentFailure

    validate_research_prompts()
    validate_pdf_prompts()
    reading_timeout = _reading.validate_document_timeout(document_timeout_s)
    settings, _, retries = synthesis.run_bounds(
        max_tokens=max_tokens, disable_thinking=disable_thinking,
        request_timeout_s=min(ANALYSIS_REQUEST_TIMEOUT_S, float(analysis_timeout_s)),
        analysis_timeout_s=analysis_timeout_s)
    active_model = session.model if session is not None else model
    if active_model is None:
        raise StrataError("Research team requires a supplied Strata model or session.")
    candidate_ids = {work.openalex_id for work in candidates}
    pre_failures: list[DocumentFailure] = []
    to_read: list[object] = []
    seen: set[str] = set()
    for doc in documents:
        work_id = str(getattr(doc, "work_id", ""))
        if not work_id or work_id in seen:
            pre_failures.append(DocumentFailure(
                work_id=work_id or "(missing)", category="duplicate"))
            continue
        seen.add(work_id)
        if work_id not in candidate_ids:
            pre_failures.append(DocumentFailure(work_id=work_id, category="unrelated"))
            continue
        to_read.append(doc)
    reading_tokens = min(int(max_tokens), DOCUMENT_MAX_TOKENS)
    reading_model = document_model if document_model is not None else active_model
    readings, read_failures = await _reading.read_documents_async(
        to_read, reading_model, max_tokens=reading_tokens,
        disable_thinking=disable_thinking, timeout_s=reading_timeout)
    all_failures = tuple(pre_failures) + tuple(read_failures)
    # Stage boundary: retain actual reading outcomes even if a later role fails.
    # Persistence remains the caller's responsibility; this module owns no DB.
    if on_documents_read is not None:
        on_documents_read(readings, all_failures)
    readings_by_id: dict[str, object] = {r.work_id: r for r in readings}
    eligible = [w for w in bound_candidates(candidates, max_candidates)
                if w.openalex_id in readings_by_id]
    if not eligible or not readings:
        return ResearchResult(
            RadarDraft(next_move="No PDF readings completed; no synthesis produced."),
            "", (), (), tuple(readings), tuple(all_failures))
    included = select_pdf_cohort(candidates, readings_by_id, max_candidates)
    if not included:
        return ResearchResult(
            RadarDraft(next_move="No PDF readings fit the research prompt budget."),
            "", (), (), tuple(readings), tuple(all_failures))
    async with asyncio.timeout(float(analysis_timeout_s)):
        semaphore = asyncio.Semaphore(SPECIALIST_CONCURRENCY)

        async def contribute(role: SpecialistRole) -> SpecialistContribution:
            async with semaphore:
                spec = specialist_prompt(role)
                prompt = synthesis.build_pdf_prompt(
                    included, readings_by_id, len(included), prompt_spec=spec)
                specialist_settings = dict(settings)
                specialist_settings["max_tokens"] = min(max_tokens, SPECIALIST_MAX_TOKENS)
                result = await _agent(active_model, spec, specialist=True).run(
                    prompt, deps=len(included), model_settings=specialist_settings,
                    usage_limits=UsageLimits(request_limit=ANALYSIS_REQUEST_LIMIT), retries=retries)
                return SpecialistContribution(role=role, draft=result.output)

        async with asyncio.TaskGroup() as group:
            tasks = [group.create_task(contribute(role)) for role in SPECIALIST_ROLES]
        reports = tuple(task.result() for task in tasks)
        prompt = synthesis.build_pdf_prompt(
            included, readings_by_id, len(included), context=_reports_context(reports))
        result = await _pdf_final_agent(
            active_model, opportunity_analysis_prompt(), included, readings_by_id).run(
            prompt, deps=len(included), model_settings=settings,
            usage_limits=UsageLimits(request_limit=ANALYSIS_REQUEST_LIMIT), retries=retries)
        return ResearchResult(result.output, prompt, tuple(included), reports,
                              tuple(readings), tuple(all_failures))


def research_candidates(candidates: list[CollectedWork], **kwargs) -> ResearchResult:
    return asyncio.run(research_candidates_async(candidates, **kwargs))
