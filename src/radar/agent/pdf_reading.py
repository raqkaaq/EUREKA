"""Full-PDF LLM reading: chunk, read, hierarchically reduce.

Code owns coverage: the full extracted text (with page markers) is split
contiguously into bounded chunks, every character and page is submitted to
real PydanticAI, and only complete readings become :class:`PDFReading`.
The model returns only :class:`DocumentNotes`; index/start/end/page
coverage is attached deterministically by code, never by the model.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from typing import TYPE_CHECKING

from radar.config.documents import (
    DOCUMENT_ANALYSIS_TIMEOUT_S,
    DOCUMENT_MAX_TOKENS,
    DOCUMENT_REDUCTION_GROUP_SIZE,
    validate_document_timeout,
)
from radar.prompts.catalog import (
    pdf_reading_prompt,
    pdf_reduction_prompt,
    validate_pdf_prompts,
)
from radar.processing.document_chunks import build_full_text, chunk_document
from radar.schema.documents import (
    ChunkReading,
    DocumentFailure,
    DocumentNotes,
    PDFDocument,
    PDFReading,
)

if TYPE_CHECKING:
    from pydantic_ai.models import Model

PDF_PROMPT_BOUND_CHARS = 12_000
PDF_REQUEST_TIMEOUT_S = 60.0
PDF_REQUEST_LIMIT = 2
PDF_RETRIES = 1


def _normalize(text: str) -> str:
    return " ".join((text or "").split())


def validate_reading_tokens(value: int) -> int:
    from radar.config.runtime import validate_max_tokens

    tokens = validate_max_tokens(value)
    return min(tokens, DOCUMENT_MAX_TOKENS)


def _run_bounds(
    *,
    max_tokens: int,
    disable_thinking: bool,
    timeout_s: float,
) -> tuple[dict, object, int]:
    from pydantic_ai.usage import UsageLimits

    from radar.provider import strata as _strata

    tokens = validate_reading_tokens(max_tokens)
    overall = validate_document_timeout(timeout_s)
    per_request = min(PDF_REQUEST_TIMEOUT_S, overall)
    settings: dict = {"max_tokens": tokens, "timeout": per_request}
    if disable_thinking:
        settings["extra_body"] = _strata.thinking_extra_body()
    return settings, UsageLimits(request_limit=PDF_REQUEST_LIMIT), PDF_RETRIES


def build_chunk_prompt(chunk_text: str, pages: list[int], index: int) -> str:
    return (
        f"CHUNK {index} (pages {pages}; untrusted extracted text):\n"
        f"--- begin chunk {index} ---\n{chunk_text}\n--- end chunk {index} ---\n"
        "TASK: Return DocumentNotes for this chunk only. Cite only pages "
        f"{pages}. Each quote must appear verbatim in this chunk."
    )


def build_reduction_prompt(children: list[DocumentNotes]) -> str:
    payload = json.dumps([note.model_dump() for note in children],
                         ensure_ascii=False, separators=(",", ":"))
    return (
        "CHILD NOTES (untrusted model interpretations; quotes are source text):\n"
        f"--- begin child notes ---\n{payload}\n--- end child notes ---\n"
        "TASK: Combine ALL child notes above into one DocumentNotes. "
        "Every reduced evidence page+quote must match one supplied child evidence entry verbatim."
    )


def _chunk_agent(model: Model, instructions: str, chunk_text: str,
                 pages: list[int], page_texts: dict[int, str]):
    from pydantic_ai import Agent, ModelRetry

    normalized_chunk = _normalize(chunk_text)
    agent = Agent(model, output_type=DocumentNotes,
                  instructions=instructions, retries=PDF_RETRIES)

    @agent.output_validator
    def _validated(notes: DocumentNotes) -> DocumentNotes:
        for item in notes.evidence:
            if isinstance(item.page, bool) or not isinstance(item.page, int):
                raise ModelRetry("Evidence page must be an integer page number.")
            if not isinstance(item.quote, str):
                raise ModelRetry("Evidence quote must be a string.")
            if item.page not in pages:
                raise ModelRetry(
                    f"Page {item.page} is not in supplied chunk pages {pages}.")
            norm_quote = _normalize(item.quote)
            if not norm_quote or norm_quote not in normalized_chunk:
                raise ModelRetry("Quote must appear verbatim in the supplied chunk.")
            source = page_texts.get(item.page, "")
            if not norm_quote or norm_quote not in _normalize(source):
                raise ModelRetry("Quote must appear verbatim in the cited source page.")
        return notes

    return agent


def _reduction_agent(model: Model, instructions: str,
                     allowed: set[tuple[int, str]]):
    from pydantic_ai import Agent, ModelRetry

    agent = Agent(model, output_type=DocumentNotes,
                  instructions=instructions, retries=PDF_RETRIES)

    @agent.output_validator
    def _validated(notes: DocumentNotes) -> DocumentNotes:
        for item in notes.evidence:
            if isinstance(item.page, bool) or not isinstance(item.page, int):
                raise ModelRetry("Evidence page must be an integer page number.")
            if (item.page, item.quote) not in allowed:
                raise ModelRetry(
                    "Reduced evidence must come verbatim from supplied child evidence.")
        return notes

    return agent


def _failure_category(exc: BaseException) -> str:
    from radar.config.yaml import ConfigurationError
    from radar.provider.strata import StrataError

    if isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
        return "deadline"
    if isinstance(exc, ConfigurationError):
        return "configuration"
    if isinstance(exc, StrataError):
        return "inference"
    if isinstance(exc, ValueError):
        return "invalid"
    return "reading_failed"


async def read_pdf_async(
    document: PDFDocument,
    model: Model,
    *,
    max_tokens: int = DOCUMENT_MAX_TOKENS,
    disable_thinking: bool = False,
    timeout_s: float = DOCUMENT_ANALYSIS_TIMEOUT_S,
) -> PDFReading:
    """Read one full PDF through chunk agents plus hierarchical reduction."""
    from radar.config.yaml import ConfigurationError
    from radar.provider.strata import StrataError

    validate_pdf_prompts()
    settings, limits, retries = _run_bounds(
        max_tokens=max_tokens, disable_thinking=disable_thinking, timeout_s=timeout_s)
    overall = validate_document_timeout(timeout_s)
    if model is None:
        raise StrataError("PDF reading requires a supplied model.")
    try:
        full_text, chunk_spans = chunk_document(document)
        page_texts = {page.number: page.text for page in document.pages}
        reader_spec = pdf_reading_prompt()
        reducer_spec = pdf_reduction_prompt()
        async with asyncio.timeout(overall):
            chunk_notes: list[DocumentNotes] = []
            chunk_models: list[ChunkReading] = []
            for index, (start, end) in enumerate(
                    [(s, e) for s, e, _ in chunk_spans]):
                pages = chunk_spans[index][2]
                chunk_text = full_text[start:end]
                prompt = build_chunk_prompt(chunk_text, pages, index)
                if len(reader_spec.instructions) + len(prompt) > PDF_PROMPT_BOUND_CHARS:
                    raise StrataError("PDF chunk prompt exceeded its bound.")
                agent = _chunk_agent(model, reader_spec.instructions,
                                     chunk_text, pages, page_texts)
                result = await agent.run(
                    prompt, model_settings=settings,  # type: ignore[arg-type]
                    usage_limits=limits, retries=retries)
                notes = result.output
                if not isinstance(notes, DocumentNotes):
                    notes = DocumentNotes.model_validate(
                        notes.model_dump() if hasattr(notes, "model_dump") else notes)
                chunk_models.append(ChunkReading(
                    index=index, start=start, end=end,
                    pages=pages, notes=notes))
                chunk_notes.append(notes)
            final_notes = await _reduce_all_async(
                chunk_notes, model, reducer_spec.instructions, settings, limits, retries)
            return PDFReading(
                work_id=document.work_id, source_url=document.source_url,
                sha256=document.sha256, relative_path=document.relative_path,
                page_count=len(document.pages), text_chars=len(full_text),
                chunks=chunk_models, notes=final_notes,
                evidence_level="pdf_text",
                extraction_warning=document.extraction_warning)
    except (ConfigurationError, StrataError):
        raise
    except TimeoutError as exc:
        raise StrataError(
            f"PDF reading exceeded the overall {overall}s deadline; no partial reading produced."
        ) from exc
    except Exception as exc:
        from radar.provider.strata import StrataError as _SE

        raise _SE(f"PDF reading failed ({type(exc).__name__}); no partial reading produced.") from exc


async def _reduce_all_async(
    notes: list[DocumentNotes],
    model: Model,
    instructions: str,
    settings: dict,
    limits: object,
    retries: int,
) -> DocumentNotes:
    from radar.provider.strata import StrataError

    if not notes:
        raise ValueError("no chunk notes to reduce")
    current = list(notes)
    while len(current) > 1:
        nxt: list[DocumentNotes] = []
        for offset in range(0, len(current), DOCUMENT_REDUCTION_GROUP_SIZE):
            group = current[offset:offset + DOCUMENT_REDUCTION_GROUP_SIZE]
            if len(group) == 1:
                nxt.append(group[0])
                continue
            prompt = build_reduction_prompt(group)
            if len(instructions) + len(prompt) > PDF_PROMPT_BOUND_CHARS:
                raise StrataError("PDF reduction prompt exceeded its bound.")
            allowed = {(item.page, item.quote)
                       for note in group for item in note.evidence}
            agent = _reduction_agent(model, instructions, allowed)
            result = await agent.run(
                prompt, model_settings=settings,  # type: ignore[arg-type]
                usage_limits=limits, retries=retries)  # type: ignore[arg-type]
            reduced = result.output
            if not isinstance(reduced, DocumentNotes):
                reduced = DocumentNotes.model_validate(
                    reduced.model_dump() if hasattr(reduced, "model_dump") else reduced)
            nxt.append(reduced)
        current = nxt
    return current[0]


async def read_documents_async(
    documents: Sequence[PDFDocument],
    model: Model,
    *,
    max_tokens: int = DOCUMENT_MAX_TOKENS,
    disable_thinking: bool = False,
    timeout_s: float = DOCUMENT_ANALYSIS_TIMEOUT_S,
) -> tuple[list[PDFReading], list[DocumentFailure]]:
    """Read papers sequentially within one global deadline; never partial."""
    overall = validate_document_timeout(timeout_s)
    if model is None:
        from radar.provider.strata import StrataError

        raise StrataError("PDF reading requires a supplied model.")
    readings: list[PDFReading] = []
    failures: list[DocumentFailure] = []
    seen: set[str] = set()
    pending = list(documents)
    try:
        async with asyncio.timeout(overall):
            for document in pending:
                if document.work_id in seen:
                    failures.append(DocumentFailure(
                        work_id=document.work_id, category="duplicate"))
                    continue
                seen.add(document.work_id)
                try:
                    reading = await read_pdf_async(
                        document, model, max_tokens=max_tokens,
                        disable_thinking=disable_thinking, timeout_s=overall)
                except Exception as exc:
                    failures.append(DocumentFailure(
                        work_id=document.work_id,
                        category=_failure_category(exc)[:100] or "reading_failed"))
                    continue
                readings.append(reading)
    except TimeoutError:
        done = {r.work_id for r in readings} | {f.work_id for f in failures}
        for document in pending:
            if document.work_id not in done:
                failures.append(DocumentFailure(
                    work_id=document.work_id, category="deadline"))
    return readings, failures


def read_pdf(
    document: PDFDocument,
    model: Model,
    **kwargs,
) -> PDFReading:
    return asyncio.run(read_pdf_async(document, model, **kwargs))


def read_documents(
    documents: Sequence[PDFDocument],
    model: Model,
    **kwargs,
) -> tuple[list[PDFReading], list[DocumentFailure]]:
    return asyncio.run(read_documents_async(documents, model, **kwargs))


__all__ = [
    "build_chunk_prompt",
    "build_full_text",
    "build_reduction_prompt",
    "chunk_document",
    "read_documents",
    "read_documents_async",
    "read_pdf",
    "read_pdf_async",
    "validate_document_timeout",
    "validate_reading_tokens",
]
