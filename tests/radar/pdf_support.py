"""Shared PDF test doubles at the AI/filesystem boundary (tests only).

Loader replaces ``source.pdf.acquire_pdf`` with an explicit callable
``(work, directory, cached)``. Reading model returns real
``DocumentNotes`` through PydanticAI ``FunctionModel``. No network,
no filesystem writes, no production imports.
"""

from __future__ import annotations

from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models.function import FunctionModel

from radar.schema.documents import (
    DocumentNotes,
    PageEvidence,
    PDFDocument,
    PDFPage,
)

SHA = "b" * 64


def make_pdf_document(work_id="https://openalex.org/W0",
                      texts=("hello world paper text",)) -> PDFDocument:
    digits = "".join(ch for ch in work_id if ch.isdigit()) or "0"
    pages = [PDFPage(number=i + 1, text=text) for i, text in enumerate(texts)]
    return PDFDocument(
        work_id=work_id, source_url="https://example.org/paper.pdf",
        sha256=SHA, relative_path=f"pdf/W{digits}-{SHA}.pdf", pages=pages)


def make_loader(docs_by_id, failures_by_id=None, calls=None):
    """Explicit (work, directory, cached) double for ``acquire_pdf``."""
    from radar.source.pdf import PDFError

    failures_by_id = failures_by_id or {}

    def _load(work, directory, cached):
        if calls is not None:
            calls.append((work.openalex_id, str(directory)))
        if work.openalex_id in failures_by_id:
            raise PDFError(failures_by_id[work.openalex_id], "fixture PDF failure.")
        doc = docs_by_id.get(work.openalex_id)
        if doc is None:
            raise PDFError("no_pdf_url", "No PDF link available from OpenAlex locations.")
        return doc

    return _load


def make_any_loader(calls=None):
    """Loader that synthesizes a tiny PDF for any requested work."""

    def _load(work, directory, cached):
        if calls is not None:
            calls.append((work.openalex_id, str(directory)))
        return make_pdf_document(work.openalex_id)

    return _load


def valid_notes(quote="hello", page=1) -> DocumentNotes:
    return DocumentNotes(
        summary="summary of findings", methods="methods used",
        results="results observed", limitations="limits noted",
        evidence=[PageEvidence(page=page, quote=quote, finding="supports claim")])


def reading_model(quote="hello", page=1) -> FunctionModel:
    """DocumentNotes model for the PDF reader seam (chunk + reduction)."""

    def _impl(messages, info):
        return ModelResponse(parts=[ToolCallPart(
            info.output_tools[0].name, valid_notes(quote=quote, page=page).model_dump())])

    return FunctionModel(_impl)


def empty_notes_model() -> FunctionModel:
    """Reader double with no evidence (always passes chunk validation)."""

    def _impl(messages, info):
        notes = DocumentNotes(
            summary="summary of findings", methods="methods used",
            results="results observed", limitations="limits noted")
        return ModelResponse(parts=[ToolCallPart(
            info.output_tools[0].name, notes.model_dump())])

    return FunctionModel(_impl)


def fixed_draft_model(next_move="done") -> FunctionModel:
    from radar.schema.opportunities import RadarDraft

    fixed = RadarDraft(opportunities=[], ignore=[], next_move=next_move)

    def _impl(messages, info):
        return ModelResponse(parts=[ToolCallPart(
            tool_name="final_result", args=fixed.model_dump())])

    return FunctionModel(_impl)
