"""Shared PDF test doubles at the AI/filesystem boundary (tests only).

Loader replaces ``source.pdf.acquire_pdf`` with an explicit callable
``(work, directory, cached)``. Reading model chooses evidence references
from the supplied registry through PydanticAI ``FunctionModel``. No network,
no filesystem writes, no production imports.
"""

from __future__ import annotations

import json

from pydantic_ai.messages import ModelResponse, ToolCallPart, UserPromptPart
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
    """Choose the requested source excerpt at the chunk/reduction model seam."""

    def _impl(messages, info):
        marker = "--- begin evidence registry ---\n"
        prompts = [part.content for message in messages for part in message.parts
                   if isinstance(part, UserPromptPart) and isinstance(part.content, str)
                   and marker in part.content]
        if not prompts:
            raise AssertionError("PDF reader did not supply its source evidence registry")
        registry = json.loads(prompts[-1].split(marker, 1)[1].split(
            "\n--- end evidence registry ---", 1)[0])
        matches = [entry for entry in registry
                   if entry["page"] == page and quote in entry["quote"]]
        if not matches:
            raise AssertionError(
                f"Requested page {page} excerpt {quote!r} was absent from the evidence registry")
        excerpt_id = matches[0]["excerpt_id"]
        if type(excerpt_id) is not int:
            raise AssertionError("Registry excerpt IDs must be actual integers")
        payload = valid_notes().model_dump(exclude={"evidence"})
        payload["evidence"] = [{"excerpt_id": excerpt_id, "finding": "supports claim"}]
        return ModelResponse(parts=[ToolCallPart(
            info.output_tools[0].name, payload)])

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
