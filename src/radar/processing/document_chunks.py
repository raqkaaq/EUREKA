"""Lossless, page-preserving spans shared by investigation and provenance checks."""

from radar.config.documents import (
    DOCUMENT_CHUNK_CHARS, DOCUMENT_CHUNK_EXCERPT_CHARS, DOCUMENT_MAX_CHUNKS, DOCUMENT_MAX_EXCERPTS,
)
from radar.schema.documents import DocumentNotes, EvidenceExcerpt, PDFDocument


def build_full_text(document: PDFDocument) -> tuple[str, list[tuple[int, int, int]]]:
    parts: list[str] = []
    segments: list[tuple[int, int, int]] = []
    offset = 0
    for page in document.pages:
        segment = f"\n[page {page.number}]\n{page.text}"
        parts.append(segment)
        segments.append((page.number, offset, offset + len(segment)))
        offset += len(segment)
    return "".join(parts), segments


def chunk_document(document: PDFDocument) -> tuple[str, list[tuple[int, int, list[int]]]]:
    text, segments = build_full_text(document)
    spans = []
    for start in range(0, len(text), DOCUMENT_CHUNK_CHARS):
        end = min(start + DOCUMENT_CHUNK_CHARS, len(text))
        pages = [number for number, left, right in segments if left < end and right > start]
        spans.append((start, end, pages))
    if len(spans) > DOCUMENT_MAX_CHUNKS:
        raise ValueError("document requires more chunks than its bound allows")
    return text, spans


def chunk_excerpts(document: PDFDocument, start: int, end: int) -> list[EvidenceExcerpt]:
    """Sample deterministic anchors from real page text inside one chunk.

    This registry is not the reading input: every character still travels in
    the complete chunk. When possible, each intersecting nonblank page gets
    an anchor; larger page sets are sampled evenly, including the last page.
    """
    text, segments = build_full_text(document)
    if not 0 <= start < end <= len(text):
        raise ValueError("Excerpt span must be within the full document text.")
    fragments: list[tuple[int, str]] = []
    for page, (_, left, right) in zip(document.pages, segments, strict=True):
        body_left = left + len(f"\n[page {page.number}]\n")
        fragment = text[max(start, body_left):min(end, right)] if end > body_left and start < right else ""
        if fragment.strip():
            fragments.append((page.number, fragment))
    if len(fragments) > DOCUMENT_MAX_EXCERPTS:
        fragments = [fragments[i * (len(fragments) - 1) // (DOCUMENT_MAX_EXCERPTS - 1)]
                     for i in range(DOCUMENT_MAX_EXCERPTS)]
    excerpts: list[EvidenceExcerpt] = []
    for index, (page, fragment) in enumerate(fragments):
        quota = DOCUMENT_MAX_EXCERPTS // len(fragments) + (index < DOCUMENT_MAX_EXCERPTS % len(fragments))
        quotes = list(dict.fromkeys(fragment[offset:offset + DOCUMENT_CHUNK_EXCERPT_CHARS].strip()
                                   for offset in range(0, len(fragment), DOCUMENT_CHUNK_EXCERPT_CHARS)))
        quotes = [quote for quote in quotes if quote]
        count = min(quota, len(quotes))
        for i in range(count):
            quote = quotes[i * (len(quotes) - 1) // (count - 1)] if count > 1 else quotes[0]
            excerpts.append(EvidenceExcerpt(excerpt_id=len(excerpts), page=page, quote=quote))
    return excerpts


def reduction_excerpts(notes: list[DocumentNotes]) -> list[EvidenceExcerpt]:
    """Registry of existing verified page/quote pairs, in child encounter order."""
    pairs = dict.fromkeys((item.page, item.quote) for note in notes for item in note.evidence)
    if len(pairs) > DOCUMENT_MAX_EXCERPTS:
        raise ValueError("Reduction evidence registry exceeds its bound.")
    return [EvidenceExcerpt(excerpt_id=index, page=page, quote=quote)
            for index, (page, quote) in enumerate(pairs)]
