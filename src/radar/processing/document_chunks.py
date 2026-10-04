"""Lossless, page-preserving spans shared by investigation and provenance checks."""

from radar.config.documents import DOCUMENT_CHUNK_CHARS, DOCUMENT_MAX_CHUNKS
from radar.schema.documents import PDFDocument


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
