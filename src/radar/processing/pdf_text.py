"""Isolated PDF text extraction with hard bounds.

Public seam: :func:`extract_pdf`, which runs the real pypdf parser in an
isolated child (``python -B -m radar.processing.pdf_text``) communicating
through pipes only. The child sets its own memory/CPU limits before the
parser is imported, prints one bounded JSON document on stdout, and writes
no sidecar, temp, or runtime files. The parent caps child output, enforces
timeouts, revalidates every bound, and never truncates: oversize documents
are refused rather than shortened.

Page numbering follows the original PDF (1-based, contiguous). Documents
with encryption, no extractable text layer, or empty pages fail with a
dedicated category so callers record them as unreadable/scanned instead of
pretending the extraction is complete.
"""

from __future__ import annotations

import json as _json
import os as _os
import sys as _sys
from pathlib import Path as _Path

from radar.config.documents import (
    PDF_EXTRACTION_MEMORY_BYTES,
    PDF_EXTRACTION_TIMEOUT_S,
    PDF_MAX_BYTES,
    PDF_MAX_PAGES,
    PDF_MAX_TEXT_CHARS,
)
from radar.schema.documents import PDFPage as _PDFPage
from radar.processing.isolated import IsolatedProcessError, run_bounded

# Parent-side cap on child stdout: the child already bounds pages/chars, so
# legitimate output stays far below this; anything larger is a misbehaving
# child and is refused.
_CHILD_OUTPUT_LIMIT = 2 * 1024 * 1024


class PDFExtractionError(RuntimeError):
    """Standalone extraction failure with a stable machine category.

    Categories: ``encrypted_pdf``, ``scanned_pdf``, ``invalid_pdf``,
    ``too_many_pages``, ``too_much_text``, ``too_large``,
    ``extraction_timeout``, ``extraction_failed``. Messages are generic and
    never carry document content, URLs, or credentials. The acquisition
    layer imports and reclassifies this error into its own safe error.
    """

    def __init__(self, category: str, message: str):
        super().__init__(message)
        self.category = category


def extract_pdf(path: Path | str) -> list[_PDFPage]:
    """Extract every page's text layer from a stored PDF file.

    Runs the parser in an isolated child process (resource-limited, no
    files written) and returns one :class:`PDFPage` per original PDF page,
    numbered contiguously from 1. Raises :class:`PDFExtractionError` with a
    safe category on any bound violation or parse failure; never truncates.
    """
    target = _Path(path) if not isinstance(path, _Path) else path
    try:
        if (
            not target.is_file()
            or target.is_symlink()
            or target.stat().st_size == 0
            or target.stat().st_size > PDF_MAX_BYTES
        ):
            raise PDFExtractionError("too_large" if (
                target.exists() and not target.is_symlink()
                and target.is_file() and target.stat().st_size > PDF_MAX_BYTES
            ) else "invalid_pdf", "PDF file is not a readable bounded document.")
    except PDFExtractionError:
        raise
    except OSError:
        raise PDFExtractionError("invalid_pdf", "PDF file is not readable.") from None

    argv = [_sys.executable, "-B", "-m", "radar.processing.pdf_text", str(target)]
    try:
        try:
            code, raw = run_bounded(argv, timeout_s=PDF_EXTRACTION_TIMEOUT_S,
                                    max_output_bytes=_CHILD_OUTPUT_LIMIT)
        except IsolatedProcessError as exc:
            category = "extraction_timeout" if exc.category == "timeout" else "extraction_failed"
            raise PDFExtractionError(category, "PDF extraction worker exceeded a bound or failed.") from None
        if not raw:
            raise PDFExtractionError(
                "extraction_failed", "PDF extraction worker did not complete."
            )
        if len(raw) > _CHILD_OUTPUT_LIMIT:
            raise PDFExtractionError(
                "extraction_failed", "PDF extraction output exceeded its bound."
            )
        try:
            payload = _json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeError):
            raise PDFExtractionError(
                "extraction_failed", "PDF extraction output was not valid."
            ) from None
        if not isinstance(payload, dict):
            raise PDFExtractionError(
                "extraction_failed", "PDF extraction output was not valid."
            )
        if payload.get("ok") is not True:
            err = payload.get("error")
            category = err.get("category") if isinstance(err, dict) else None
            if not isinstance(category, str) or not category:
                category = "extraction_failed"
            raise PDFExtractionError(category, "PDF text extraction failed.")
        if code != 0:
            raise PDFExtractionError(
                "extraction_failed", "PDF extraction worker did not complete."
            )
        entries = payload.get("pages")
        if not isinstance(entries, list) or not entries:
            raise PDFExtractionError(
                "scanned_pdf", "PDF has no extractable text layer."
            )
        pages: list[_PDFPage] = []
        for index, entry in enumerate(entries, start=1):
            text = entry.get("text") if isinstance(entry, dict) else None
            if not isinstance(text, str) or not text.strip():
                raise PDFExtractionError(
                    "scanned_pdf", "PDF has no extractable text layer."
                )
            try:
                pages.append(_PDFPage(number=index, text=text))
            except ValueError:
                raise PDFExtractionError(
                    "too_much_text", "PDF text exceeds its bound."
                ) from None
        if [p.number for p in pages] != list(range(1, len(pages) + 1)):
            raise PDFExtractionError(
                "invalid_pdf", "PDF page numbering is not complete."
            )
        if sum(len(p.text) for p in pages) > PDF_MAX_TEXT_CHARS:
            raise PDFExtractionError(
                "too_much_text", "PDF text exceeds its bound."
            )
        return pages
    except PDFExtractionError:
        raise


def _child_main(argv: list[str]) -> int:
    """Child entry point: limit the process, extract, print bounded JSON."""
    # Hardening first: contain a hostile PDF before the parser is imported.
    try:
        import resource as _resource

        _mem = int(PDF_EXTRACTION_MEMORY_BYTES)
        _resource.setrlimit(_resource.RLIMIT_AS, (_mem, _mem))
        _cpu = int(PDF_EXTRACTION_TIMEOUT_S)
        _resource.setrlimit(_resource.RLIMIT_CPU, (_cpu, _cpu))
    except Exception:
        _sys.stdout.buffer.write(b'{"ok":false,"error":{"category":"extraction_failed"}}')
        _sys.stdout.buffer.flush()
        return 1

    def _fail(category: str) -> int:
        _sys.stdout.buffer.write(
            _json.dumps(
                {"ok": False, "error": {"category": category}}, ensure_ascii=False
            ).encode("utf-8")
        )
        _sys.stdout.buffer.flush()
        return 1

    if len(argv) != 2:
        return _fail("invalid_pdf")
    try:
        from pypdf import PdfReader as _PdfReader
    except Exception:
        return _fail("extraction_failed")
    try:
        reader = _PdfReader(argv[1])
        if getattr(reader, "is_encrypted", False):
            return _fail("encrypted_pdf")
        count = len(reader.pages)
        if count < 1:
            return _fail("invalid_pdf")
        if count > PDF_MAX_PAGES:
            return _fail("too_many_pages")
        out: list[dict[str, object]] = []
        total = 0
        for number in range(1, count + 1):
            try:
                text = reader.pages[number - 1].extract_text() or ""
            except Exception:
                return _fail("invalid_pdf")
            if not text.strip():
                # An unreadable page must fail the document, never be
                # silently dropped (that would fake complete coverage).
                return _fail("scanned_pdf")
            if len(text) > PDF_MAX_TEXT_CHARS:
                return _fail("too_much_text")
            total += len(text)
            if total > PDF_MAX_TEXT_CHARS:
                return _fail("too_much_text")
            out.append({"number": number, "text": text})
        _sys.stdout.buffer.write(
            _json.dumps({"ok": True, "pages": out}, ensure_ascii=False).encode("utf-8")
        )
        _sys.stdout.buffer.flush()
        return 0
    except PDFExtractionError:
        return _fail("extraction_failed")
    except Exception:
        return _fail("invalid_pdf")


if __name__ == "__main__":  # pragma: no cover - exercised via extract_pdf
    _os._exit(_child_main(_sys.argv))
