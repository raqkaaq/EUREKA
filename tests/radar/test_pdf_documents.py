"""PDF acquisition + extraction seams only (issue 59).

Test-first coverage for the two public seams owned by this worker:

- :func:`radar.processing.pdf_text.extract_pdf`
- :func:`radar.source.pdf.acquire_pdf`

Real PDF bytes are generated with pypdf (simple text content streams with
a standard font); extraction itself is never mocked. HTTP is exercised only
through :class:`httpx.MockTransport` as the real transport boundary, with an
injected resolver, so no network, live hosts, or environment secrets are
ever touched. Scratch files live under the workspace runtime dir
(``TMPDIR`` when it points outside ``/tmp``, else
``.venv/radar-test-runtime``) in deterministic per-test directories that are
always removed.
"""

from __future__ import annotations

import hashlib
import asyncio
import io
import os
import shutil
import unittest
import sys
from unittest import mock
from pathlib import Path

import httpx

from radar.config.documents import PDF_MAX_BYTES
from radar.processing.pdf_text import PDFExtractionError, extract_pdf
from radar.schema.documents import PDFDocument
from radar.schema.papers import CollectedWork, LocationInfo
from radar.source.pdf import PDFError, acquire_pdf

REPO_ROOT = Path(__file__).resolve().parents[2]
_WORKSPACE_RUNTIME = REPO_ROOT / ".venv" / "radar-test-runtime"

# Documentation-safe stand-in for "a public host": globally routable, but no
# packet ever leaves the process because MockTransport answers everything.
_PUBLIC_IP = "93.184.216.34"
_HTML_BODY = b"<html><body>Not a pdf.</body></html>"


# ---------------------------------------------------------------------------
# Helpers: real PDF bytes, works, transports, scratch dirs
# ---------------------------------------------------------------------------


def _pdf_bytes(texts: list[str], *, password: str | None = None) -> bytes:
    """Generate a real multi-page PDF with pypdf (standard Helvetica font)."""
    from pypdf import PdfWriter
    from pypdf.generic import (
        DecodedStreamObject,
        DictionaryObject,
        NameObject,
    )

    writer = PdfWriter()
    for text in texts:
        page = writer.add_blank_page(width=612, height=792)
        font = DictionaryObject()
        font[NameObject("/Type")] = NameObject("/Font")
        font[NameObject("/Subtype")] = NameObject("/Type1")
        font[NameObject("/BaseFont")] = NameObject("/Helvetica")
        fonts = DictionaryObject()
        fonts[NameObject("/F1")] = writer._add_object(font)
        resources = DictionaryObject()
        resources[NameObject("/Font")] = fonts
        page[NameObject("/Resources")] = resources
        escaped = (
            text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        )
        stream = DecodedStreamObject()
        stream.set_data(
            f"BT /F1 24 Tf 72 720 Td ({escaped}) Tj ET".encode("latin-1")
        )
        page[NameObject("/Contents")] = writer._add_object(stream)
    if password is not None:
        writer.encrypt(password)
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


def _blank_pdf_bytes(pages: int = 2) -> bytes:
    """Real PDF with pages but no text layer (scanned-document stand-in)."""
    from pypdf import PdfWriter

    writer = PdfWriter()
    for _ in range(pages):
        writer.add_blank_page(width=612, height=792)
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


def _work(*pdf_urls: str, wid: str = "https://openalex.org/W123") -> CollectedWork:
    return CollectedWork(
        openalex_id=wid,
        title="Test work",
        locations=[LocationInfo(pdf_url=url) for url in pdf_urls],
    )


def _resolver(mapping: dict[str, list[str]]):
    def resolve(host: str) -> list[str]:
        return list(mapping[host])

    return resolve


def _ok_handler(body: bytes, seen: dict | None = None):
    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen["url_host"] = request.url.host
            seen["host_header"] = request.headers.get("host")
            seen["sni"] = (request.extensions or {}).get("sni_hostname")
            seen["count"] = int(seen.get("count", 0)) + 1
        return httpx.Response(
            200,
            headers={"content-type": "application/pdf"},
            content=body,
        )

    return handler


class _ScratchTest(unittest.TestCase):
    """Deterministic scratch dir per test; always cleaned up."""

    def scratch(self) -> Path:
        env_tmp = os.environ.get("TMPDIR", "")
        base = Path(env_tmp) if env_tmp else _WORKSPACE_RUNTIME
        try:
            # Acquisition rightly refuses /tmp roots; never test there.
            if base.resolve() == Path("/tmp") or Path("/tmp") in base.resolve().parents:
                base = _WORKSPACE_RUNTIME
        except OSError:
            base = _WORKSPACE_RUNTIME
        path = base / f"pdfdocs-{self.__class__.__name__}-{self._testMethodName}"
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)
        path.mkdir(parents=True, exist_ok=True)
        self.addCleanup(shutil.rmtree, path, True)
        return path


# ---------------------------------------------------------------------------
# Extraction seam: radar.processing.pdf_text.extract_pdf
# ---------------------------------------------------------------------------


class TestExtractPdf(_ScratchTest):
    def test_roundtrip_preserves_all_pages_in_order(self):
        directory = self.scratch()
        target = directory / "doc.pdf"
        target.write_bytes(
            _pdf_bytes(["alpha first page", "beta second page", "gamma late page"])
        )
        pages = extract_pdf(target)
        self.assertEqual([p.number for p in pages], [1, 2, 3])
        self.assertIn("alpha first page", pages[0].text)
        self.assertIn("beta second page", pages[1].text)
        # Late pages must survive: no truncation of the document tail.
        self.assertIn("gamma late page", pages[2].text)

    def test_encrypted_pdf_is_refused(self):
        directory = self.scratch()
        target = directory / "enc.pdf"
        target.write_bytes(_pdf_bytes(["secret text"], password="userpw"))
        with self.assertRaises(PDFExtractionError) as ctx:
            extract_pdf(target)
        self.assertEqual(ctx.exception.category, "encrypted_pdf")

    def test_scanned_pdf_without_text_layer_is_recorded(self):
        directory = self.scratch()
        target = directory / "scan.pdf"
        target.write_bytes(_blank_pdf_bytes(2))
        with self.assertRaises(PDFExtractionError) as ctx:
            extract_pdf(target)
        self.assertEqual(ctx.exception.category, "scanned_pdf")

    def test_oversize_page_count_is_refused_not_truncated(self):
        directory = self.scratch()
        target = directory / "big.pdf"
        target.write_bytes(_pdf_bytes([f"page {i} content words" for i in range(81)]))
        with self.assertRaises(PDFExtractionError) as ctx:
            extract_pdf(target)
        self.assertEqual(ctx.exception.category, "too_many_pages")

    def test_missing_file_is_refused(self):
        with self.assertRaises(PDFExtractionError):
            extract_pdf(self.scratch() / "absent.pdf")

    def test_extraction_leaves_no_sidecar_files(self):
        directory = self.scratch()
        target = directory / "doc.pdf"
        target.write_bytes(_pdf_bytes(["plain text here"]))
        before = sorted(p.name for p in directory.iterdir())
        extract_pdf(target)
        after = sorted(p.name for p in directory.iterdir())
        self.assertEqual(before, after)
        self.assertEqual(after, ["doc.pdf"])


# ---------------------------------------------------------------------------
# Acquisition seam: radar.source.pdf.acquire_pdf
# ---------------------------------------------------------------------------


class TestAcquirePdf(_ScratchTest):
    def test_drip_stream_cannot_extend_absolute_download_deadline(self):
        class Drip(httpx.AsyncByteStream):
            async def __aiter__(self):
                yield b"%PDF-1.7\n"
                while True:
                    await asyncio.sleep(0.01)
                    yield b"x"

        async def handler(request):
            return httpx.Response(200, stream=Drip())

        directory = self.scratch()
        with mock.patch("radar.source.pdf.PDF_DOWNLOAD_TIMEOUT_S", 0.1):
            with self.assertRaises(PDFError) as caught:
                acquire_pdf(_work("https://pdf.example.org/drip.pdf"), directory,
                            transport=httpx.MockTransport(handler), resolver=lambda _: [_PUBLIC_IP])
        self.assertEqual(caught.exception.category, "download_timeout")
        self.assertEqual(list(directory.iterdir()), [])

    def test_isolated_child_output_and_time_are_bounded(self):
        from radar.processing.isolated import IsolatedProcessError, run_bounded
        with self.assertRaises(IsolatedProcessError) as caught:
            run_bounded([sys.executable, "-B", "-c", "print('x'*20000)"],
                        timeout_s=2, max_output_bytes=100)
        self.assertEqual(caught.exception.category, "output_limit")
        with self.assertRaises(IsolatedProcessError) as caught:
            run_bounded([sys.executable, "-B", "-c", "import time; time.sleep(5)"],
                        timeout_s=0.1, max_output_bytes=100)
        self.assertEqual(caught.exception.category, "timeout")

    def test_cached_page_text_cannot_override_immutable_pdf_bytes(self):
        from radar.schema.documents import PDFPage
        directory = self.scratch()
        work = _work("https://pdf.example.org/cache.pdf")
        doc = acquire_pdf(work, directory,
                          transport=httpx.MockTransport(_ok_handler(_pdf_bytes(["Actual negative result."]))),
                          resolver=lambda _: [_PUBLIC_IP])
        forged = doc.model_copy(update={"pages": [PDFPage(number=1, text="Invented positive result.")]})
        with self.assertRaises(PDFError) as caught:
            acquire_pdf(work, directory, cached=forged)
        self.assertEqual(caught.exception.category, "cache_mismatch")

    def test_acquire_ok_layout_and_reuse(self):
        directory = self.scratch()
        body = _pdf_bytes(["alpha first page", "beta second page", "gamma late page"])
        digest = hashlib.sha256(body).hexdigest()
        seen: dict = {}
        transport = httpx.MockTransport(_ok_handler(body, seen))
        work = _work("https://pdf.example.org/paper.pdf")
        resolver = _resolver({"pdf.example.org": [_PUBLIC_IP]})

        doc = acquire_pdf(work, directory, transport=transport, resolver=resolver)

        self.assertIsInstance(doc, PDFDocument)
        self.assertEqual(doc.work_id, "https://openalex.org/W123")
        self.assertEqual(doc.source_url, "https://pdf.example.org/paper.pdf")
        self.assertEqual(doc.sha256, digest)
        self.assertEqual(doc.relative_path, f"pdf/W123-{digest}.pdf")
        self.assertEqual([p.number for p in doc.pages], [1, 2, 3])
        self.assertIn("gamma late page", doc.pages[2].text)
        self.assertIn("Text-layer", doc.extraction_warning)
        stored = directory / doc.relative_path
        self.assertTrue(stored.is_file())
        self.assertEqual(hashlib.sha256(stored.read_bytes()).hexdigest(), digest)
        # DNS-rebinding mitigation: TCP goes to the validated IP while the
        # original host travels in Host and the TLS SNI extension.
        self.assertEqual(seen["url_host"], _PUBLIC_IP)
        self.assertEqual(seen["host_header"], "pdf.example.org")
        self.assertEqual(seen["sni"], "pdf.example.org")
        # Only the final PDF path exists: no .part/.html/log/harness files.
        self.assertEqual(
            sorted(p.name for p in directory.iterdir()), ["pdf"]
        )
        self.assertEqual(
            sorted(p.name for p in (directory / "pdf").iterdir()),
            [f"W123-{digest}.pdf"],
        )

        # A second acquisition of identical bytes reuses the file cleanly.
        before = stored.read_bytes()
        doc2 = acquire_pdf(
            work,
            directory,
            transport=httpx.MockTransport(_ok_handler(body)),
            resolver=resolver,
        )
        self.assertEqual(doc2, doc)
        self.assertEqual(stored.read_bytes(), before)

    def test_cached_document_skips_network_after_validation(self):
        directory = self.scratch()
        body = _pdf_bytes(["cached content page"])
        work = _work("https://pdf.example.org/cached.pdf")
        resolver = _resolver({"pdf.example.org": [_PUBLIC_IP]})
        doc = acquire_pdf(
            work,
            directory,
            transport=httpx.MockTransport(_ok_handler(body)),
            resolver=resolver,
        )

        def _exploding(_request: httpx.Request) -> httpx.Response:
            raise AssertionError("network must not be used for a valid cache")

        again = acquire_pdf(
            work,
            directory,
            transport=httpx.MockTransport(_exploding),
            resolver=_resolver({}),
            cached=doc,
        )
        self.assertEqual(again, doc)

    def test_cached_hash_tampering_is_rejected(self):
        directory = self.scratch()
        body = _pdf_bytes(["cached content page"])
        work = _work("https://pdf.example.org/cached.pdf")
        resolver = _resolver({"pdf.example.org": [_PUBLIC_IP]})
        doc = acquire_pdf(
            work,
            directory,
            transport=httpx.MockTransport(_ok_handler(body)),
            resolver=resolver,
        )
        tampered = PDFDocument(
            work_id=doc.work_id,
            source_url=doc.source_url,
            sha256="0" * 64,
            relative_path=f"pdf/W123-{'0' * 64}.pdf",
            pages=list(doc.pages),
        )
        with self.assertRaises(PDFError) as ctx:
            acquire_pdf(work, directory, cached=tampered, resolver=resolver)
        self.assertEqual(ctx.exception.category, "cache_mismatch")

    def test_cached_file_bytes_tampering_is_rejected(self):
        directory = self.scratch()
        body = _pdf_bytes(["cached content page"])
        work = _work("https://pdf.example.org/cached.pdf")
        resolver = _resolver({"pdf.example.org": [_PUBLIC_IP]})
        doc = acquire_pdf(
            work,
            directory,
            transport=httpx.MockTransport(_ok_handler(body)),
            resolver=resolver,
        )
        (directory / doc.relative_path).write_bytes(b"tampered-by-test")
        with self.assertRaises(PDFError) as ctx:
            acquire_pdf(work, directory, cached=doc, resolver=resolver)
        self.assertEqual(ctx.exception.category, "cache_mismatch")

    def test_cached_wrong_source_or_work_is_rejected(self):
        directory = self.scratch()
        body = _pdf_bytes(["cached content page"])
        work = _work("https://pdf.example.org/cached.pdf")
        resolver = _resolver({"pdf.example.org": [_PUBLIC_IP]})
        doc = acquire_pdf(
            work,
            directory,
            transport=httpx.MockTransport(_ok_handler(body)),
            resolver=resolver,
        )
        foreign_source = PDFDocument(
            work_id=doc.work_id,
            source_url="https://other.example.org/x.pdf",
            sha256=doc.sha256,
            relative_path=doc.relative_path,
            pages=list(doc.pages),
        )
        with self.assertRaises(PDFError) as ctx:
            acquire_pdf(work, directory, cached=foreign_source, resolver=resolver)
        self.assertEqual(ctx.exception.category, "cache_mismatch")
        other_work = _work("https://pdf.example.org/cached.pdf",
                           wid="https://openalex.org/W999")
        with self.assertRaises(PDFError) as ctx:
            acquire_pdf(other_work, directory, cached=doc, resolver=resolver)
        self.assertEqual(ctx.exception.category, "cache_mismatch")

    def test_no_pdf_url(self):
        with self.assertRaises(PDFError) as ctx:
            acquire_pdf(_work(), self.scratch(), resolver=_resolver({}))
        self.assertEqual(ctx.exception.category, "no_pdf_url")

    def test_credentials_in_url_are_rejected_without_echo(self):
        secret = "s3cret-u5er"
        with self.assertRaises(PDFError) as ctx:
            acquire_pdf(
                _work(f"https://{secret}:p4ss@pdf.example.org/x.pdf"),
                self.scratch(),
                transport=httpx.MockTransport(_ok_handler(b"%PDF-1.4\n")),
                resolver=_resolver({"pdf.example.org": [_PUBLIC_IP]}),
            )
        self.assertEqual(ctx.exception.category, "invalid_url")
        self.assertNotIn(secret, str(ctx.exception))

    def test_query_secret_is_rejected_without_echo(self):
        secret = "tok-abc-123-xyz"
        with self.assertRaises(PDFError) as ctx:
            acquire_pdf(
                _work(f"https://pdf.example.org/x.pdf?token={secret}"),
                self.scratch(),
                transport=httpx.MockTransport(_ok_handler(b"%PDF-1.4\n")),
                resolver=_resolver({"pdf.example.org": [_PUBLIC_IP]}),
            )
        self.assertEqual(ctx.exception.category, "invalid_url")
        self.assertNotIn(secret, str(ctx.exception))

    def test_nondefault_port_and_private_literal_are_rejected(self):
        for url, category in [
            ("https://pdf.example.org:8443/x.pdf", "invalid_url"),
            ("http://10.0.0.5/x.pdf", "blocked_host"),
        ]:
            with self.subTest(url=url):
                with self.assertRaises(PDFError) as ctx:
                    acquire_pdf(
                        _work(url),
                        self.scratch(),
                        transport=httpx.MockTransport(_ok_handler(b"%PDF-1.4\n")),
                        resolver=_resolver({"pdf.example.org": [_PUBLIC_IP]}),
                    )
                self.assertEqual(ctx.exception.category, category)
                self.assertNotIn("10.0.0.5", str(ctx.exception))

    def test_redirect_to_lan_is_blocked_and_nothing_persists(self):
        directory = self.scratch()

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/start.pdf":
                return httpx.Response(
                    302, headers={"location": "http://lan.example.net/in.pdf"}
                )
            return httpx.Response(200, content=b"%PDF-1.4 LAN\n")

        with self.assertRaises(PDFError) as ctx:
            acquire_pdf(
                _work("https://pdf.example.org/start.pdf"),
                directory,
                transport=httpx.MockTransport(handler),
                resolver=_resolver(
                    {
                        "pdf.example.org": [_PUBLIC_IP],
                        "lan.example.net": ["192.168.0.9"],
                    }
                ),
            )
        self.assertEqual(ctx.exception.category, "blocked_host")
        self.assertNotIn("192.168", str(ctx.exception))
        leftovers = [p for p in directory.rglob("*")]
        self.assertEqual(leftovers, [])

    def test_redirect_chain_bound_is_enforced(self):
        seen = {"count": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["count"] += 1
            step = int(request.url.path[2:])
            return httpx.Response(
                302, headers={"location": f"https://pdf.example.org/r{step + 1}"}
            )

        with self.assertRaises(PDFError) as ctx:
            acquire_pdf(
                _work("https://pdf.example.org/r0"),
                self.scratch(),
                transport=httpx.MockTransport(handler),
                resolver=_resolver({"pdf.example.org": [_PUBLIC_IP]}),
            )
        self.assertEqual(ctx.exception.category, "redirect_limit")
        self.assertLessEqual(seen["count"], 6)

    def test_html_mime_and_bad_magic_are_refused_without_persist(self):
        for name, body, headers in [
            ("html", _HTML_BODY, {"content-type": "text/html"}),
            ("magic", b"definitely not a pdf document", {"content-type": "application/pdf"}),
        ]:
            with self.subTest(case=name):
                directory = self.scratch()

                def handler(_request: httpx.Request, _b=body, _h=headers) -> httpx.Response:
                    return httpx.Response(200, headers=dict(_h), content=_b)

                with self.assertRaises(PDFError) as ctx:
                    acquire_pdf(
                        _work("https://pdf.example.org/x.pdf"),
                        directory,
                        transport=httpx.MockTransport(handler),
                        resolver=_resolver({"pdf.example.org": [_PUBLIC_IP]}),
                    )
                self.assertEqual(ctx.exception.category, "invalid_pdf")
                self.assertEqual([p for p in directory.rglob("*")], [])

    def test_download_size_limit_is_enforced_without_persist(self):
        directory = self.scratch()
        body = b"%PDF-1.4\n" + b"0" * PDF_MAX_BYTES

        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200, headers={"content-type": "application/pdf"}, content=body
            )

        with self.assertRaises(PDFError) as ctx:
            acquire_pdf(
                _work("https://pdf.example.org/huge.pdf"),
                directory,
                transport=httpx.MockTransport(handler),
                resolver=_resolver({"pdf.example.org": [_PUBLIC_IP]}),
            )
        self.assertEqual(ctx.exception.category, "too_large")
        self.assertEqual([p for p in directory.rglob("*")], [])

    def test_http_error_carries_status_not_url(self):
        url = "https://pdf.example.org/missing.pdf"

        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(404, content=b"nope")

        with self.assertRaises(PDFError) as ctx:
            acquire_pdf(
                _work(url),
                self.scratch(),
                transport=httpx.MockTransport(handler),
                resolver=_resolver({"pdf.example.org": [_PUBLIC_IP]}),
            )
        self.assertEqual(ctx.exception.category, "http_error")
        self.assertIn("404", str(ctx.exception))
        self.assertNotIn("missing.pdf", str(ctx.exception))
        self.assertNotIn("pdf.example.org", str(ctx.exception))

    def test_extraction_failures_surface_with_category(self):
        cases = [
            ("enc", _pdf_bytes(["locked"], password="pw"), "encrypted_pdf"),
            ("scan", _blank_pdf_bytes(1), "scanned_pdf"),
            (
                "pages",
                _pdf_bytes([f"page {i} words" for i in range(81)]),
                "too_many_pages",
            ),
        ]
        for name, body, category in cases:
            with self.subTest(case=name):
                directory = self.scratch()

                def handler(_r: httpx.Request, _b=body) -> httpx.Response:
                    return httpx.Response(
                        200, headers={"content-type": "application/pdf"}, content=_b
                    )

                with self.assertRaises(PDFError) as ctx:
                    acquire_pdf(
                        _work("https://pdf.example.org/x.pdf"),
                        directory,
                        transport=httpx.MockTransport(handler),
                        resolver=_resolver({"pdf.example.org": [_PUBLIC_IP]}),
                    )
                self.assertEqual(ctx.exception.category, category)

    def test_tmp_and_symlink_roots_are_refused_before_network(self):
        work = _work("https://pdf.example.org/x.pdf")

        def _exploding(_request: httpx.Request) -> httpx.Response:
            raise AssertionError("network must not be used with a bad root")

        with self.assertRaises(PDFError) as ctx:
            acquire_pdf(
                work,
                "/tmp/radar-pdf-must-refuse",
                transport=httpx.MockTransport(_exploding),
                resolver=_resolver({}),
            )
        self.assertEqual(ctx.exception.category, "storage_error")

        directory = self.scratch()
        real = directory / "real"
        real.mkdir()
        link = directory / "linkroot"
        try:
            link.symlink_to(real, target_is_directory=True)
        except OSError:
            self.skipTest("symlinks unavailable")
        with self.assertRaises(PDFError) as ctx:
            acquire_pdf(
                work,
                link,
                transport=httpx.MockTransport(_exploding),
                resolver=_resolver({}),
            )
        self.assertEqual(ctx.exception.category, "storage_error")


if __name__ == "__main__":
    unittest.main()
