"""PDF acquisition from OpenAlex locations only (issue 59).

Public seam: :func:`acquire_pdf`.

Only ``work.locations[].pdf_url`` values supplied by OpenAlex are ever
requested; landing pages are never scraped and no new APIs are added. Every
candidate link is validated (http/https, no credentials, no control
characters, no secret-looking query parameters, ports 80/443 only) and its
host is resolved to an IP that must be globally routable -- private,
loopback, link-local, and other non-global addresses are refused.

DNS-rebinding mitigation: each request opens a connection to the chosen,
validated IP literal while the original hostname travels in the ``Host``
header and in the TLS SNI extension (``extensions={"sni_hostname": host}``).
This is honored by the installed httpcore primary path, which reads
``request.extensions["sni_hostname"]`` and uses it as the TLS
``server_hostname`` (httpcore 1.0.9 ``_sync/connection.py``). Redirects are
handled manually (client ``follow_redirects=False``, at most
``PDF_MAX_REDIRECTS`` hops) with every hop re-validated and re-resolved.

Downloads stream with a hard ``PDF_MAX_BYTES`` bound, the ``%PDF-`` magic is
checked in memory before any file is written, and only the final PDF path
(``pdf/W<digits>-<sha256>.pdf``) is ever created -- exclusively
(``O_EXCL|O_NOFOLLOW``), never overwriting prior files, and never writing
``.part``, HTML, error, config, log, or harness files.

Errors are safe :class:`PDFError` values carrying a machine ``category``;
messages are fixed strings that never echo URLs, document content,
credentials, or query secrets. Extraction failures from
:mod:`radar.processing.pdf_text` are imported and reclassified here.
"""

from __future__ import annotations

import hashlib as _hashlib
import asyncio as _asyncio
import ipaddress as _ipaddress
import os as _os
import re as _re
import json as _json
import sys as _sys
import time as _time
from collections.abc import Callable as _Callable
from pathlib import Path as _Path
from urllib.parse import parse_qsl as _parse_qsl
from urllib.parse import urljoin as _urljoin
from urllib.parse import urlsplit as _urlsplit

import httpx as _httpx
from pydantic import ValidationError as _ValidationError

from radar.config.documents import (
    PDF_DOWNLOAD_TIMEOUT_S,
    PDF_MAX_BYTES,
    PDF_MAX_REDIRECTS,
)
from radar.processing.pdf_text import (
    PDFExtractionError as _PDFExtractionError,
)
from radar.processing.isolated import IsolatedProcessError, run_bounded
from radar.processing.link_validation import is_http_link
from radar.processing.pdf_text import (
    extract_pdf as _extract_pdf,
)
from radar.schema.documents import PDFDocument as _PDFDocument
from radar.schema.papers import CollectedWork as _CollectedWork

#: Dependency seam for name resolution. The default performs real DNS;
#: tests inject a fake. Maps a hostname to a list of IP-address strings.
Resolver = _Callable[[str], list[str]]

_TMP_ROOT = _Path("/tmp")
_WORK_DIGITS = _re.compile(r"W(\d+)")
_RELATIVE_PATH = _re.compile(r"^pdf/W[0-9]+-[a-f0-9]{64}\.pdf$")
_REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})
_ALLOWED_PORTS = frozenset({80, 443})

# Query parameter names treated as secrets: such links are refused outright
# (and their values never echoed) instead of being requested.
_SECRET_PARAM_NAMES = frozenset(
    {
        "token",
        "api_key",
        "apikey",
        "key",
        "secret",
        "password",
        "passwd",
        "pwd",
        "auth",
        "authorization",
        "signature",
        "sig",
        "access_token",
        "access_key",
        "client_secret",
        "session",
        "sessionid",
    }
)
_SECRET_PARAM_SUFFIXES = ("_token", "-token", "_secret", "-secret",
                          "_signature", "-signature", "_key", "-key")

_EXTRACTION_MESSAGES = {
    "encrypted_pdf": "PDF is encrypted and cannot be read.",
    "scanned_pdf": "PDF has no extractable text layer.",
    "invalid_pdf": "PDF could not be parsed.",
    "too_many_pages": "PDF exceeds its page bound.",
    "too_much_text": "PDF text exceeds its bound.",
    "too_large": "PDF exceeds its size bound.",
    "extraction_timeout": "PDF extraction exceeded its time bound.",
    "extraction_failed": "PDF text extraction failed.",
}


class PDFError(RuntimeError):
    """Safe acquisition/extraction failure with a stable machine category.

    Categories: ``no_pdf_url``, ``invalid_url``, ``invalid_work_id``,
    ``unresolvable_host``, ``blocked_host``, ``redirect_limit``,
    ``http_error``, ``network_error``, ``too_large``, ``invalid_pdf``,
    ``encrypted_pdf``, ``scanned_pdf``, ``too_many_pages``,
    ``too_much_text``, ``extraction_timeout``, ``extraction_failed``,
    ``cache_mismatch``, ``storage_error``. Messages are fixed strings that
    never carry URLs, document content, credentials, or query secrets.
    """

    def __init__(self, category: str, message: str):
        super().__init__(message)
        self.category = category


def acquire_pdf(
    work: _CollectedWork,
    directory: _Path | str,
    *,
    transport: _httpx.AsyncBaseTransport | None = None,
    resolver: Resolver | None = None,
    cached: _PDFDocument | None = None,
) -> _PDFDocument:
    """Download, store, and extract the work's OpenAlex PDF link.

    Tries each ``work.locations[].pdf_url`` in order until one yields a
    complete :class:`PDFDocument`; re-raises the last failure when none
    does. A byte-identical file already on disk is reused, never
    overwritten. When ``cached`` is supplied it is validated against the
    work (id, allowed source links, on-disk bytes hash) instead of being
    trusted, and returned without any network use.
    """
    deadline = _time.monotonic() + PDF_DOWNLOAD_TIMEOUT_S
    resolve = resolver if resolver is not None else lambda host: _default_resolver(host, _remaining(deadline))
    _require_work_id(work)
    root = _resolve_root(directory)
    allowed = _candidate_urls(work)
    if cached is not None:
        return _use_cached(cached, work, allowed, root)
    if not allowed:
        raise PDFError("no_pdf_url", "No PDF link available from OpenAlex locations.")
    async def acquire() -> _PDFDocument:
        async with _httpx.AsyncClient(transport=transport, trust_env=False,
                                      follow_redirects=False) as client:
            return await _try_candidates(work, root, allowed, client, resolve, deadline)
    return _asyncio.run(acquire())


async def _try_candidates(work, root, allowed, client, resolve, deadline):
    last: PDFError | None = None
    for candidate in allowed:
        try:
            _remaining(deadline)
            return await _acquire_candidate(work, candidate, root, client, resolve, deadline)
        except PDFError as exc:
            if exc.category in ("storage_error", "download_timeout"):
                raise
            last = exc
    assert last is not None
    raise last


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


def _use_cached(
    cached: _PDFDocument,
    work: _CollectedWork,
    allowed: list[str],
    root: _Path,
) -> _PDFDocument:
    dump = getattr(cached, "model_dump", None)
    if not callable(dump):
        raise PDFError("cache_mismatch", "Cached PDF document is not valid.")
    try:
        doc = _PDFDocument.model_validate(cached.model_dump())
    except _ValidationError:
        raise PDFError("cache_mismatch", "Cached PDF document is not valid.") from None
    if doc.work_id != work.openalex_id:
        raise PDFError("cache_mismatch", "Cached PDF does not match this work.")
    if doc.source_url not in allowed:
        raise PDFError("cache_mismatch", "Cached PDF source is not an allowed work link.")
    _validated_parts(doc.source_url)
    if not _RELATIVE_PATH.match(doc.relative_path):
        raise PDFError("cache_mismatch", "Cached PDF file is not available.")
    target = root / doc.relative_path
    if (target.parent != root / "pdf" or target.parent.is_symlink()
            or target.is_symlink() or not target.is_file()):
        raise PDFError("cache_mismatch", "Cached PDF file is not available.")
    if _sha256_file(target) != doc.sha256:
        raise PDFError("cache_mismatch", "Cached PDF bytes do not match.")
    try:
        if _extract_pdf(target) != doc.pages:
            raise PDFError("cache_mismatch", "Cached PDF text does not match its bytes.")
    except _PDFExtractionError:
        raise PDFError("cache_mismatch", "Cached PDF could not be verified.") from None
    return doc


def _sha256_file(path: _Path) -> str:
    digest = _hashlib.sha256()
    total = 0
    try:
        with open(path, "rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > PDF_MAX_BYTES + 16:
                    return ""
                digest.update(chunk)
    except OSError:
        return ""
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# One candidate link
# ---------------------------------------------------------------------------


async def _acquire_candidate(
    work: _CollectedWork,
    candidate: str,
    root: _Path,
    client: _httpx.AsyncClient,
    resolve: Resolver,
    deadline: float,
) -> _PDFDocument:
    try:
        async with _asyncio.timeout(_remaining(deadline)):
            body = await _download(candidate, client, resolve, deadline)
    except TimeoutError:
        raise PDFError("download_timeout", "PDF download exceeded its overall time bound.") from None
    digits = _require_work_id(work)
    sha = _hashlib.sha256(body).hexdigest()
    name = f"W{digits}-{sha}.pdf"
    if not _RELATIVE_PATH.match(f"pdf/{name}"):
        raise PDFError("storage_error", "PDF storage path is not valid.")
    dest = _ensure_pdf_dir(root) / name
    if dest.parent != root / "pdf":
        raise PDFError("storage_error", "PDF storage path is not valid.")
    _store_exclusive(dest, body, sha)
    try:
        pages = _extract_pdf(dest)
    except _PDFExtractionError as exc:
        raise PDFError(
            exc.category,
            _EXTRACTION_MESSAGES.get(exc.category, "PDF text extraction failed."),
        ) from None
    try:
        return _PDFDocument(
            work_id=work.openalex_id,
            source_url=candidate,
            sha256=sha,
            relative_path=f"pdf/{name}",
            pages=pages,
        )
    except _ValidationError:
        raise PDFError("extraction_failed", "PDF text extraction failed.") from None


def _store_exclusive(dest: _Path, body: bytes, sha: str) -> None:
    """Write the final PDF path exclusively; reuse byte-identical files."""
    try:
        fd = _os.open(dest, _os.O_WRONLY | _os.O_CREAT | _os.O_EXCL | _os.O_NOFOLLOW, 0o600)
    except FileExistsError:
        if _os.path.islink(dest):
            raise PDFError("storage_error", "PDF storage path is not valid.") from None
        if _sha256_file(dest) == sha:
            return
        raise PDFError("storage_error", "PDF storage holds a conflicting file.") from None
    except OSError:
        raise PDFError("storage_error", "PDF file could not be stored.") from None
    try:
        view = memoryview(body)
        while view:
            written = _os.write(fd, view)
            view = view[written:]
    except OSError:
        try:
            _os.close(fd)
        except OSError:
            pass
        try:
            _os.unlink(dest)
        except OSError:
            pass
        raise PDFError("storage_error", "PDF file could not be stored.") from None
    try:
        _os.close(fd)
    except OSError:
        raise PDFError("storage_error", "PDF file could not be stored.") from None


# ---------------------------------------------------------------------------
# HTTP download with manual redirect handling
# ---------------------------------------------------------------------------


async def _download(
    logical_url: str, client: _httpx.AsyncClient, resolve: Resolver, deadline: float
) -> bytes:
    current = logical_url
    for _ in range(PDF_MAX_REDIRECTS + 1):
        scheme, host, port, path, query = _validated_parts(current)
        address = _choose_ip(host, resolve)
        remaining = _remaining(deadline)
        target = _connection_url(scheme, address, port, path, query)
        if port is None or (scheme == "http" and port == 80) or (
            scheme == "https" and port == 443
        ):
            host_header = host
        else:
            host_header = f"{host}:{port}"
        headers = {"host": host_header, "accept": "application/pdf", "accept-encoding": "identity"}
        extensions = {"sni_hostname": host} if scheme == "https" else None
        redirect_to: str | None = None
        try:
            async with client.stream(
                "GET", target, headers=headers, extensions=extensions,
                timeout=remaining,
            ) as response:
                status = int(response.status_code)
                if status in _REDIRECT_STATUSES:
                    location = (response.headers.get("location") or "").strip()
                    if not location:
                        raise PDFError(
                            "http_error",
                            f"PDF download failed with HTTP {status}.",
                        )
                    redirect_to = _urljoin(current, location)
                elif status != 200:
                    raise PDFError(
                        "http_error", f"PDF download failed with HTTP {status}."
                    )
                else:
                    encoding = response.headers.get("content-encoding", "identity").lower().strip()
                    if encoding != "identity":
                        raise PDFError("invalid_pdf", "Unexpected PDF transfer encoding.")
                    content_type = (response.headers.get("content-type") or "").lower()
                    if "html" in content_type:
                        raise PDFError(
                            "invalid_pdf", "Downloaded bytes are not a PDF document."
                        )
                    declared = response.headers.get("content-length")
                    if declared is not None:
                        try:
                            if int(declared.strip()) > PDF_MAX_BYTES:
                                raise PDFError(
                                    "too_large",
                                    "PDF exceeds its download size bound.",
                                )
                        except (ValueError, AttributeError):
                            pass
                    chunks: list[bytes] = []
                    total = 0
                    async for chunk in response.aiter_bytes():
                        _remaining(deadline)
                        total += len(chunk)
                        if total > PDF_MAX_BYTES:
                            raise PDFError(
                                "too_large", "PDF exceeds its download size bound."
                            )
                        chunks.append(chunk)
                    body = b"".join(chunks)
                    _remaining(deadline)
                    if len(body) < 5 or body[:5] != b"%PDF-":
                        raise PDFError(
                            "invalid_pdf", "Downloaded bytes are not a PDF document."
                        )
                    return body
        except PDFError:
            raise
        except (_httpx.TimeoutException, _httpx.TransportError):
            raise PDFError(
                "network_error", "PDF download failed (network or timeout)."
            ) from None
        except _httpx.HTTPError:
            raise PDFError(
                "network_error", "PDF download failed (network or timeout)."
            ) from None
        if redirect_to is not None:
            current = redirect_to
            continue
    raise PDFError("redirect_limit", "PDF download exceeded its redirect bound.")


# ---------------------------------------------------------------------------
# Validation helpers (never echo secrets or URLs in errors)
# ---------------------------------------------------------------------------


def _require_work_id(work: _CollectedWork) -> str:
    try:
        raw = work.openalex_id
    except AttributeError:
        raise PDFError("invalid_work_id", "Work identifier is not usable.") from None
    match = _WORK_DIGITS.search(raw) if isinstance(raw, str) else None
    if match is None:
        raise PDFError("invalid_work_id", "Work identifier is not usable.")
    return match.group(1)


def _candidate_urls(work: _CollectedWork) -> list[str]:
    try:
        locations = work.locations
    except AttributeError:
        raise PDFError("no_pdf_url", "No PDF link available from OpenAlex locations.") from None
    out: list[str] = []
    for location in locations:
        try:
            raw = location.pdf_url
        except AttributeError:
            continue
        if isinstance(raw, str) and raw.strip() and raw.strip() not in out:
            out.append(raw.strip())
    return out


def _validated_parts(raw: str) -> tuple[str, str, int | None, str, str]:
    if not isinstance(raw, str) or not (1 <= len(raw) <= 2000):
        raise PDFError("invalid_url", "PDF link is not a usable public document link.")
    if not is_http_link(raw):
        raise PDFError("invalid_url", "PDF link is not a usable public document link.")
    try:
        parts = _urlsplit(raw)
    except ValueError:
        raise PDFError("invalid_url", "PDF link is not a usable public document link.") from None
    if parts.scheme not in ("http", "https"):
        raise PDFError("invalid_url", "PDF link is not a usable public document link.")
    netloc = parts.netloc
    if not netloc or "@" in netloc:
        # Any userinfo (credentials) refuses the link outright.
        raise PDFError("invalid_url", "PDF link is not a usable public document link.")
    host = parts.hostname or ""
    try:
        host.encode("ascii")
    except UnicodeEncodeError:
        raise PDFError("invalid_url", "PDF link is not a usable public document link.") from None
    if not host or parts.fragment:
        raise PDFError("invalid_url", "PDF link is not a usable public document link.") from None
    try:
        port = parts.port
    except ValueError:
        raise PDFError("invalid_url", "PDF link is not a usable public document link.") from None
    if port is not None and port not in _ALLOWED_PORTS:
        raise PDFError("invalid_url", "PDF link is not a usable public document link.")
    for name, _ in _parse_qsl(parts.query, keep_blank_values=True):
        lowered = name.strip().lower()
        if lowered in _SECRET_PARAM_NAMES or lowered.endswith(_SECRET_PARAM_SUFFIXES):
            raise PDFError("invalid_url", "PDF link is not a usable public document link.")
    return parts.scheme, host, port, parts.path or "/", parts.query


def _choose_ip(host: str, resolve: Resolver) -> str:
    try:
        literal = _ipaddress.ip_address(host)
    except ValueError:
        literal = None
    if literal is not None:
        if literal.is_global:
            return str(literal)
        raise PDFError("blocked_host", "PDF host address is not permitted.")
    try:
        candidates = resolve(host)
    except PDFError:
        raise
    except Exception:
        raise PDFError("unresolvable_host", "PDF host could not be resolved.") from None
    if not candidates:
        raise PDFError("unresolvable_host", "PDF host could not be resolved.")
    for candidate in candidates:
        if not isinstance(candidate, str):
            continue
        try:
            parsed = _ipaddress.ip_address(candidate.strip())
        except ValueError:
            continue
        if parsed.is_global:
            return str(parsed)
    raise PDFError("blocked_host", "PDF host address is not permitted.")


def _connection_url(
    scheme: str, address: str, port: int | None, path: str, query: str
) -> str:
    netloc = f"[{address}]" if ":" in address else address
    if port is not None:
        netloc += f":{port}"
    url = f"{scheme}://{netloc}{path or '/'}"
    if query:
        url += f"?{query}"
    return url


def _remaining(deadline: float) -> float:
    remaining = deadline - _time.monotonic()
    if remaining <= 0:
        raise PDFError("download_timeout", "PDF download exceeded its overall time bound.")
    return remaining


def _default_resolver(host: str, timeout_s: float = PDF_DOWNLOAD_TIMEOUT_S) -> list[str]:
    # OS DNS calls have no dependable socket timeout. Isolate the call so it
    # cannot strand this process past the shared download deadline; pipes only.
    code = (
        "import json,socket,sys; "
        "rows=socket.getaddrinfo(sys.argv[1],None,type=socket.SOCK_STREAM); "
        "print(json.dumps(list(dict.fromkeys(row[4][0] for row in rows))))"
    )
    try:
        status, raw = run_bounded([_sys.executable, "-B", "-c", code, host],
                                  timeout_s=timeout_s, max_output_bytes=65536)
        if status != 0:
            raise ValueError("DNS failed")
        addresses = _json.loads(raw)
        if not isinstance(addresses, list) or not all(isinstance(a, str) for a in addresses):
            raise ValueError("DNS output invalid")
        return addresses
    except IsolatedProcessError as exc:
        if exc.category == "timeout":
            raise PDFError("download_timeout", "PDF host resolution exceeded its time bound.") from None
        raise PDFError("unresolvable_host", "PDF host could not be resolved.") from None
    except (UnicodeError, ValueError):
        raise PDFError("unresolvable_host", "PDF host could not be resolved.") from None


def _resolve_root(directory: _Path | str) -> _Path:
    """Validate the storage root without creating anything."""
    try:
        incoming = directory if isinstance(directory, _Path) else _Path(directory)
    except (TypeError, ValueError):
        raise PDFError("storage_error", "PDF directory is not usable.") from None
    if _os.path.islink(incoming):
        raise PDFError("storage_error", "PDF directory is not usable.")
    try:
        resolved = incoming.resolve()
    except OSError:
        raise PDFError("storage_error", "PDF directory is not usable.") from None
    if resolved == _TMP_ROOT or _TMP_ROOT in resolved.parents:
        raise PDFError("storage_error", "PDF directory is not usable.")
    return resolved


def _ensure_pdf_dir(root: _Path) -> _Path:
    """Create the final ``pdf/`` dir only when validated bytes await storage."""
    pdf_dir = root / "pdf"
    try:
        if _os.path.lexists(pdf_dir) and _os.path.islink(pdf_dir):
            raise PDFError("storage_error", "PDF directory is not usable.")
        pdf_dir.mkdir(parents=True, exist_ok=True)
        if _os.path.islink(pdf_dir) or not pdf_dir.is_dir():
            raise PDFError("storage_error", "PDF directory is not usable.")
    except PDFError:
        raise
    except OSError:
        raise PDFError("storage_error", "PDF directory is not usable.") from None
    return pdf_dir
