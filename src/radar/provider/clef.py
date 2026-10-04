"""LAN-only CLEF/SystemOne full-pool paper screening (HTTPX, no PydanticAI).

Native ``POST {base}/v1/systemone`` with two named ``noul`` questions per
paper. Confirmed-unavailable service fails fast (no retries); every work
is accounted for with an explicit status. The owned HTTP client always
closes, on success, failure, or deadline. Errors never carry request URLs,
credentials, prompts, or raw bodies.
"""

from __future__ import annotations

import asyncio as _asyncio
import json as _json
import math as _math
import os as _os
import urllib.parse as _urlparse
from dataclasses import dataclass
from typing import Any

import httpx as _httpx

from radar.config.interests import RadarProfile
from radar.provider.strata import StrataError, check_local_network
from radar.processing.triage_input import build_input
from radar.schema.papers import CollectedWork
from radar.schema.triage import SystemOneResponse, TriageBatch, TriageResult
from radar.prompts.catalog import screening_questions

DEFAULT_MODEL = "clef-flash"
CLEF_BASE_URL_ENV = "CLEF_BASE_URL"
CLEF_MODEL_ENV = "CLEF_MODEL"

#: Text-only SystemOne bodies must fit 64 KiB; measured after serialization.
MAX_PAYLOAD_BYTES = 64 * 1024

#: Concurrency ceiling: bounded thread pool, never more.
MAX_CONCURRENCY = 8

AI_ML_QUESTION = "ai_ml_relevance"
CROSS_DOMAIN_QUESTION = "cross_domain_potential"


class ClefError(RuntimeError):
    """Unusable CLEF configuration (missing/invalid endpoint or bounds)."""


class _ServiceDown(Exception):
    """One screening call hit a transport/service failure: abort the pool."""


def _finite_number(value: object, name: str, low: float, high: float) -> float:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ClefError(f"{name} must be a number within ({low}, {high}]") from exc
    if (
        isinstance(value, bool)
        or not _math.isfinite(number)
        or not (low < number <= high)
    ):
        raise ClefError(f"{name} must be a number within ({low}, {high}]")
    return number


def _checked_concurrency(value: object) -> int:
    if isinstance(value, bool):
        raise ClefError("concurrency must be an integer within 1..8")
    try:
        number = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ClefError("concurrency must be an integer within 1..8") from exc
    if not 1 <= number <= MAX_CONCURRENCY:
        raise ClefError("concurrency must be an integer within 1..8")
    return number


@dataclass(frozen=True)
class ClefConfig:
    """Resolved LAN-only CLEF connection and bound settings."""

    base_url: str
    model: str = DEFAULT_MODEL
    request_timeout_s: float = 10.0
    overall_timeout_s: float = 60.0
    concurrency: int = 4

    @classmethod
    def resolve(
        cls,
        base_url: str | None = None,
        model: str | None = None,
        request_timeout_s: object = 10.0,
        overall_timeout_s: object = 60.0,
        concurrency: object = 4,
    ) -> "ClefConfig":
        raw = ((base_url or _os.environ.get(CLEF_BASE_URL_ENV, "")) or "").strip()
        if not raw:
            raise ClefError(
                "CLEF base URL is not configured; set CLEF_BASE_URL or pass "
                "--clef-base-url with a loopback/private-LAN endpoint base "
                "(empty path or /v1; the /systemone path is appended "
                "automatically). No endpoint is ever guessed."
            )
        base = _normalize_base(raw)
        try:
            check_local_network(base)
        except StrataError:
            raise ClefError(
                "CLEF base URL must be a loopback or private-LAN http(s) URL "
                "without credentials."
            ) from None
        name = ((model or _os.environ.get(CLEF_MODEL_ENV, "")) or "").strip()
        return cls(
            base_url=base,
            model=name or DEFAULT_MODEL,
            request_timeout_s=_finite_number(request_timeout_s, "request_timeout_s", 0, 30),
            overall_timeout_s=_finite_number(overall_timeout_s, "overall_timeout_s", 0, 300),
            concurrency=_checked_concurrency(concurrency),
        )


def _normalize_base(raw: str) -> str:
    """Normalize an explicit endpoint to a ``/v1``-rooted base URL.

    A bare private host base gains ``/v1``; an explicit ``/v1/systemone``
    URL is trimmed back to ``/v1`` (the transport appends ``/systemone``);
    a ``/v1`` root -- including a reverse-proxy prefix ending in ``/v1`` --
    is retained. Credentials, query strings, fragments, non-http(s)
    schemes, and other paths are rejected with redacted errors that never
    echo secrets. No host is ever guessed.
    """
    try:
        parts = _urlparse.urlsplit(raw)
    except ValueError as exc:
        raise ClefError("CLEF base URL is not a valid URL.") from exc
    if parts.username is not None or parts.password is not None:
        raise ClefError(
            "CLEF base URL must not contain credentials; configure a plain "
            "loopback/private-LAN endpoint."
        )
    if parts.query or parts.fragment:
        raise ClefError(
            "CLEF base URL must be a plain endpoint without query or fragment."
        )
    if parts.scheme not in ("http", "https"):
        raise ClefError("CLEF base URL must use http(s).")
    host = parts.hostname or ""
    if not host:
        raise ClefError("CLEF base URL must contain a host.")
    try:
        port = parts.port
    except ValueError as exc:
        raise ClefError("CLEF base URL has an invalid port.") from exc
    path = (parts.path or "").rstrip("/")
    if path in ("", "/"):
        path = "/v1"
    elif path.endswith("/v1/systemone"):
        path = path[: -len("/systemone")]
    elif not (path == "/v1" or path.endswith("/v1")):
        raise ClefError(
            "CLEF base URL path is unsupported; use an empty path, "
            "a /v1 root (reverse-proxy prefixes ending in /v1 are kept), "
            "or a /v1/systemone URL."
        )
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    netloc = f"{host}:{port}" if port is not None else host
    return f"{parts.scheme}://{netloc}{path}"


def _payload_bytes(
    work: CollectedWork, profile: RadarProfile, model: str
) -> bytes | None:
    """Serialize one screening call; None when the abstract is missing."""
    if not (work.abstract or "").strip():
        return None
    body = _json.dumps(
        build_input(work, profile, model),
        ensure_ascii=False,
    ).encode("utf-8")
    return body


def _validate_answer(
    work_id: str, payload: Any, model: str
) -> TriageResult:
    """Validate one SystemOne response body into a scored/failed result."""
    if not isinstance(payload, dict):
        return TriageResult(work_id=work_id, status="failed")
    if payload.get("model") != model:
        return TriageResult(work_id=work_id, status="failed")
    try:
        return SystemOneResponse.model_validate(payload).to_result(work_id)
    except ValueError:
        return TriageResult(work_id=work_id, status="failed")


async def _post_one_async(
    client: _httpx.AsyncClient,
    url: str,
    body: bytes,
    timeout_s: float,
    work_id: str,
    model: str,
) -> TriageResult:
    """POST one screening call; service problems raise _ServiceDown."""
    try:
        response = await client.post(
            url,
            content=body,
            headers={"Content-Type": "application/json"},
            timeout=timeout_s,
        )
    except (_httpx.TimeoutException, _httpx.TransportError) as exc:
        raise _ServiceDown(type(exc).__name__) from None
    except _asyncio.CancelledError:
        raise
    except Exception as exc:
        raise _ServiceDown(type(exc).__name__) from None
    if response.status_code != 200:
        raise _ServiceDown(f"HTTP {response.status_code}")
    try:
        payload = _json.loads(response.content.decode("utf-8", errors="replace"))
    except ValueError:
        return TriageResult(work_id=work_id, status="failed")
    return _validate_answer(work_id, payload, model)


def screen_works(
    works: list[CollectedWork],
    profile: RadarProfile,
    *,
    config: ClefConfig,
    transport: _httpx.BaseTransport | None = None,
) -> TriageBatch:
    """Screen every work via SystemOne; never raises for service outcomes.

    Missing abstracts need no call; oversize payloads (measured UTF-8
    bytes) are refused without calling. The first service failure aborts
    the pool (fail fast); unattempted works are marked failed, works
    outstanding at the overall deadline are marked deadline. The owned
    client always closes.
    """
    pending: list[tuple[CollectedWork, bytes]] = []
    results: dict[str, TriageResult] = {}
    order: list[str] = []
    for work in works:
        order.append(work.openalex_id)
        body = _payload_bytes(work, profile, config.model)
        if body is None:
            results[work.openalex_id] = TriageResult(
                work_id=work.openalex_id, status="missing_abstract"
            )
        elif len(body) > MAX_PAYLOAD_BYTES:
            results[work.openalex_id] = TriageResult(
                work_id=work.openalex_id, status="oversized"
            )
        else:
            pending.append((work, body))
    if pending:
        results.update(
            _asyncio.run(
                _screen_pool_async(pending, profile, config, transport)
            )
        )
    return TriageBatch(
        model_id=config.model,
        rubric_version=screening_questions().rubric_version,
        rubric_hash=screening_questions().fingerprint,
        results=[results[wid] for wid in order],
    )


async def _screen_pool_async(
    pending: list[tuple[CollectedWork, bytes]],
    profile: RadarProfile,
    config: ClefConfig,
    transport: _httpx.BaseTransport | None,
) -> dict[str, TriageResult]:
    """Run all calls with bounded concurrency under a hard overall deadline.

    The first call runs alone as a service probe, so a dead endpoint aborts
    after exactly one call. Cancellation is cooperative through httpx: when
    the overall deadline fires, in-flight calls are cancelled, no work is
    left running, and the owned client closes deterministically.
    """
    if transport is not None:
        client = _httpx.AsyncClient(
            transport=transport, trust_env=False, follow_redirects=False
        )
    else:
        client = _httpx.AsyncClient(trust_env=False, follow_redirects=False)
    results: dict[str, TriageResult] = {}
    url = f"{config.base_url.rstrip('/')}/systemone"
    try:
        async with client:
            start = _asyncio.get_running_loop().time()
            budget = float(config.overall_timeout_s)
            try:
                async with _asyncio.timeout(budget):
                    first, rest = pending[0], pending[1:]
                    try:
                        outcome = await _post_one_async(
                            client, url, first[1],
                            float(config.request_timeout_s),
                            first[0].openalex_id, config.model,
                        )
                    except _ServiceDown:
                        _mark_all(results, pending, "failed")
                    except Exception:
                        _mark_all(results, pending, "failed")
                    else:
                        if _asyncio.get_running_loop().time() - start > budget:
                            _mark_all(results, pending, "deadline")
                        else:
                            results[first[0].openalex_id] = outcome
                            if rest:
                                await _fanout_async(client, url, rest, config, results)
            except (TimeoutError, _asyncio.CancelledError):
                for work, _body in pending:
                    if work.openalex_id not in results:
                        results[work.openalex_id] = TriageResult(
                            work_id=work.openalex_id, status="deadline"
                        )
    finally:
        try:
            if not client.is_closed:
                await client.aclose()
        except Exception:
            pass
    return results


def _mark_all(
    results: dict[str, TriageResult],
    pending: list[tuple[CollectedWork, bytes]],
    status: str,
) -> None:
    for work, _body in pending:
        results[work.openalex_id] = TriageResult(
            work_id=work.openalex_id, status=status  # type: ignore[arg-type]
        )


async def _fanout_async(
    client: _httpx.AsyncClient,
    url: str,
    rest: list[tuple[CollectedWork, bytes]],
    config: ClefConfig,
    results: dict[str, TriageResult],
) -> None:
    """Fan out remaining calls; first service failure aborts the rest.

    Uncompleted works are left unmarked here: the caller assigns failed
    after an abort and deadline after the overall timeout fires.
    """
    sem = _asyncio.Semaphore(config.concurrency)

    async def _one(work: CollectedWork, body: bytes) -> TriageResult:
        async with sem:
            return await _post_one_async(
                client, url, body, float(config.request_timeout_s),
                work.openalex_id, config.model,
            )

    tasks = {
        _asyncio.ensure_future(_one(work, body)): work for work, body in rest
    }
    start = _asyncio.get_running_loop().time()
    budget = float(config.overall_timeout_s)
    try:
        for index, (task, work) in enumerate(list(tasks.items())):
            try:
                outcome = await task
            except _ServiceDown:
                _abort_rest(tasks, results, work.openalex_id)
                return
            except _asyncio.CancelledError:
                raise
            except Exception:
                _abort_rest(tasks, results, work.openalex_id)
                return
            if _asyncio.get_running_loop().time() - start > budget:
                results[work.openalex_id] = TriageResult(
                    work_id=work.openalex_id, status="deadline"
                )
                _mark_pending_deadline(list(tasks.items())[index + 1 :], results)
                for later in list(tasks)[index + 1 :]:
                    if not later.done():
                        later.cancel()
                break
            results[work.openalex_id] = outcome
    finally:
        # Settle everything: no task left running or unretrieved.
        for task in tasks:
            if not task.done():
                task.cancel()
        await _asyncio.gather(*tasks, return_exceptions=True)


def _mark_pending_deadline(
    rest: list, results: dict[str, TriageResult]
) -> None:
    for _task, work in rest:
        if work.openalex_id not in results:
            results[work.openalex_id] = TriageResult(
                work_id=work.openalex_id, status="deadline"  # type: ignore[arg-type]
            )


def _abort_rest(
    tasks: dict, results: dict[str, TriageResult], failed_id: str
) -> None:
    """Mark one work failed and every unattempted work failed (fail fast)."""
    for _task, work in tasks.items():
        if work.openalex_id not in results:
            results[work.openalex_id] = TriageResult(
                work_id=work.openalex_id, status="failed"  # type: ignore[arg-type]
            )
