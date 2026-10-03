"""LAN-only CLEF/SystemOne full-pool paper screening (HTTPX, no PydanticAI).

Native ``POST {base}/v1/systemone`` with two named ``noul`` questions per
paper. Confirmed-unavailable service fails fast (no retries); every work
is accounted for with an explicit status. The owned HTTP client always
closes, on success, failure, or deadline. Errors never carry request URLs,
credentials, prompts, or raw bodies.
"""

from __future__ import annotations

import concurrent.futures as _futures
import json as _json
import math as _math
import os as _os
import time as _time
from dataclasses import dataclass
from typing import Any

import httpx as _httpx

from radar.config.interests import RadarProfile
from radar.provider.freetoken import FreeTokenError, check_local_network
from radar.schema.papers import CollectedWork
from radar.schema.triage import RUBRIC_VERSION, TriageBatch, TriageResult

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
                "--clef-base-url with a loopback/private-LAN /v1/systemone "
                "endpoint. No endpoint is ever guessed."
            )
        try:
            base = check_local_network(raw)
        except FreeTokenError as exc:
            raise ClefError(
                "CLEF base URL must be a loopback or private-LAN http(s) URL "
                "without credentials."
            ) from exc
        name = ((model or _os.environ.get(CLEF_MODEL_ENV, "")) or "").strip()
        return cls(
            base_url=base,
            model=name or DEFAULT_MODEL,
            request_timeout_s=_finite_number(request_timeout_s, "request_timeout_s", 0, 30),
            overall_timeout_s=_finite_number(overall_timeout_s, "overall_timeout_s", 0, 300),
            concurrency=_checked_concurrency(concurrency),
        )


def _questions(profile: RadarProfile) -> dict[str, Any]:
    keywords = ", ".join(profile.keywords[:8]) or "machine learning"
    domains = ", ".join(profile.domains[:4]) or "behavioral science"
    return {
        AI_ML_QUESTION: {
            "type": "noul",
            "instructions": f"Is this paper relevant to AI/ML ({keywords})?",
            "criteria": {
                "true": "The paper is relevant to AI/ML research or methods.",
                "false": "The paper is not relevant to AI/ML.",
            },
        },
        CROSS_DOMAIN_QUESTION: {
            "type": "noul",
            "instructions": (
                "Does this paper show cross-domain transfer potential between "
                f"AI/ML and {domains} (either direction)?"
            ),
            "criteria": {
                "true": "The paper connects AI/ML with behavioral or economic ideas.",
                "false": "No such cross-domain connection.",
            },
        },
    }


def _payload_bytes(
    work: CollectedWork, profile: RadarProfile, model: str
) -> bytes | None:
    """Serialize one screening call; None when the abstract is missing."""
    if not (work.abstract or "").strip():
        return None
    state = {
        "title": work.title or "(untitled)",
        "abstract": work.abstract,
        "publication_year": work.publication_year,
    }
    body = _json.dumps(
        {"model": model, "state": state, "questions": _questions(profile)},
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
    answers = payload.get("answers")
    if not isinstance(answers, dict):
        return TriageResult(work_id=work_id, status="failed")
    probs: dict[str, float] = {}
    for key in (AI_ML_QUESTION, CROSS_DOMAIN_QUESTION):
        entry = answers.get(key)
        if not isinstance(entry, dict) or entry.get("type") != "noul":
            return TriageResult(work_id=work_id, status="failed")
        probs[key] = entry.get("noul")
    try:
        return TriageResult(
            work_id=work_id,
            status="scored",
            ai_ml_relevance=probs[AI_ML_QUESTION],
            cross_domain_potential=probs[CROSS_DOMAIN_QUESTION],
        )
    except ValueError:
        return TriageResult(work_id=work_id, status="failed")


def _post_one(
    client: _httpx.Client,
    url: str,
    body: bytes,
    timeout_s: float,
    work_id: str,
    model: str,
) -> TriageResult:
    """POST one screening call; service problems raise _ServiceDown."""
    try:
        response = client.post(
            url,
            content=body,
            headers={"Content-Type": "application/json"},
            timeout=timeout_s,
        )
    except (_httpx.TimeoutException, _httpx.TransportError) as exc:
        raise _ServiceDown(type(exc).__name__) from None
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
        if transport is not None:
            client = _httpx.Client(
                transport=transport, trust_env=False, follow_redirects=False
            )
        else:
            client = _httpx.Client(trust_env=False, follow_redirects=False)
        try:
            _run_pool(client, pending, profile, config, results)
        finally:
            try:
                client.close()
            except Exception:
                pass
    return TriageBatch(
        model_id=config.model,
        rubric_version=RUBRIC_VERSION,
        results=[results[wid] for wid in order],
    )


def _run_pool(
    client: _httpx.Client,
    pending: list[tuple[CollectedWork, bytes]],
    profile: RadarProfile,
    config: ClefConfig,
    results: dict[str, TriageResult],
) -> None:
    """Execute calls with bounded concurrency, fail-fast, overall deadline.

    The first call runs synchronously as a service probe: when the
    endpoint is down the pool aborts after exactly one call instead of
    repeating the failure across every work. Remaining calls fan out over
    a bounded thread pool.
    """
    url = f"{config.base_url.rstrip('/')}/systemone"
    deadline = _time.monotonic() + float(config.overall_timeout_s)
    first, rest = pending[0], pending[1:]
    try:
        outcome = _post_one(
            client, url, first[1], float(config.request_timeout_s),
            first[0].openalex_id, config.model,
        )
    except _ServiceDown:
        results[first[0].openalex_id] = TriageResult(
            work_id=first[0].openalex_id, status="failed"
        )
        for work, _body in rest:
            results[work.openalex_id] = TriageResult(
                work_id=work.openalex_id, status="failed"
            )
        return
    except Exception:
        results[first[0].openalex_id] = TriageResult(
            work_id=first[0].openalex_id, status="failed"
        )
        for work, _body in rest:
            results[work.openalex_id] = TriageResult(
                work_id=work.openalex_id, status="failed"
            )
        return
    if _time.monotonic() > deadline:
        results[first[0].openalex_id] = TriageResult(
            work_id=first[0].openalex_id, status="deadline"
        )
        for work, _body in rest:
            results[work.openalex_id] = TriageResult(
                work_id=work.openalex_id, status="deadline"
            )
        return
    results[first[0].openalex_id] = outcome
    if not rest:
        return
    with _futures.ThreadPoolExecutor(max_workers=config.concurrency) as pool:
        future_of = {
            pool.submit(
                _post_one, client, url, body,
                float(config.request_timeout_s),
                work.openalex_id, config.model,
            ): work
            for work, body in rest
        }
        queue = list(future_of.items())
        aborted = False
        for index, (future, work) in enumerate(queue):
            if aborted:
                results[work.openalex_id] = TriageResult(
                    work_id=work.openalex_id, status="failed"
                )
                try:
                    future.cancel()
                except Exception:
                    pass
                continue
            remaining = deadline - _time.monotonic()
            if remaining <= 0:
                _mark_rest_deadline(queue[index:], results)
                break
            try:
                outcome = future.result(timeout=remaining)
            except _futures.TimeoutError:
                _mark_rest_deadline(queue[index:], results)
                break
            except _ServiceDown:
                results[work.openalex_id] = TriageResult(
                    work_id=work.openalex_id, status="failed"
                )
                aborted = True
                continue
            except Exception:
                results[work.openalex_id] = TriageResult(
                    work_id=work.openalex_id, status="failed"
                )
                aborted = True
                continue
            if _time.monotonic() > deadline:
                results[work.openalex_id] = TriageResult(
                    work_id=work.openalex_id, status="deadline"
                )
                _mark_rest_deadline(queue[index + 1 :], results)
                break
            results[work.openalex_id] = outcome


def _mark_rest_deadline(
    rest: list, results: dict[str, TriageResult]
) -> None:
    for future, work in rest:
        try:
            future.cancel()
        except Exception:
            pass
        if work.openalex_id not in results:
            results[work.openalex_id] = TriageResult(
                work_id=work.openalex_id, status="deadline"
            )
