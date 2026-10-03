"""OpenAlex discovery source: HTTPX requests, retry/quota policy,
normalization, dedup, and pre-scoring hooks.

Owns all OpenAlex HTTP. No ranking (see :mod:`radar.processing.ranking`),
no CLI, no output. Errors never carry request URLs, credentials, headers,
or raw bodies.
"""

from __future__ import annotations

import json as _json
import math as _math
import os as _os
import time as _time
from typing import Any, Protocol

import httpx as _httpx

from radar.config.runtime import (
    DEFAULT_RETRY_AFTER_S,
    DEFAULT_TIMEOUT_S,
    LONG_QUOTA_RESET_S,
    MAX_429_RETRIES,
    MAX_ABSTRACT_CHARS,
    MAX_PER_PAGE,
    MAX_QUERIES,
    MAX_RETRY_AFTER_S,
    MAX_TERM_CHARS,
    MAX_TIMEOUT_S,
    MAX_TOTAL_WORKS,
    QUOTA_BODY_READ_LIMIT,
    RESPONSE_READ_LIMIT,
)
from radar.processing.link_validation import is_openalex_work_link
from radar.schema.papers import (
    CollectedWork,
    LocationInfo,
    PlannedQuery,
    QueryPlan,
)

BASE_URL = "https://api.openalex.org/works"
# Bound the `select` payload; respected by current OpenAlex /works API.
SELECT_FIELDS = (
    "id,title,abstract_inverted_index,doi,publication_year,"
    "primary_location,best_oa_location,locations,cited_by_count"
)
USER_AGENT = "eureka-radar-prototype/0.1 (mailto:prototype@example.com)"


# ---------------------------------------------------------------------------
# Transports
# ---------------------------------------------------------------------------


class OpenAlexTransport(Protocol):
    """Injectable HTTP seam. Implementations return parsed JSON."""

    def get_json(
        self, url: str, params: dict[str, str], headers: dict[str, str], timeout: float
    ) -> dict[str, Any]: ...


class OpenAlexHttpError(RuntimeError):
    """Non-quota OpenAlex transport failure (redacted, typed by status).

    Carries only the numeric ``status_code`` (None for transport-level
    failures such as timeouts) plus the parsed ``retry_after`` seconds for
    429s (None when the server sent none). Never carries request URLs,
    query params, credentials, headers, or raw bodies.
    """

    def __init__(
        self, status_code: int | None, message: str,
        retry_after: float | None = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.retry_after = retry_after


def default_client() -> _httpx.Client:
    """Build the default OpenAlex client: no proxy env, no redirects."""
    return _httpx.Client(trust_env=False, follow_redirects=False)


class HttpxTransport:
    """Default httpx transport with strict timeouts and redacted errors."""

    def __init__(self, client: _httpx.Client | None = None):
        self._client = client if client is not None else default_client()
        self._owned = client is None

    def close(self) -> None:
        """Close the owned client; safe to call for injected clients too."""
        if self._owned:
            try:
                self._client.close()
            except Exception:
                pass

    def __enter__(self) -> "HttpxTransport":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def get_json(self, url, params, headers, timeout):
        timeout = _check_timeout(timeout)
        try:
            resp = self._client.get(
                url, params=params, headers=headers or {}, timeout=timeout
            )
            resp.raise_for_status()
        except _httpx.HTTPStatusError as exc:
            status = exc.response.status_code if exc.response is not None else None
            if status == 429 and exc.response is not None:
                quota = _quota_error_from_parts(
                    exc.response.status_code,
                    exc.response.headers,
                    bytes(exc.response.content[:QUOTA_BODY_READ_LIMIT]),
                )
                if quota is not None:
                    raise quota from None
                raise OpenAlexHttpError(
                    status, f"OpenAlex request failed with HTTP {status} (transient).",
                    retry_after=_retry_after_seconds(exc.response.headers),
                ) from None
            raise OpenAlexHttpError(
                status,
                f"OpenAlex request failed with HTTP {status}."
                if status is not None
                else "OpenAlex request failed (no HTTP status).",
            ) from None
        except (_httpx.TimeoutException, _httpx.TransportError) as exc:
            raise OpenAlexHttpError(
                None, f"OpenAlex request failed ({type(exc).__name__}); "
                "check network access to https://api.openalex.org."
            ) from None
        raw = resp.content[:RESPONSE_READ_LIMIT]
        try:
            data = _json.loads(raw.decode("utf-8", errors="replace"))
        except (ValueError, UnicodeError) as exc:
            raise ValueError(f"OpenAlex returned non-JSON payload: {exc}") from exc
        if not isinstance(data, dict):
            raise ValueError("OpenAlex returned malformed envelope (not an object)")
        return data


class RetryingTransport:
    """Wrap a transport with bounded 429 retries plus quota fast-fail.

    Retries only transient HTTP 429s, honoring the server's ``Retry-After``
    header up to ``MAX_RETRY_AFTER_S``. Confirmed quota exhaustion
    (:class:`OpenAlexQuotaError`) and all other errors propagate
    immediately, never sleeping.
    """

    def __init__(self, inner: OpenAlexTransport, max_retries: int = MAX_429_RETRIES):
        self._inner = inner
        self._max_retries = max(0, min(int(max_retries), 5))

    def get_json(self, url, params, headers, timeout):
        attempts = 0
        while True:
            try:
                return self._inner.get_json(url, params, headers, timeout)
            except Exception as exc:
                if isinstance(exc, OpenAlexQuotaError):
                    raise
                wait = _transient_429_wait(exc, attempts, self._max_retries)
                if wait is None:
                    raise
                attempts += 1
                _time.sleep(wait)


def _transient_429_wait(exc: BaseException, attempts: int, max_retries: int) -> float | None:
    """Bounded backoff for a transient 429, else None (propagate)."""
    status: int | None = None
    retry_after: float | None = None
    if isinstance(exc, OpenAlexHttpError):
        status = exc.status_code
        retry_after = _retry_after_value(getattr(exc, "retry_after", None))
    else:
        # Legacy urllib shape (older test doubles): honor ``code``.
        code = getattr(exc, "code", None)
        if isinstance(code, int):
            status = code
        elif getattr(exc, "response", None) is not None:
            try:
                status = int(exc.response.status_code)
            except (TypeError, ValueError):
                status = None
        headers = getattr(exc, "headers", None)
        if headers is not None:
            retry_after = _retry_after_value(_header_first(headers, "retry-after"))
    if status != 429 or attempts >= max_retries:
        return None
    if retry_after is None:
        return DEFAULT_RETRY_AFTER_S
    return min(MAX_RETRY_AFTER_S, max(1.0, retry_after))


def _retry_after_value(raw: Any) -> float | None:
    """Parse Retry-After seconds; None when missing, negative, or malformed."""
    if raw is None:
        return None
    try:
        seconds = float(str(raw).strip())
    except (TypeError, ValueError):
        return None
    if not _math.isfinite(seconds) or seconds < 0:
        return None
    return seconds


# ---------------------------------------------------------------------------
# Daily-quota boundary
# ---------------------------------------------------------------------------

#: Bounded peek into a 429 error body; quota verdicts never need more.
QUOTA_BODY_READ_LIMIT = 4096

#: A Retry-After at or above this (with Remaining 0) means a daily budget
#: reset, not transient throttling. Well below the observed ~22h reset.
LONG_QUOTA_RESET_S = 3600.0


class OpenAlexQuotaError(RuntimeError):
    """Confirmed OpenAlex daily-budget exhaustion: fail fast, do not retry.

    Deliberately NOT an HTTP error type, so the bounded 429 retry wrapper
    lets it through without sleeping. The message carries only safe scalars
    (remaining/reset summary plus the remediation); never request URLs,
    credentials, or raw error bodies.
    """


def _header_first(headers: Any, *names: str) -> str | None:
    """Case-insensitive first present header value, else None."""
    if headers is None:
        return None
    try:
        get = headers.get
    except AttributeError:
        return None
    for name in names:
        try:
            value = get(name)
        except Exception:
            continue
        if value is not None:
            return str(value)
    # Fallback for plain-dict headers without case-insensitive lookup.
    try:
        items = list(headers.items())
    except Exception:
        return None
    lowered = {n.lower() for n in names}
    for key, value in items:
        try:
            if str(key).lower() in lowered:
                return str(value)
        except Exception:
            continue
    return None


def _retry_after_seconds(headers: Any) -> float | None:
    """Parse Retry-After seconds from headers; None when missing/malformed."""
    raw = _header_first(headers, "retry-after")
    return _retry_after_value(raw)


def _remaining_budget(headers: Any) -> int | None:
    """Parse the remaining-budget header; None when missing or malformed."""
    raw = _header_first(headers, "x-ratelimit-remaining", "ratelimit-remaining")
    if raw is None:
        return None
    try:
        return int(raw.strip())
    except (TypeError, ValueError):
        return None


def _body_confirms_budget_exhaustion(raw: bytes) -> bool:
    """Whether the bounded body explicitly reports an exhausted budget."""
    try:
        text = raw.decode("utf-8", errors="replace").lower()
    except Exception:
        return False
    return "insufficient" in text and "budget" in text


def _quota_error_from_parts(
    status: int | None,
    headers: Any,
    body: bytes,
) -> OpenAlexQuotaError | None:
    """Classify a response as confirmed daily-budget exhaustion, or None.

    Confirmed when the bounded error body explicitly reports an
    insufficient budget, or when headers show remaining budget 0 with a
    long (daily-scale) Retry-After. Anything else -- including remaining 0
    with a short Retry-After and no explicit body -- returns None so the
    caller treats the failure as transient. Never raises, never exposes
    credentials, URLs, or raw bodies.
    """
    if status != 429:
        return None
    retry_after = _retry_after_seconds(headers)
    if _body_confirms_budget_exhaustion(body):
        detail = (
            f"shared anonymous daily budget exhausted (remaining 0"
            + (
                f"; resets in ~{retry_after / 3600:.0f}h"
                if retry_after is not None
                else "; resets at midnight UTC"
            )
            + ")."
        )
    elif _remaining_budget(headers) == 0 and (
        retry_after is not None and retry_after >= LONG_QUOTA_RESET_S
    ):
        detail = (
            f"shared anonymous daily budget exhausted (remaining 0; "
            f"retry-after ~{retry_after / 3600:.0f}h)."
        )
    else:
        return None
    return OpenAlexQuotaError(
        f"OpenAlex {detail} Set a free OPENALEX_API_KEY for a personal "
        f"budget (https://help.openalex.org/api/authentication/) or wait "
        f"until reset. This failure happens before any snapshot write, so "
        f"the last snapshot is preserved."
    )


class DictTransport:
    """Test helper: serve canned responses keyed by search terms."""

    def __init__(self, pages: dict[str, dict[str, Any]]):
        self.pages = pages
        self.calls: list[dict[str, Any]] = []

    def get_json(self, url, params, headers, timeout):
        _check_timeout(timeout)
        self.calls.append({"url": url, "params": dict(params)})
        key = params.get("search", "")
        payload = self.pages.get(key, {"results": []})
        if not isinstance(payload, dict):
            raise ValueError("canned payload must be an object")
        return payload


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _check_timeout(timeout: float) -> float:
    if not isinstance(timeout, (int, float)) or not (0 < timeout <= MAX_TIMEOUT_S):
        raise ValueError(f"timeout must be within (0, {MAX_TIMEOUT_S}]s")
    return float(timeout)


def get_api_key(explicit: str | None = None) -> str | None:
    """Return explicit key or OPENALEX_API_KEY env var, else None."""
    if explicit and explicit.strip():
        return explicit.strip()
    env = _os.environ.get("OPENALEX_API_KEY", "").strip()
    return env or None


def _safe_str(value: Any, limit: int = 2000) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        try:
            value = str(value)
        except Exception:
            return ""
    return value.strip()[:limit]


# ---------------------------------------------------------------------------
# Request construction (query policy lives in radar.config.searches)
# ---------------------------------------------------------------------------


def build_request(
    planned: PlannedQuery,
    api_key: str | None = None,
    mailto: str | None = None,
    timeout: float = DEFAULT_TIMEOUT_S,
) -> tuple[str, dict[str, str], dict[str, str], float]:
    """Build (url, params, headers, timeout) respecting OpenAlex constraints.

    - `search` carries keyword/semantic terms (OpenAlex has no separate
      semantic endpoint on /works; relevance ranking applies when `search`
      is present).
    - Every query with `from_date` adds
      `filter=from_publication_date:<date>` so the configured lookback
      window applies to the whole plan; only the recent kind additionally
      sends `sort=publication_date:desc`. The semantic kind omits `sort`
      because OpenAlex applies relevance ranking by default when `search`
      is present (verified live 2026-09-30: `sort=relevance` is rejected
      with 400, while omitting `sort` returns 200).
    - `per-page` capped at 50; `select` bounds payload; `mailto` polite pool.
    """
    timeout = _check_timeout(timeout)
    per_page = max(1, min(int(planned.per_page), MAX_PER_PAGE))
    params: dict[str, str] = {
        "search": planned.terms.strip()[:MAX_TERM_CHARS],
        "per-page": str(per_page),
        "select": SELECT_FIELDS,
    }
    if not params["search"]:
        raise ValueError("planned query has empty search terms")
    if planned.from_date:
        params["filter"] = f"from_publication_date:{planned.from_date}"
    if planned.kind == "recent":
        if not planned.from_date:
            raise ValueError("recent query requires from_date")
        params["sort"] = "publication_date:desc"
    # Semantic kind: no `sort` param — OpenAlex relevance ranking is the
    # default when `search` is present. (`sort=relevance` is invalid.)
    key = get_api_key(api_key)
    if key:
        params["api_key"] = key
    if mailto and mailto.strip():
        params["mailto"] = mailto.strip()[:200]
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    return BASE_URL, params, headers, timeout


# ---------------------------------------------------------------------------
# Abstract reconstruction + normalization
# ---------------------------------------------------------------------------


def reconstruct_abstract(index: Any) -> str:
    """Rebuild text from OpenAlex abstract_inverted_index (untrusted input)."""
    if not isinstance(index, dict) or not index:
        return ""
    slots: list[tuple[int, str]] = []
    for word, positions in index.items():
        w = _safe_str(word, 200)
        if not w or not isinstance(positions, list):
            continue
        if " " in w:
            # Single tokens expected; skip suspicious multi-word keys.
            continue
        for pos in positions:
            if isinstance(pos, bool) or not isinstance(pos, int):
                continue
            if 0 <= pos < 100_000:
                slots.append((pos, w))
    if not slots:
        return ""
    slots.sort(key=lambda s: s[0])
    text = " ".join(w for _, w in slots)
    return text[:MAX_ABSTRACT_CHARS]


def _normalize_locations(raw: Any) -> list[LocationInfo]:
    out: list[LocationInfo] = []
    candidates: list[Any] = []
    if isinstance(raw, dict):
        for key in ("primary_location", "best_oa_location"):
            loc = raw.get(key)
            if isinstance(loc, dict):
                candidates.append(loc)
        locs = raw.get("locations")
        if isinstance(locs, list):
            candidates.extend([x for x in locs if isinstance(x, dict)])
    for loc in candidates:
        source = loc.get("source") if isinstance(loc.get("source"), dict) else {}
        src_name = _safe_str(source.get("display_name"), 300) if source else ""
        landing = _safe_str(
            loc.get("landing_page_url") or loc.get("id") or source.get("homepage_url", ""),
            2000,
        )
        pdf = _safe_str(loc.get("pdf_url"), 2000)
        is_oa = loc.get("is_oa") is True
        # arXiv may only appear as a returned location string, never queried.
        entry = LocationInfo(
            source_name=src_name, landing_url=landing, pdf_url=pdf, is_oa=is_oa
        )
        out.append(entry)
        if len(out) >= 5:
            break
    return out


def normalize_work(
    raw: Any, matched_query: str, query_kind: str = "semantic"
) -> CollectedWork | None:
    """Normalize one raw OpenAlex work dict; None when malformed/unusable."""
    if not isinstance(raw, dict):
        return None
    wid = _safe_str(raw.get("id"), 500)
    if not is_openalex_work_link(wid):
        return None  # essential key missing -> skip, never raise
    title = _safe_str(raw.get("title") or raw.get("display_name"), 2000)
    abstract = reconstruct_abstract(raw.get("abstract_inverted_index"))
    year = raw.get("publication_year")
    if isinstance(year, bool) or not isinstance(year, int):
        year = None
    if year is not None and not (1500 <= year <= 2100):
        year = None
    doi = _safe_str(raw.get("doi"), 500)
    cited = raw.get("cited_by_count")
    if isinstance(cited, bool) or not isinstance(cited, int) or cited < 0:
        cited = 0
    locations = _normalize_locations(raw)
    primary_url = ""
    for loc in locations:
        if loc.landing_url:
            primary_url = loc.landing_url
            break
    if not primary_url:
        primary_url = _safe_str(raw.get("id"), 2000)
    mq = _safe_str(matched_query, 300)
    return CollectedWork(
        openalex_id=wid,
        title=title,
        abstract=abstract,
        publication_year=year,
        doi=doi,
        primary_url=primary_url,
        locations=locations,
        cited_by_count=cited,
        matched_queries=[mq] if mq else [],
        query_kinds=[query_kind] if query_kind else [],
        score=0.0,
    )


# ---------------------------------------------------------------------------
# Collection orchestration (dedup only; scoring/ranking is a pipeline step)
# ---------------------------------------------------------------------------


def collect(
    plan: QueryPlan,
    transport: OpenAlexTransport,
    api_key: str | None = None,
    mailto: str | None = None,
    timeout: float = DEFAULT_TIMEOUT_S,
    max_total: int = MAX_TOTAL_WORKS,
) -> list[CollectedWork]:
    """Execute the whole bounded plan via transport and dedup by OpenAlex ID.

    ``max_total`` is the hard candidate-pool cap (≤200), not the final
    output bound: every planned query is executed in order until the pool
    reaches the cap. Returned works are unscored, in first-seen plan order;
    callers rank via :mod:`radar.processing.ranking` and apply the
    user-facing ``--max-candidates`` bound as a final slice so later query
    branches still contribute when the output bound is small.
    """
    if not plan.queries:
        raise ValueError("query plan must contain at least one query")
    timeout = _check_timeout(timeout)
    max_total = max(1, min(int(max_total), MAX_TOTAL_WORKS))
    by_id: dict[str, CollectedWork] = {}
    for planned in plan.queries[:MAX_QUERIES]:
        url, params, headers, _ = build_request(
            planned, api_key=api_key, mailto=mailto, timeout=timeout
        )
        payload = transport.get_json(url, params, headers, timeout)
        if not isinstance(payload, dict):
            continue  # malformed envelope -> skip page, keep others
        results = payload.get("results")
        if not isinstance(results, list):
            continue
        for raw in results[:MAX_PER_PAGE]:  # bound per-query results
            work = normalize_work(raw, planned.terms, planned.kind)
            if work is None:
                continue
            existing = by_id.get(work.openalex_id)
            if existing is None:
                by_id[work.openalex_id] = work
            else:
                # Dedup: merge provenance deterministically, keep first text.
                for q in work.matched_queries:
                    if q and q not in existing.matched_queries:
                        existing.matched_queries.append(q)
                for k in work.query_kinds:
                    if k and k not in existing.query_kinds:
                        existing.query_kinds.append(k)
            if len(by_id) >= max_total:
                break
        if len(by_id) >= max_total:
            break
    return list(by_id.values())[:max_total]
