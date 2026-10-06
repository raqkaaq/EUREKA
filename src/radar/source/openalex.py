"""OpenAlex discovery source: HTTPX requests, retry/quota policy,
normalization, dedup, and pre-scoring hooks.

Owns all OpenAlex HTTP. No ranking (see :mod:`radar.processing.ranking`),
no CLI, no output. Errors never carry request URLs, credentials, headers,
or raw bodies.
"""

from __future__ import annotations

import json as _json
import datetime as _dt
import math as _math
import os as _os
import time as _time
from typing import Any, Protocol

import httpx as _httpx

from collections.abc import Sequence as _Sequence

from radar.config.runtime import (
    DEFAULT_RETRY_AFTER_S,
    DEFAULT_TIMEOUT_S,
    LONG_QUOTA_RESET_S,
    MAX_429_RETRIES,
    MAX_ABSTRACT_CHARS,
    MAX_QUERIES,
    MAX_RETRY_AFTER_S,
    MAX_TERM_CHARS,
    MAX_SEMANTIC_CHARS,
    MAX_TIMEOUT_S,
    MAX_TOTAL_WORKS,
    QUOTA_BODY_READ_LIMIT,
    RESPONSE_READ_LIMIT,
)
from radar.processing.link_validation import is_openalex_work_link
from radar.schema.papers import (
    CITATION_KINDS,
    CollectedWork,
    DiscoveryMatch,
    LocationInfo,
    PlannedQuery,
    QueryPlan,
)

BASE_URL = "https://api.openalex.org/works"
# Bound the `select` payload; respected by current OpenAlex /works API.
SELECT_FIELDS = (
    "id,title,abstract_inverted_index,doi,publication_year,publication_date,"
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


def _get_stored_semantic_completion(transport: object) -> float | None:
    """Last semantic completion for this transport instance, else None.

    Reads the owned ``_last_semantic_completion`` attribute when present.
    Duck-typed doubles without the attribute (or with ``__slots__`` that
    reject it) simply yield None; callers keep a local fallback so
    within-run pacing still holds.
    """
    try:
        value = getattr(transport, "_last_semantic_completion", None)
    except Exception:
        return None
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        result = float(value)
        if _math.isfinite(result):
            return result
    return None


def _store_semantic_completion(transport: object, value: float) -> None:
    """Persist pacing state on the transport instance when possible."""
    try:
        setattr(transport, "_last_semantic_completion", float(value))
    except Exception:
        # ``__slots__`` doubles or read-only proxies: within-run pacing
        # still holds via the executor-local timestamp; only cross-wave
        # reuse via this instance is unavailable. Wrap with
        # :class:`SemanticPacingAdapter` in that case (see below).
        pass


class SemanticPacingAdapter:
    """Small adapter holding 1/s semantic pacing for slot-constrained transports.

    Reuse the same adapter instance across waves (it forwards ``get_json``
    to the wrapped transport). Normal code reuses the same
    :class:`RetryingTransport` across waves instead; this adapter is only
    needed when the underlying transport uses ``__slots__`` and cannot
    store ``_last_semantic_completion`` itself.
    """

    def __init__(self, inner: OpenAlexTransport):
        self._inner = inner
        self._last_semantic_completion: float | None = None

    def get_json(self, url, params, headers, timeout):
        return self._inner.get_json(url, params, headers, timeout)


class HttpxTransport:
    """Default httpx transport with strict timeouts and redacted errors."""

    def __init__(self, client: _httpx.Client | None = None):
        self._client = client if client is not None else default_client()
        self._owned = client is None
        self._last_semantic_completion: float | None = None

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
        self._last_semantic_completion: float | None = None

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
    """Test helper: serve canned responses keyed by search terms or filters."""

    def __init__(self, pages: dict[str, dict[str, Any]]):
        self.pages = pages
        self.calls: list[dict[str, Any]] = []
        self._last_semantic_completion: float | None = None

    def get_json(self, url, params, headers, timeout):
        _check_timeout(timeout)
        self.calls.append({"url": url, "params": dict(params)})
        if "search" in params:
            key = params.get("search", "")
        elif "search.semantic" in params:
            key = params.get("search.semantic", "")
        else:
            key = params.get("filter", "")
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


def _revalidate_query(planned: PlannedQuery | dict[str, Any]) -> PlannedQuery:
    """Revalidate a query so model_construct/mutated models cannot bypass."""
    if isinstance(planned, PlannedQuery):
        return PlannedQuery.model_validate(planned.model_dump())
    return PlannedQuery.model_validate(planned)


def _revalidate_plan(plan: QueryPlan | dict[str, Any]) -> QueryPlan:
    """Revalidate a plan so model_construct/mutated models cannot bypass."""
    if isinstance(plan, QueryPlan):
        return QueryPlan.model_validate(plan.model_dump())
    return QueryPlan.model_validate(plan)


def _seed_short_id(seed_work_id: str) -> str:
    """Short OpenAlex ID (``W123``) from a canonical work URL."""
    short = seed_work_id.rsplit("/", 1)[-1].strip()
    if not short.startswith("W") or not short[1:].isdigit():
        raise ValueError("seed_work_id must be canonical https://openalex.org/W<digits>")
    return short


def build_request(
    planned: PlannedQuery,
    api_key: str | None = None,
    mailto: str | None = None,
    timeout: float = DEFAULT_TIMEOUT_S,
) -> tuple[str, dict[str, str], dict[str, str], float]:
    """Build (url, params, headers, timeout) respecting OpenAlex constraints.

    - `search.semantic` carries natural-language research questions.
      `search` carries Boolean keyword/recent searches. Never send both.
    - Citation kinds use the same ``/works`` endpoint with a directional
      filter only: ``references`` -> ``filter=cited_by:<WID>`` (outgoing
      referenced works); ``citations`` -> ``filter=cites:<WID>``
      (incoming citing works). No ``search`` key is sent for citations.
      An optional ``from_publication_date`` is comma-combined.
    - Date filters apply only when requested; historical agenda branches
      have none. Only newest-first queries send an explicit sort.
    - `per-page` is strictly validated (1..50); `select` bounds payload;
      `mailto` polite pool. No caller-controlled URL or filter fields:
      the URL is always :data:`BASE_URL` and filters are constructed here.
    """
    planned = _revalidate_query(planned)
    timeout = _check_timeout(timeout)
    per_page = int(planned.per_page)
    params: dict[str, str] = {
        "per-page": str(per_page),
        "select": SELECT_FIELDS,
    }
    if planned.kind in CITATION_KINDS:
        assert planned.seed_work_id is not None
        short = _seed_short_id(planned.seed_work_id)
        direction = "cited_by" if planned.kind == "references" else "cites"
        filt = f"{direction}:{short}"
        if planned.from_date:
            filt += f",from_publication_date:{planned.from_date}"
        params["filter"] = filt
    else:
        search_key = "search.semantic" if planned.kind == "semantic" else "search"
        limit = MAX_SEMANTIC_CHARS if planned.kind == "semantic" else MAX_TERM_CHARS
        terms = planned.terms.strip()
        if not terms or len(terms) > limit:
            raise ValueError("planned query is empty or exceeds its character bound")
        params[search_key] = terms
        if planned.from_date:
            params["filter"] = (f"publication_year:{planned.from_date[:4]}-"
                                if planned.kind == "semantic" else
                                f"from_publication_date:{planned.from_date}")
        if planned.kind == "recent":
            if not planned.from_date:
                raise ValueError("recent query requires from_date")
            params["sort"] = "publication_date:desc"
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
    publication_date = raw.get("publication_date")
    try:
        if not isinstance(publication_date, str) or len(publication_date) != 10:
            publication_date = None
        else:
            if _dt.date.fromisoformat(publication_date).isoformat() != publication_date:
                publication_date = None
    except ValueError:
        publication_date = None
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
    mq = _safe_str(matched_query, MAX_SEMANTIC_CHARS if query_kind == "semantic" else MAX_TERM_CHARS)
    return CollectedWork(
        openalex_id=wid,
        title=title,
        abstract=abstract,
        publication_year=year,
        publication_date=publication_date,
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


def _matched_label(planned: PlannedQuery) -> str:
    """Provenance label for a query: terms, or seed for citation kinds."""
    if planned.kind in CITATION_KINDS:
        return (planned.seed_work_id or "").strip()
    return planned.terms


def _merge_provenance(target: CollectedWork, source: CollectedWork) -> None:
    """Dedup: merge provenance deterministically, keep first text."""
    for q in source.matched_queries:
        if q and q not in target.matched_queries:
            if len(target.matched_queries) < MAX_QUERIES:
                target.matched_queries.append(q)
    for k in source.query_kinds:
        if k and k not in target.query_kinds:
            if len(target.query_kinds) < MAX_QUERIES:
                target.query_kinds.append(k)
    for match in source.discovery_matches:
        if match not in target.discovery_matches:
            if len(target.discovery_matches) < MAX_QUERIES:
                target.discovery_matches.append(match)


def collect_with_feedback(
    plan: QueryPlan,
    transport: OpenAlexTransport,
    *,
    existing: _Sequence[CollectedWork] = (),
    known_work_ids: _Sequence[str] = (),
    api_key: str | None = None,
    mailto: str | None = None,
    timeout: float = DEFAULT_TIMEOUT_S,
    max_total: int = MAX_TOTAL_WORKS,
) -> Any:
    """Execute a bounded plan and report observed per-query retrieval.

    Shares one global per-run pool through ``existing``: caller passes the
    previous wave's works so dedup/provenance spans waves. ``known_work_ids``
    is the persistent library (not quality-filtered). Counts are observed
    metadata availability, never scientific quality:

    - ``returned``: bounded raw page length (``results[:per_page]``).
    - ``accepted``: distinct normalized/date-valid IDs for this query.
    - ``new_to_run``: accepted IDs absent from ``existing`` and earlier
      queries this run.
    - ``new_to_library``: accepted IDs absent from ``known_work_ids``.
    - ``with_abstract``/``with_pdf``: normalized availability among the
      accepted distinct IDs.
    - ``status``: ``ok``, ``malformed`` (invalid envelope), or
      ``skipped_budget`` (unexecuted because the pool cap was reached).

    At most 12 queries execute per call; upstream validates that one source
    round uses at most 6. Works are unscored (``score`` stays ``0.0``);
    density/citation counts never imply importance.

    Pacing: OpenAlex's semantic endpoint permits one request/second. State
    is stored per transport instance (``_last_semantic_completion`` owned by
    :class:`HttpxTransport`, :class:`RetryingTransport`,
    :class:`DictTransport`). Reuse the same transport instance — normally
    the same :class:`RetryingTransport` — across waves so pacing holds.
    If a custom transport uses ``__slots__`` and cannot store the
    attribute, wrap it once in :class:`SemanticPacingAdapter` and reuse
    that adapter. No global pacing state is kept.
    """
    from radar.schema.discovery import QueryFeedback, RetrievalResult

    plan = _revalidate_plan(plan)
    if not plan.queries:
        raise ValueError("query plan must contain at least one query")
    timeout = _check_timeout(timeout)
    max_total = max(1, min(int(max_total), MAX_TOTAL_WORKS))
    by_id: dict[str, CollectedWork] = {}
    for item in existing:
        copy = CollectedWork.model_validate(
            item.model_dump() if isinstance(item, CollectedWork) else item
        )
        if copy.openalex_id not in by_id:
            by_id[copy.openalex_id] = copy
    known_set = {str(wid) for wid in known_work_ids}
    seen_run: set[str] = set(by_id.keys())
    last_semantic_completion = _get_stored_semantic_completion(transport)
    feedback: list[Any] = []
    queries = plan.queries[:MAX_QUERIES]
    for index, planned in enumerate(queries):
        if len(by_id) >= max_total:
            for remaining in queries[index:]:
                feedback.append(
                    QueryFeedback(
                        query=remaining,
                        returned=0,
                        accepted=0,
                        new_to_run=0,
                        new_to_library=0,
                        with_abstract=0,
                        with_pdf=0,
                        status="skipped_budget",
                    )
                )
            break
        url, params, headers, _ = build_request(
            planned, api_key=api_key, mailto=mailto, timeout=timeout
        )
        if planned.kind == "semantic":
            now = _time.monotonic()
            if last_semantic_completion is not None:
                _time.sleep(max(0.0, 1.0 - (now - last_semantic_completion)))
        payload = transport.get_json(url, params, headers, timeout)
        if planned.kind == "semantic":
            last_semantic_completion = _time.monotonic()
            _store_semantic_completion(transport, last_semantic_completion)
        if not isinstance(payload, dict):
            feedback.append(
                QueryFeedback(
                    query=planned,
                    returned=0,
                    accepted=0,
                    new_to_run=0,
                    new_to_library=0,
                    with_abstract=0,
                    with_pdf=0,
                    status="malformed",
                )
            )
            continue
        results = payload.get("results")
        if not isinstance(results, list):
            feedback.append(
                QueryFeedback(
                    query=planned,
                    returned=0,
                    accepted=0,
                    new_to_run=0,
                    new_to_library=0,
                    with_abstract=0,
                    with_pdf=0,
                    status="malformed",
                )
            )
            continue
        raw_page = results[: int(planned.per_page)]
        returned = len(raw_page)
        per_query: dict[str, CollectedWork] = {}
        label = _matched_label(planned)
        for raw in raw_page:
            work = normalize_work(raw, label, planned.kind)
            if work is None:
                continue
            if planned.kind == "semantic" and planned.from_date:
                # Semantic API rejects day-level date filters. Fetch its
                # supported year range, then enforce the exact lower bound.
                if not work.publication_date or work.publication_date < planned.from_date:
                    continue
            if planned.question_id is not None:
                work.discovery_matches = [
                    DiscoveryMatch(
                        question_id=planned.question_id, role=planned.role
                    )
                ]
            if work.openalex_id not in per_query:
                per_query[work.openalex_id] = work
        accepted = len(per_query)
        with_abstract = sum(1 for w in per_query.values() if w.abstract.strip())
        with_pdf = sum(
            1 for w in per_query.values() if any(loc.pdf_url for loc in w.locations)
        )
        new_to_run = sum(1 for wid in per_query if wid not in seen_run)
        new_to_library = sum(1 for wid in per_query if wid not in known_set)
        for wid, work in per_query.items():
            target = by_id.get(wid)
            if target is None:
                if len(by_id) < max_total:
                    by_id[wid] = work
                else:
                    # Pool cap reached mid-page: counts already reflect the
                    # observed page; only the stored pool is bounded.
                    pass
            else:
                _merge_provenance(target, work)
        seen_run.update(per_query.keys())
        feedback.append(
            QueryFeedback(
                query=planned,
                returned=returned,
                accepted=accepted,
                new_to_run=new_to_run,
                new_to_library=new_to_library,
                with_abstract=with_abstract,
                with_pdf=with_pdf,
                status="ok",
            )
        )
    return RetrievalResult(
        works=list(by_id.values())[:max_total], feedback=feedback
    )


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

    Delegates to :func:`collect_with_feedback` with an empty pool so both
    entry points share one executor, pacing, and provenance policy.
    Reuse the same transport instance across waves for cross-wave 1/s
    semantic pacing (see :func:`collect_with_feedback`).
    """
    result = collect_with_feedback(
        plan,
        transport,
        existing=(),
        known_work_ids=(),
        api_key=api_key,
        mailto=mailto,
        timeout=timeout,
        max_total=max_total,
    )
    return list(result.works)
