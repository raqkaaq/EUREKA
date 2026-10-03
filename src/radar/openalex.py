"""OpenAlex collection seam: query-plan building, request construction,
abstract reconstruction, normalization, dedup, provenance, pre-scoring.

Constraints honored:
- stdlib + pydantic only, no LLM calls;
- strict timeouts; bounded requests/results;
- optional OPENALEX_API_KEY;
- injectable HTTP transport for tests;
- all remote text treated as untrusted (coerced, truncated, never eval'd).
"""

from __future__ import annotations

import datetime as _dt
import json as _json
import os as _os
import urllib.parse as _parse
import urllib.request as _request
from typing import Any, Protocol

from radar.models import (
    CollectedWork,
    LocationInfo,
    PlannedQuery,
    QueryPlan,
    RadarProfile,
)

BASE_URL = "https://api.openalex.org/works"
DEFAULT_TIMEOUT_S = 10.0
MAX_TIMEOUT_S = 30.0
MAX_TOTAL_WORKS = 200
MAX_ABSTRACT_CHARS = 20000
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


class UrllibTransport:
    """Default stdlib transport with strict timeouts."""

    def get_json(self, url, params, headers, timeout):
        timeout = _check_timeout(timeout)
        qs = _parse.urlencode(params)
        full = f"{url}?{qs}" if qs else url
        req = _request.Request(full, headers=headers or {}, method="GET")
        with _request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            raw = resp.read(2_000_000)  # bound: ~2MB cap per response
        try:
            data = _json.loads(raw.decode("utf-8", errors="replace"))
        except (ValueError, UnicodeError) as exc:
            raise ValueError(f"OpenAlex returned non-JSON payload: {exc}") from exc
        if not isinstance(data, dict):
            raise ValueError("OpenAlex returned malformed envelope (not an object)")
        return data


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
# Query plan + request construction
# ---------------------------------------------------------------------------


def build_query_plan(profile: RadarProfile, max_queries: int = 4) -> QueryPlan:
    """Build a bounded plan: semantic queries + one recency-biased query.

    Every query carries ``from_date`` derived from ``profile.lookback_days``
    so the configured lookback window applies to the whole plan. Semantic
    queries preserve OpenAlex relevance ranking (no ``sort`` param); only
    the recent query sorts by ``publication_date:desc`` (see
    :func:`build_request`).
    """
    keywords = [k for k in profile.keywords if k]
    if not keywords:
        raise ValueError("profile must contain at least one non-empty keyword")
    n = max(1, min(int(max_queries), 6))
    from_date = (
        _dt.date.today() - _dt.timedelta(days=profile.lookback_days)
    ).isoformat()
    # Semantic queries: one per keyword, capped.
    queries: list[PlannedQuery] = []
    for kw in keywords[:n]:
        terms = kw if not profile.domains else f"{kw} {' '.join(profile.domains[:2])}"
        queries.append(
            PlannedQuery(
                kind="semantic", terms=terms[:300], per_page=25, from_date=from_date
            )
        )
        if len(queries) >= n - 1 and n > 1:
            break
    # Recent-keyword query: newest-first slice over the lookback window.
    if n > 1 or not queries:
        recent_terms = " ".join(keywords[:3])[:300]
        queries.append(
            PlannedQuery(
                kind="recent",
                terms=recent_terms,
                per_page=25,
                from_date=from_date,
            )
        )
    queries = queries[:n]
    if not queries:
        raise ValueError("could not build a non-empty query plan")
    summary = f"keywords={len(keywords)} domains={len(profile.domains)}"
    return QueryPlan(queries=queries, profile_summary=summary)


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
    per_page = max(1, min(int(planned.per_page), 50))
    params: dict[str, str] = {
        "search": planned.terms.strip()[:300],
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
    if not wid or "openalex.org" not in wid:
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
# Cheap deterministic pre-scoring
# ---------------------------------------------------------------------------


def cheap_score(work: CollectedWork, keywords: list[str]) -> float:
    """Deterministic keyword-overlap score in [0, ~1.3]."""
    hay = f"{work.title}\n{work.abstract}".lower()
    kws = [k.lower().strip() for k in keywords if isinstance(k, str) and k.strip()]
    if not kws or not hay.strip():
        base = 0.0
    else:
        hits = sum(1 for k in kws[:20] if k in hay)
        base = hits / max(1, len(kws[:20]))
    # Tiny deterministic tie-breakers (no randomness, no network).
    cite_bonus = min(0.2, (work.cited_by_count or 0) / 5000.0)
    recency_bonus = 0.0
    if work.publication_year:
        try:
            age = _dt.date.today().year - int(work.publication_year)
            recency_bonus = max(0.0, min(0.1, 0.1 - 0.01 * max(0, age)))
        except (ValueError, TypeError):
            recency_bonus = 0.0
    return round(base + cite_bonus + recency_bonus, 6)


# ---------------------------------------------------------------------------
# Collection orchestration
# ---------------------------------------------------------------------------


def collect(
    plan: QueryPlan,
    transport: OpenAlexTransport,
    keywords_for_scoring: list[str] | None = None,
    api_key: str | None = None,
    mailto: str | None = None,
    timeout: float = DEFAULT_TIMEOUT_S,
    max_total: int = MAX_TOTAL_WORKS,
) -> list[CollectedWork]:
    """Execute the whole bounded plan via transport; dedup; pre-score.

    ``max_total`` is the hard candidate-pool cap (≤200), not the final
    output bound: every planned query is executed in order until the pool
    reaches the cap, then the pooled works are ranked by score. Callers
    apply the user-facing ``--max-candidates`` bound as a final slice of
    the returned ranked pool so later query branches still contribute when
    the output bound is small.
    """
    if not plan.queries:
        raise ValueError("query plan must contain at least one query")
    timeout = _check_timeout(timeout)
    max_total = max(1, min(int(max_total), MAX_TOTAL_WORKS))
    scoring_kws = keywords_for_scoring or []
    by_id: dict[str, CollectedWork] = {}
    for planned in plan.queries[:6]:
        url, params, headers, _ = build_request(
            planned, api_key=api_key, mailto=mailto, timeout=timeout
        )
        payload = transport.get_json(url, params, headers, timeout)
        if not isinstance(payload, dict):
            continue  # malformed envelope -> skip page, keep others
        results = payload.get("results")
        if not isinstance(results, list):
            continue
        for raw in results[:50]:  # bound per-query results
            work = normalize_work(raw, planned.terms, planned.kind)
            if work is None:
                continue
            existing = by_id.get(work.openalex_id)
            if existing is None:
                work.score = cheap_score(work, scoring_kws)
                by_id[work.openalex_id] = work
            else:
                # Dedup: merge provenance deterministically, keep first text.
                for q in work.matched_queries:
                    if q and q not in existing.matched_queries:
                        existing.matched_queries.append(q)
                for k in work.query_kinds:
                    if k and k not in existing.query_kinds:
                        existing.query_kinds.append(k)
                merged = cheap_score(existing, scoring_kws)
                existing.score = merged
            if len(by_id) >= max_total:
                break
        if len(by_id) >= max_total:
            break
    ranked = sorted(
        by_id.values(), key=lambda w: (-w.score, w.openalex_id)
    )[:max_total]
    return ranked
