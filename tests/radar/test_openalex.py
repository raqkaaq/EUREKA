"""Unit tests for the OpenAlex radar collector (injectable transport only,
no network, no LLM).
"""

from __future__ import annotations

import unittest

from pydantic import ValidationError

from radar.config.interests import RadarProfile
from radar.processing.ranking import cheap_score, rank_works
from radar.schema.papers import PlannedQuery, QueryPlan
from radar.source.openalex import (
    BASE_URL,
    DictTransport,
    build_request,
    collect,
    collect_with_feedback,
    normalize_work,
    reconstruct_abstract,
)


def _work(
    wid: str = "https://openalex.org/W1",
    title: str = "Diffusion models for protein design",
    index: dict | None = None,
    year: int | None = 2024,
    locations: bool = True,
) -> dict:
    raw: dict = {
        "id": wid,
        "title": title,
        "abstract_inverted_index": (
            index if index is not None else {"Diffusion": [0], "proteins": [1]}
        ),
        "doi": "https://doi.org/10.1234/x",
        "publication_year": year,
        "cited_by_count": 42,
    }
    if locations:
        raw["primary_location"] = {
            "landing_page_url": "https://example.org/paper",
            "pdf_url": "https://example.org/paper.pdf",
            "is_oa": True,
            "source": {"display_name": "Example Journal"},
        }
        raw["best_oa_location"] = {
            "landing_page_url": "https://arxiv.org/abs/2401.00001",
            "pdf_url": "https://arxiv.org/pdf/2401.00001",
            "is_oa": True,
            "source": {"display_name": "arXiv"},
        }
    return raw


class TestQueryPlan(unittest.TestCase):
    def test_explicit_plan_ok(self):
        plan = QueryPlan(
            queries=[
                PlannedQuery(kind="semantic", terms="diffusion models"),
                PlannedQuery(kind="keyword", terms="protein design"),
                PlannedQuery(kind="recent", terms="ml", from_date="2025-01-01"),
            ]
        )
        self.assertEqual(len(plan.queries), 3)
        kinds = {q.kind for q in plan.queries}
        self.assertIn("semantic", kinds)
        self.assertIn("keyword", kinds)

    def test_invalid_empty_profile_rejected(self):
        with self.assertRaises((ValidationError, ValueError)):
            RadarProfile(keywords=[])
        with self.assertRaises((ValidationError, ValueError)):
            RadarProfile(keywords=["   "])
        with self.assertRaises((ValidationError, ValueError)):
            QueryPlan(queries=[])
        with self.assertRaises((ValidationError, ValueError)):
            PlannedQuery(kind="semantic", terms="   ")
        with self.assertRaises(ValueError):
            QueryPlan.model_validate({"queries": [], "profile_summary": ""})

    def test_empty_plan_collect_raises(self):
        plan = QueryPlan.model_construct(queries=[])
        with self.assertRaises(ValueError):
            collect(plan, DictTransport({}))


class TestPlannedQueryContracts(unittest.TestCase):
    def test_citation_terms_default_empty_allowed(self):
        q = PlannedQuery(kind="citations", seed_work_id="https://openalex.org/W123")
        self.assertEqual(q.terms, "")
        q2 = PlannedQuery(
            kind="references", terms="", seed_work_id="https://openalex.org/W1"
        )
        self.assertEqual(q2.terms, "")

    def test_empty_terms_rejected_for_noncitation(self):
        with self.assertRaises(ValidationError):
            PlannedQuery(kind="keyword", terms="")
        with self.assertRaises(ValidationError):
            PlannedQuery(kind="semantic", terms="   ")
        with self.assertRaises(ValidationError):
            PlannedQuery(kind="recent", terms="", from_date="2025-01-01")

    def test_seed_mandatory_for_citations_forbidden_otherwise(self):
        with self.assertRaises(ValidationError):
            PlannedQuery(kind="citations")
        with self.assertRaises(ValidationError):
            PlannedQuery(kind="references", terms="x")
        with self.assertRaises(ValidationError):
            PlannedQuery(
                kind="keyword", terms="x",
                seed_work_id="https://openalex.org/W1",
            )
        with self.assertRaises(ValidationError):
            PlannedQuery(
                kind="semantic", terms="x",
                seed_work_id="https://openalex.org/W1",
            )

    def test_seed_must_be_canonical(self):
        with self.assertRaises(ValidationError):
            PlannedQuery(kind="citations", seed_work_id="https://example.com/W1")
        with self.assertRaises(ValidationError):
            PlannedQuery(kind="citations", seed_work_id="W123")
        with self.assertRaises(ValidationError):
            PlannedQuery(
                kind="citations", seed_work_id="https://openalex.org/W123?x=1"
            )
        with self.assertRaises(ValidationError):
            PlannedQuery(
                kind="citations", seed_work_id="https://openalex.org/S123"
            )

    def test_per_page_strict(self):
        with self.assertRaises(ValidationError):
            PlannedQuery(kind="keyword", terms="x", per_page=True)  # type: ignore
        with self.assertRaises(ValidationError):
            PlannedQuery(kind="keyword", terms="x", per_page=5.0)  # type: ignore
        with self.assertRaises(ValidationError):
            PlannedQuery(kind="keyword", terms="x", per_page="25")  # type: ignore
        with self.assertRaises(ValidationError):
            PlannedQuery(kind="keyword", terms="x", per_page=99)
        with self.assertRaises(ValidationError):
            PlannedQuery(kind="keyword", terms="x", per_page=0)
        ok = PlannedQuery(kind="keyword", terms="x", per_page=50)
        self.assertEqual(ok.per_page, 50)

    def test_extra_forbidden(self):
        with self.assertRaises(ValidationError):
            PlannedQuery.model_validate(
                {"kind": "keyword", "terms": "x", "url": "https://evil.example"}
            )
        with self.assertRaises(ValidationError):
            QueryPlan.model_validate(
                {"queries": [{"kind": "keyword", "terms": "x"}],
                 "profile_summary": "", "url": "https://evil.example"}
            )

    def test_recent_requires_date(self):
        with self.assertRaises(ValidationError):
            PlannedQuery(kind="recent", terms="ml")
        ok = PlannedQuery(kind="recent", terms="ml", from_date="2025-01-01")
        self.assertEqual(ok.from_date, "2025-01-01")

    def test_roles_preserve_old_and_add_new(self):
        for role in ("foundation", "frontier", "counterevidence",
                     "cross_domain", "exploration"):
            q = PlannedQuery(
                kind="keyword", terms="x", question_id="q1", role=role  # type: ignore
            )
            self.assertEqual(q.role, role)
        with self.assertRaises(ValidationError):
            PlannedQuery(kind="keyword", terms="x", question_id="q1", role="other")  # type: ignore

    def test_source_revalidates_bypass(self):
        bypass = PlannedQuery.model_construct(
            kind="semantic", terms="diffusion models", per_page=99
        )
        with self.assertRaises(ValidationError):
            build_request(bypass)
        with self.assertRaises(ValidationError):
            collect(QueryPlan.model_construct(queries=[bypass]), DictTransport({}))
        mutated = PlannedQuery(kind="keyword", terms="alpha")
        mutated.terms = ""  # type: ignore
        with self.assertRaises(ValidationError):
            build_request(mutated)
        recent_bypass = PlannedQuery.model_construct(kind="recent", terms="ml")
        with self.assertRaises(ValidationError):
            build_request(recent_bypass)


class TestRequestConstruction(unittest.TestCase):
    def test_semantic_request_constraints(self):
        _, params, headers, timeout = build_request(
            PlannedQuery(kind="semantic", terms="diffusion models", per_page=50)
        )
        self.assertEqual(params["per-page"], "50")
        # No `sort` for semantic queries: OpenAlex relevance ranking is the
        # default when `search` is present (`sort=relevance` is rejected
        # with 400 by the live API — regression guard).
        self.assertNotIn("sort", params)
        self.assertNotIn("filter", params)
        self.assertIn("select", params)
        self.assertIn("User-Agent", headers)
        self.assertLessEqual(timeout, 30.0)

    def test_invalid_per_page_rejected_not_capped(self):
        bypass = PlannedQuery.model_construct(
            kind="semantic", terms="diffusion models", per_page=99
        )
        with self.assertRaises(ValidationError):
            build_request(bypass)

    def test_recent_request_has_date_filter(self):
        _, params, _, _ = build_request(
            PlannedQuery(kind="recent", terms="ml", from_date="2025-01-01")
        )
        self.assertIn("from_publication_date:2025-01-01", params["filter"])
        self.assertEqual(params["sort"], "publication_date:desc")

    def test_api_key_attached(self):
        _, params, _, _ = build_request(
            PlannedQuery(kind="semantic", terms="ml"), api_key="secret123"
        )
        self.assertEqual(params["api_key"], "secret123")

    def test_bad_timeout_rejected(self):
        with self.assertRaises(ValueError):
            build_request(
                PlannedQuery(kind="semantic", terms="ml"), timeout=999
            )


class TestCitationRequests(unittest.TestCase):
    def test_references_direction(self):
        url, params, _, _ = build_request(
            PlannedQuery(
                kind="references", seed_work_id="https://openalex.org/W123"
            )
        )
        self.assertEqual(url, BASE_URL)
        self.assertEqual(url, "https://api.openalex.org/works")
        self.assertEqual(params["filter"], "cited_by:W123")
        self.assertNotIn("search", params)
        self.assertNotIn("search.semantic", params)
        self.assertNotIn("sort", params)

    def test_citations_direction(self):
        url, params, _, _ = build_request(
            PlannedQuery(
                kind="citations", seed_work_id="https://openalex.org/W999"
            )
        )
        self.assertEqual(url, BASE_URL)
        self.assertEqual(params["filter"], "cites:W999")
        self.assertNotIn("search", params)
        self.assertNotIn("search.semantic", params)

    def test_citation_from_date_combined(self):
        _, params, _, _ = build_request(
            PlannedQuery(
                kind="citations",
                seed_work_id="https://openalex.org/W5",
                from_date="2024-03-01",
            )
        )
        self.assertEqual(
            params["filter"], "cites:W5,from_publication_date:2024-03-01"
        )

    def test_citation_no_injected_seed_or_arbitrary_fields(self):
        url, params, _, _ = build_request(
            PlannedQuery(
                kind="references", seed_work_id="https://openalex.org/W7"
            )
        )
        self.assertEqual(url, "https://api.openalex.org/works")
        # Seed appears only inside the constructed filter, never as its own
        # param and never as a caller-controlled URL.
        self.assertNotIn("seed_work_id", params)
        self.assertNotIn("seed", params)
        self.assertNotIn("url", params)
        for value in params.values():
            self.assertNotIn("https://openalex.org/W7", value.replace("cited_by:W7", ""))
            self.assertNotIn("https://", value.replace("select", "") if "select" not in value else "")


class TestAbstractAndNormalize(unittest.TestCase):
    def test_reconstruct_orders_positions(self):
        text = reconstruct_abstract({"world": [1], "hello": [0]})
        self.assertEqual(text, "hello world")

    def test_reconstruct_malformed_inputs(self):
        self.assertEqual(reconstruct_abstract(None), "")
        self.assertEqual(reconstruct_abstract({}), "")
        self.assertEqual(reconstruct_abstract("nope"), "")
        self.assertEqual(reconstruct_abstract({"hi": "nope"}), "")
        self.assertEqual(reconstruct_abstract({"hi": [True, "x", -5]}), "")

    def test_normal_results(self):
        work = normalize_work(_work(), "diffusion models", "semantic")
        assert work is not None
        self.assertEqual(work.openalex_id, "https://openalex.org/W1")
        self.assertEqual(work.abstract, "Diffusion proteins")
        self.assertEqual(work.matched_queries, ["diffusion models"])
        self.assertGreaterEqual(len(work.locations), 1)
        # arXiv appears only as a returned location string.
        urls = [loc.landing_url for loc in work.locations]
        self.assertTrue(any("arxiv.org" in u for u in urls))

    def test_missing_abstracts_locations(self):
        raw = _work(index=None, locations=False)
        raw["abstract_inverted_index"] = None
        work = normalize_work(raw, "q", "recent")
        assert work is not None
        self.assertEqual(work.abstract, "")
        self.assertEqual(work.locations, [])

    def test_malformed_data_skipped(self):
        self.assertIsNone(normalize_work(None, "q"))
        self.assertIsNone(normalize_work({"no": "id"}, "q"))
        self.assertIsNone(normalize_work({"id": 123}, "q"))
        self.assertIsNone(normalize_work("string", "q"))
        # Wrong-typed year/citations must not raise.
        raw = _work()
        raw["publication_year"] = "not-a-year"
        raw["cited_by_count"] = "lots"
        work = normalize_work(raw, "q")
        assert work is not None
        self.assertIsNone(work.publication_year)
        self.assertEqual(work.cited_by_count, 0)


class TestCollect(unittest.TestCase):
    def test_collect_normal_results(self):
        plan = QueryPlan(
            queries=[
                PlannedQuery(kind="semantic", terms="diffusion"),
                PlannedQuery(kind="keyword", terms="diffusion models"),
            ]
        )
        pages = {"diffusion": {"results": [_work()]},
                 "diffusion models": {"results": [_work()]}}
        works = rank_works(collect(plan, DictTransport(pages)), ["diffusion"])
        self.assertEqual(len(works), 1)
        self.assertGreaterEqual(works[0].score, 0.0)

    def test_duplicate_works_across_queries_merged(self):
        plan = QueryPlan(
            queries=[
                PlannedQuery(kind="semantic", terms="alpha"),
                PlannedQuery(kind="recent", terms="beta", from_date="2025-01-01"),
            ]
        )
        pages = {
            "alpha": {"results": [_work()]},
            "beta": {"results": [_work()]},
        }
        works = rank_works(collect(plan, DictTransport(pages)), ["diffusion"])
        self.assertEqual(len(works), 1)
        self.assertEqual(
            sorted(works[0].matched_queries), ["alpha", "beta"]
        )
        self.assertEqual(
            sorted(works[0].query_kinds), ["recent", "semantic"]
        )

    def test_collect_skips_malformed_pages_and_entries(self):
        plan = QueryPlan(queries=[PlannedQuery(kind="semantic", terms="alpha")])
        pages = {
            "alpha": {
                "results": [
                    {"id": "junk-without-openalex-marker"},
                    "not-a-dict",
                    _work(),
                ]
            }
        }
        works = rank_works(collect(plan, DictTransport(pages)), ["diffusion"])
        self.assertEqual(len(works), 1)

    def test_collect_delegates_to_feedback_executor(self):
        plan = QueryPlan(
            queries=[PlannedQuery(kind="keyword", terms="alpha")]
        )
        pages = {"alpha": {"results": [_work()]}}
        t1 = DictTransport(dict(pages))
        t2 = DictTransport(dict(pages))
        works = collect(plan, t1)
        result = collect_with_feedback(plan, t2)
        self.assertEqual([w.openalex_id for w in works],
                         [w.openalex_id for w in result.works])
        self.assertEqual(len(result.feedback), 1)
        self.assertEqual(result.feedback[0].status, "ok")
        for w in works:
            self.assertEqual(w.score, 0.0)

    def test_prescore_deterministic(self):
        work = normalize_work(_work(), "q")
        assert work is not None
        s1 = cheap_score(work, ["diffusion", "protein"])
        s2 = cheap_score(work, ["diffusion", "protein"])
        self.assertEqual(s1, s2)
        s_other = cheap_score(work, ["unrelated-topic-zzz"])
        self.assertGreaterEqual(s1, s_other)


class TestLookbackAndBounds(unittest.TestCase):
    def test_semantic_year_filter_and_exact_local_date(self):
        query = PlannedQuery(
            kind="semantic", terms="causal learning", from_date="2026-07-06"
        )
        _, params, _, _ = build_request(query)
        self.assertEqual(params["filter"], "publication_year:2026-")
        rows = [{"id": f"https://openalex.org/W{i}", "publication_date": date}
                for i, date in enumerate(("2026-07-05", "2026-07-06", "2026-08-10", None,
                                          "invalid", "2026-W28-1"))]
        papers = collect(QueryPlan(queries=[query]), DictTransport({query.terms: {"results": rows}}))
        self.assertEqual([p.openalex_id for p in papers],
                         ["https://openalex.org/W1", "https://openalex.org/W2"])

    def test_later_query_branches_contribute_with_small_output_max(self):
        # First branch: low-scoring filler; last branch: keyword hit that
        # must win the ranked top slice even when the output bound is tiny.
        scoring_kw = "qubit-lattice-zzz"
        plan = QueryPlan(
            queries=[
                PlannedQuery(
                    kind="keyword", terms="alpha", from_date="2026-06-22"
                ),
                PlannedQuery(
                    kind="keyword", terms="beta", from_date="2026-06-22"
                ),
                PlannedQuery(
                    kind="recent", terms="beta", from_date="2026-06-22"
                ),
            ]
        )

        def filler(i: int) -> dict:
            raw = _work(
                wid=f"https://openalex.org/W{i}",
                title="Unrelated filler topic",
                index={"Unrelated": [0], "filler": [1]},
                year=2024,
            )
            raw["cited_by_count"] = 0
            return raw

        hit = _work(
            wid="https://openalex.org/WLATE",
            title=f"Critical advance in {scoring_kw} methods",
            index={"Critical": [0], "advance": [1], scoring_kw: [2]},
            year=2026,
        )
        pages = {
            "alpha": {"results": [filler(1), filler(2), filler(3)]},
            "beta": {"results": [hit]},
        }
        transport = DictTransport(pages)
        pool = rank_works(
            collect(plan, transport, max_total=200), [scoring_kw])
        # Whole bounded plan executed despite the small output bound.
        self.assertEqual(len(transport.calls), len(plan.queries))
        self.assertEqual(len(pool), 4)
        # Final output bound is a slice of the ranked pool: the late-branch
        # hit ranks first and survives a --max-candidates 1 slice.
        top1 = pool[:1]
        self.assertEqual(top1[0].openalex_id, "https://openalex.org/WLATE")
        self.assertIn("beta", top1[0].matched_queries)


def _mock_transport(status: int, headers: dict | None = None,
                    body: bytes = b"") -> "HttpxTransport":
    """HttpxTransport backed by an httpx.MockTransport (no network)."""
    import httpx

    from radar.source.openalex import HttpxTransport

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, headers=headers or {}, content=body,
                              request=request)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    return HttpxTransport(client=client)


def _quota_body() -> bytes:
    return (
        b'{"error":"Rate limit exceeded","message":"Insufficient budget. '
        b'This request has no API key ($0 remaining; resets at midnight UTC)."}'
    )


def _quota_headers(retry_after: str = "79188") -> dict:
    return {
        "Retry-After": retry_after,
        "X-RateLimit-Limit": "1000",
        "X-RateLimit-Remaining": "0",
        "X-RateLimit-Reset": retry_after,
    }


class TestDailyQuota(unittest.TestCase):
    """Confirmed daily-budget 429s fail fast with a typed non-HTTP error.

    Transient 429s stay redacted transport errors so the bounded retry
    wrapper still handles them; other statuses are untouched. No network.
    """

    def _get_json_raising(self, status: int, headers: dict | None,
                          body: bytes) -> BaseException:
        try:
            _mock_transport(status, headers, body).get_json(
                "https://api.openalex.org/works",
                {"search": "x", "api_key": "SECRET-KEY-123"},
                {},
                10.0,
            )
        except BaseException as got:  # noqa: BLE001
            return got
        raise AssertionError("get_json did not raise")

    def test_confirmed_quota_raises_typed_error(self):
        import httpx

        from radar.source.openalex import OpenAlexQuotaError

        got = self._get_json_raising(429, _quota_headers(), _quota_body())
        self.assertIsInstance(got, OpenAlexQuotaError)
        self.assertNotIsInstance(got, httpx.HTTPError)

    def test_quota_message_actionable_and_redacted(self):
        got = self._get_json_raising(429, _quota_headers(), _quota_body())
        text = str(got)
        self.assertIn("OPENALEX_API_KEY", text)
        self.assertNotIn("SECRET-KEY-123", text)
        self.assertNotIn("api.openalex.org/works?search=x", text)
        self.assertNotIn("Insufficient budget. This request", text)

    def test_headers_alone_confirm_quota(self):
        from radar.source.openalex import OpenAlexQuotaError

        got = self._get_json_raising(429, _quota_headers(), b"")
        self.assertIsInstance(got, OpenAlexQuotaError)

    def test_transient_429_stays_http_error(self):
        from radar.source.openalex import OpenAlexHttpError

        got = self._get_json_raising(429, {"Retry-After": "2"}, b"busy, try again")
        self.assertIsInstance(got, OpenAlexHttpError)
        self.assertEqual(got.status_code, 429)

    def test_zero_remaining_short_retry_not_quota_without_body(self):
        from radar.source.openalex import OpenAlexHttpError

        headers = {"Retry-After": "5", "X-RateLimit-Remaining": "0"}
        got = self._get_json_raising(429, headers, b"")
        self.assertIsInstance(got, OpenAlexHttpError)
        self.assertEqual(got.status_code, 429)

    def test_malformed_body_transient_headers_stays_http_error(self):
        from radar.source.openalex import OpenAlexHttpError

        got = self._get_json_raising(429, {"Retry-After": "2"}, b"{not json###")
        self.assertIsInstance(got, OpenAlexHttpError)
        self.assertEqual(got.status_code, 429)

    def test_missing_headers_and_body_stays_http_error(self):
        from radar.source.openalex import OpenAlexHttpError

        got = self._get_json_raising(429, {}, b"")
        self.assertIsInstance(got, OpenAlexHttpError)
        self.assertEqual(got.status_code, 429)

    def test_non_429_status_untouched(self):
        from radar.source.openalex import OpenAlexHttpError

        got = self._get_json_raising(503, _quota_headers(), _quota_body())
        self.assertIsInstance(got, OpenAlexHttpError)
        self.assertEqual(got.status_code, 503)

    def test_quota_fail_fast_no_retry_sleep(self):
        from unittest.mock import patch

        from radar.source.openalex import OpenAlexQuotaError, RetryingTransport

        calls: list = []

        class QuotaInner:
            def get_json(self, url, params, headers, timeout):
                calls.append(1)
                raise OpenAlexQuotaError("quota exhausted")

        with patch("time.sleep") as asleep:
            with self.assertRaises(OpenAlexQuotaError):
                RetryingTransport(QuotaInner()).get_json("u", {}, {}, 1.0)
        asleep.assert_not_called()
        self.assertEqual(len(calls), 1)

    def test_transient_429_honors_server_retry_after(self):
        from unittest.mock import patch

        import httpx

        from radar.source.openalex import HttpxTransport, RetryingTransport

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(429, headers={"Retry-After": "30"},
                                  content=b"busy", request=request)

        client = httpx.Client(transport=httpx.MockTransport(handler))
        with patch("time.sleep") as asleep:
            with self.assertRaises(Exception):
                RetryingTransport(HttpxTransport(client=client),
                                  max_retries=1).get_json(
                    "https://api.openalex.org/works", {}, {}, 1.0)
        asleep.assert_called_once_with(30.0)

    def test_transient_429_still_retried(self):
        from unittest.mock import patch

        import httpx

        from radar.source.openalex import RetryingTransport

        calls: list = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(1)
            if len(calls) == 1:
                return httpx.Response(429, headers={"Retry-After": "1"},
                                      content=b"busy", request=request)
            return httpx.Response(200, content=b'{"results": []}',
                                  request=request)

        client = httpx.Client(transport=httpx.MockTransport(handler))
        from radar.source.openalex import HttpxTransport

        with patch("time.sleep"):
            out = RetryingTransport(HttpxTransport(client=client)).get_json(
                "https://api.openalex.org/works", {}, {}, 1.0)
        self.assertEqual(out, {"results": []})
        self.assertEqual(len(calls), 2)


class TestHttpxTransportConfig(unittest.TestCase):
    """Direct-network-call guarantees: timeouts, no proxy env, no redirects."""

    def test_success_returns_parsed_envelope(self):
        import httpx

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=b'{"results": [], "meta": {}}',
                                  request=request)

        client = httpx.Client(transport=httpx.MockTransport(handler))
        from radar.source.openalex import HttpxTransport

        out = HttpxTransport(client=client).get_json("https://u", {"a": "b"}, {}, 5.0)
        self.assertEqual(out, {"results": [], "meta": {}})

    def test_client_ignores_proxy_env_and_redirects(self):
        from radar.source.openalex import default_client

        client = default_client()
        try:
            self.assertFalse(client.trust_env)
            self.assertFalse(client.follow_redirects)
        finally:
            client.close()

    def test_timeout_surfaces_without_request_details(self):
        import httpx

        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectTimeout("slow", request=request)

        client = httpx.Client(transport=httpx.MockTransport(handler))
        from radar.source.openalex import HttpxTransport

        with self.assertRaises(Exception) as ctx:
            HttpxTransport(client=client).get_json(
                "https://api.openalex.org/works",
                {"search": "x", "api_key": "SECRET-KEY-123"}, {}, 5.0)
        self.assertNotIn("SECRET-KEY-123", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
