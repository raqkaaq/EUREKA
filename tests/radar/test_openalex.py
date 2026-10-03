"""Unit tests for the OpenAlex radar collector (injectable transport only,
no network, no LLM).
"""

from __future__ import annotations

import unittest

from pydantic import ValidationError

from radar.models import PlannedQuery, QueryPlan, RadarProfile
from radar.openalex import (
    DictTransport,
    build_query_plan,
    build_request,
    cheap_score,
    collect,
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
    def test_build_plan_ok(self):
        plan = build_query_plan(
            RadarProfile(keywords=["diffusion models", "protein design"])
        )
        self.assertGreaterEqual(len(plan.queries), 2)
        kinds = {q.kind for q in plan.queries}
        self.assertIn("semantic", kinds)
        self.assertIn("recent", kinds)
        recent = [q for q in plan.queries if q.kind == "recent"][0]
        self.assertIsNotNone(recent.from_date)

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
            build_query_plan(RadarProfile.model_construct(keywords=[]))

    def test_empty_plan_collect_raises(self):
        plan = QueryPlan.model_construct(queries=[])
        with self.assertRaises(ValueError):
            collect(plan, DictTransport({}))


class TestRequestConstruction(unittest.TestCase):
    def test_semantic_request_constraints(self):
        _, params, headers, timeout = build_request(
            PlannedQuery.model_construct(kind="semantic", terms="diffusion models", per_page=99)
        )
        self.assertEqual(params["per-page"], "50")  # capped
        # No `sort` for semantic queries: OpenAlex relevance ranking is the
        # default when `search` is present (`sort=relevance` is rejected
        # with 400 by the live API — regression guard).
        self.assertNotIn("sort", params)
        self.assertNotIn("filter", params)
        self.assertIn("select", params)
        self.assertIn("User-Agent", headers)
        self.assertLessEqual(timeout, 30.0)

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
        plan = build_query_plan(RadarProfile(keywords=["diffusion"]))
        terms = [q.terms for q in plan.queries]
        pages = {t: {"results": [_work()]} for t in terms}
        works = collect(plan, DictTransport(pages), ["diffusion"])
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
        works = collect(plan, DictTransport(pages), ["diffusion"])
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
        works = collect(plan, DictTransport(pages), ["diffusion"])
        self.assertEqual(len(works), 1)

    def test_prescore_deterministic(self):
        work = normalize_work(_work(), "q")
        assert work is not None
        s1 = cheap_score(work, ["diffusion", "protein"])
        s2 = cheap_score(work, ["diffusion", "protein"])
        self.assertEqual(s1, s2)
        s_other = cheap_score(work, ["unrelated-topic-zzz"])
        self.assertGreaterEqual(s1, s_other)


class TestLookbackAndBounds(unittest.TestCase):
    def test_semantic_queries_carry_lookback_filter(self):
        import datetime as _dt

        profile = RadarProfile(
            keywords=["diffusion models", "protein design"], lookback_days=90
        )
        plan = build_query_plan(profile)
        expected = (_dt.date.today() - _dt.timedelta(days=90)).isoformat()
        self.assertGreaterEqual(len(plan.queries), 2)
        for q in plan.queries:
            self.assertEqual(q.from_date, expected)
        for q in plan.queries:
            _, params, _, _ = build_request(q)
            self.assertIn(f"from_publication_date:{expected}", params["filter"])
            if q.kind == "semantic":
                # Relevance ranking preserved: no `sort` on semantic queries.
                self.assertNotIn("sort", params)
            else:
                self.assertEqual(params["sort"], "publication_date:desc")

    def test_later_query_branches_contribute_with_small_output_max(self):
        # First branch: low-scoring filler; last branch: keyword hit that
        # must win the ranked top slice even when the output bound is tiny.
        # With the old bug (pool cap == output max) only the first query
        # would ever execute, so the late-branch work could never surface.
        scoring_kw = "qubit-lattice-zzz"
        plan = QueryPlan(
            queries=[
                PlannedQuery(
                    kind="semantic", terms="alpha", from_date="2026-06-22"
                ),
                PlannedQuery(
                    kind="semantic", terms="beta", from_date="2026-06-22"
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
        pool = collect(plan, transport, [scoring_kw], max_total=200)
        # Whole bounded plan executed despite the small output bound.
        self.assertEqual(len(transport.calls), len(plan.queries))
        self.assertEqual(len(pool), 4)
        # Final output bound is a slice of the ranked pool: the late-branch
        # hit ranks first and survives a --max-candidates 1 slice.
        top1 = pool[:1]
        self.assertEqual(top1[0].openalex_id, "https://openalex.org/WLATE")
        self.assertIn("beta", top1[0].matched_queries)


if __name__ == "__main__":
    unittest.main()
