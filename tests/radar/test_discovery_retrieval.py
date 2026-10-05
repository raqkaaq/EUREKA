"""Adaptive retrieval executor: bounded OpenAlex waves with observed feedback.

All HTTP is faked at the transport boundary (no network). Counts describe
metadata availability only, never scientific quality.
"""

from __future__ import annotations

import unittest
from unittest import mock

from radar.schema.papers import CollectedWork, PlannedQuery, QueryPlan
from radar.source.openalex import (
    BASE_URL,
    DictTransport,
    RetryingTransport,
    OpenAlexHttpError,
    collect,
    collect_with_feedback,
)


def _raw(
    wid: str,
    title: str = "Study",
    abstract: dict | None = None,
    date: str | None = None,
    pdf: bool = False,
) -> dict:
    raw: dict = {
        "id": wid,
        "title": title,
        "abstract_inverted_index": abstract if abstract is not None else {"Study": [0]},
        "publication_year": 2024,
        "publication_date": date,
        "cited_by_count": 3,
    }
    if pdf:
        raw["primary_location"] = {
            "landing_page_url": "https://example.org/paper",
            "pdf_url": "https://example.org/paper.pdf",
            "is_oa": True,
            "source": {"display_name": "Example"},
        }
    else:
        raw["primary_location"] = {
            "landing_page_url": "https://example.org/paper",
            "is_oa": False,
            "source": {"display_name": "Example"},
        }
    return raw


class TestCitationRetrieval(unittest.TestCase):
    def test_citation_direction_and_params(self):
        plan = QueryPlan(
            queries=[
                PlannedQuery(
                    kind="references",
                    seed_work_id="https://openalex.org/W10",
                ),
                PlannedQuery(
                    kind="citations",
                    seed_work_id="https://openalex.org/W10",
                    from_date="2024-01-01",
                ),
            ]
        )
        pages = {
            "cited_by:W10": {"results": [_raw("https://openalex.org/W11")]},
            "cites:W10,from_publication_date:2024-01-01": {
                "results": [_raw("https://openalex.org/W12")]
            },
        }
        transport = DictTransport(pages)
        result = collect_with_feedback(plan, transport)
        self.assertEqual(
            [w.openalex_id for w in result.works],
            ["https://openalex.org/W11", "https://openalex.org/W12"],
        )
        self.assertEqual(len(result.feedback), 2)
        self.assertTrue(all(f.status == "ok" for f in result.feedback))
        # Same endpoint, directional filters, no search keys, no seed param.
        for call in transport.calls:
            self.assertEqual(call["url"], BASE_URL)
            self.assertEqual(call["url"], "https://api.openalex.org/works")
            self.assertNotIn("search", call["params"])
            self.assertNotIn("search.semantic", call["params"])
            self.assertNotIn("seed_work_id", call["params"])
        self.assertEqual(
            transport.calls[0]["params"]["filter"], "cited_by:W10"
        )
        self.assertEqual(
            transport.calls[1]["params"]["filter"],
            "cites:W10,from_publication_date:2024-01-01",
        )
        # Provenance uses the seed label; no scientific scores.
        for work in result.works:
            self.assertEqual(work.score, 0.0)
            self.assertIn("https://openalex.org/W10", work.matched_queries)

    def test_citation_rejects_bypass_and_wrong_direction(self):
        # Missing seed bypasses model validation but not the source boundary.
        bad = PlannedQuery.model_construct(kind="citations", terms="")
        with self.assertRaises(Exception):
            collect_with_feedback(
                QueryPlan.model_construct(queries=[bad]), DictTransport({})
            )
        # Seed forbidden for non-citations, even via bypass.
        bad2 = PlannedQuery.model_construct(
            kind="keyword", terms="x",
            seed_work_id="https://openalex.org/W1",
        )
        with self.assertRaises(Exception):
            collect_with_feedback(
                QueryPlan.model_construct(queries=[bad2]), DictTransport({})
            )


class TestCrossWaveMerge(unittest.TestCase):
    def test_duplicate_provenance_merges_across_waves(self):
        wave1 = QueryPlan(
            queries=[
                PlannedQuery(
                    kind="keyword", terms="alpha",
                    question_id="q1", role="foundation",
                )
            ]
        )
        shared = _raw("https://openalex.org/W1", title="Shared")
        t1 = DictTransport({"alpha": {"results": [dict(shared)]}})
        with mock.patch("radar.source.openalex._time.sleep"):
            r1 = collect_with_feedback(wave1, t1)
        self.assertEqual(len(r1.works), 1)

        wave2 = QueryPlan(
            queries=[
                PlannedQuery(
                    kind="keyword", terms="beta",
                    question_id="q1", role="counterevidence",
                ),
                PlannedQuery(
                    kind="citations",
                    seed_work_id="https://openalex.org/W1",
                    question_id="q1", role="exploration",
                ),
            ]
        )
        pages = {
            "beta": {"results": [dict(shared), _raw("https://openalex.org/W2")]},
            "cites:W1": {"results": [dict(shared)]},
        }
        t2 = DictTransport(pages)
        with mock.patch("radar.source.openalex._time.sleep"):
            r2 = collect_with_feedback(wave2, t2, existing=r1.works)
        ids = sorted(w.openalex_id for w in r2.works)
        self.assertEqual(ids, ["https://openalex.org/W1", "https://openalex.org/W2"])
        w1 = next(w for w in r2.works if w.openalex_id == "https://openalex.org/W1")
        # Provenance merged deterministically across waves.
        self.assertIn("alpha", w1.matched_queries)
        self.assertIn("beta", w1.matched_queries)
        self.assertIn("https://openalex.org/W1", w1.matched_queries)
        self.assertIn("keyword", w1.query_kinds)
        self.assertIn("citations", w1.query_kinds)
        roles = {(m.question_id, m.role) for m in w1.discovery_matches}
        self.assertIn(("q1", "foundation"), roles)
        self.assertIn(("q1", "counterevidence"), roles)
        self.assertIn(("q1", "exploration"), roles)
        # Second wave feedback: first query saw one old + one new.
        self.assertEqual(r2.feedback[0].accepted, 2)
        self.assertEqual(r2.feedback[0].new_to_run, 1)
        # Citation wave re-observed the shared work: no new ids this run.
        self.assertEqual(r2.feedback[1].accepted, 1)
        self.assertEqual(r2.feedback[1].new_to_run, 0)

    def test_existing_copies_not_mutated(self):
        plan = QueryPlan(queries=[PlannedQuery(kind="keyword", terms="beta")])
        seed_work = CollectedWork.model_validate(
            {
                "openalex_id": "https://openalex.org/W1",
                "title": "Seed",
                "matched_queries": ["alpha"],
                "query_kinds": ["keyword"],
                "score": 0.0,
            }
        )
        before = [w.model_dump() for w in [seed_work]]
        transport = DictTransport(
            {"beta": {"results": [_raw("https://openalex.org/W1")]}}
        )
        result = collect_with_feedback(plan, transport, existing=[seed_work])
        # Caller list untouched; stored pool holds merged copies.
        self.assertEqual([w.model_dump() for w in [seed_work]], before)
        self.assertEqual(seed_work.matched_queries, ["alpha"])
        merged = next(
            w for w in result.works if w.openalex_id == "https://openalex.org/W1"
        )
        self.assertIn("alpha", merged.matched_queries)
        self.assertIn("beta", merged.matched_queries)


class TestObservedFeedback(unittest.TestCase):
    def test_counts_distinguish_run_from_library(self):
        plan = QueryPlan(
            queries=[
                PlannedQuery(kind="keyword", terms="alpha"),
                PlannedQuery(kind="keyword", terms="beta"),
            ]
        )
        with_pdf = _raw(
            "https://openalex.org/W1",
            abstract={"Full": [0], "text": [1]},
            date="2024-05-01",
            pdf=True,
        )
        no_abs = dict(_raw("https://openalex.org/W2", date="2024-05-02"))
        no_abs["abstract_inverted_index"] = None
        pages = {
            "alpha": {"results": [with_pdf, no_abs, {"id": "junk"}, "nope"]},
            "beta": {"results": [dict(with_pdf), _raw("https://openalex.org/W3")]},
        }
        transport = DictTransport(pages)
        result = collect_with_feedback(
            plan, transport, known_work_ids=["https://openalex.org/W1"]
        )
        first, second = result.feedback
        # returned counts the bounded raw page including skipped entries.
        self.assertEqual(first.returned, 4)
        self.assertEqual(first.accepted, 2)
        self.assertEqual(first.new_to_run, 2)
        # W1 already in the library: only W2 is new to the library.
        self.assertEqual(first.new_to_library, 1)
        self.assertEqual(first.with_abstract, 1)
        self.assertEqual(first.with_pdf, 1)
        # Second query re-observes W1: accepted but not new to this run.
        self.assertEqual(second.accepted, 2)
        self.assertEqual(second.new_to_run, 1)
        self.assertEqual(second.new_to_library, 1)  # W3 only
        self.assertEqual(second.status, "ok")

    def test_returned_bounded_by_per_page(self):
        plan = QueryPlan(
            queries=[PlannedQuery(kind="keyword", terms="alpha", per_page=2)]
        )
        pages = {
            "alpha": {
                "results": [
                    _raw("https://openalex.org/W1"),
                    _raw("https://openalex.org/W2"),
                    _raw("https://openalex.org/W3"),
                    _raw("https://openalex.org/W4"),
                ]
            }
        }
        result = collect_with_feedback(plan, DictTransport(pages))
        self.assertEqual(result.feedback[0].returned, 2)
        self.assertEqual(result.feedback[0].accepted, 2)
        self.assertEqual(len(result.works), 2)

    def test_malformed_envelope(self):
        plan = QueryPlan(
            queries=[
                PlannedQuery(kind="keyword", terms="alpha"),
                PlannedQuery(kind="keyword", terms="beta"),
            ]
        )

        class BadTransport:
            def get_json(self, url, params, headers, timeout):
                if params.get("search") == "alpha":
                    return ["not-a-dict"]
                return {"noresults": []}

        result = collect_with_feedback(plan, BadTransport())
        self.assertEqual(result.feedback[0].status, "malformed")
        self.assertEqual(result.feedback[0].returned, 0)
        self.assertEqual(result.feedback[0].accepted, 0)
        self.assertEqual(result.feedback[1].status, "malformed")
        self.assertEqual(result.works, [])

    def test_skipped_budget(self):
        plan = QueryPlan(
            queries=[
                PlannedQuery(kind="keyword", terms="alpha"),
                PlannedQuery(kind="keyword", terms="beta"),
                PlannedQuery(kind="keyword", terms="gamma"),
            ]
        )
        pages = {
            "alpha": {"results": [_raw("https://openalex.org/W1")]},
            "beta": {"results": [_raw("https://openalex.org/W2")]},
            "gamma": {"results": [_raw("https://openalex.org/W3")]},
        }
        result = collect_with_feedback(plan, DictTransport(pages), max_total=1)
        self.assertEqual(result.feedback[0].status, "ok")
        self.assertEqual(result.feedback[1].status, "skipped_budget")
        self.assertEqual(result.feedback[2].status, "skipped_budget")
        self.assertEqual(len(result.works), 1)

    def test_no_scientific_scores(self):
        plan = QueryPlan(queries=[PlannedQuery(kind="keyword", terms="alpha")])
        pages = {"alpha": {"results": [_raw("https://openalex.org/W1")]}}
        result = collect_with_feedback(plan, DictTransport(pages))
        for work in result.works:
            self.assertEqual(work.score, 0.0)


class TestSemanticPacingAcrossWaves(unittest.TestCase):
    def test_pacing_holds_on_same_transport(self):
        clock, starts = [0.0], []

        class Source:
            _last_semantic_completion = None

            def get_json(self, url, params, headers, timeout):
                starts.append(clock[0])
                return {"results": []}

        def advance(seconds):
            clock[0] += seconds

        transport = RetryingTransport(Source())
        first = QueryPlan(queries=[PlannedQuery(kind="semantic", terms="first")])
        second = QueryPlan(queries=[PlannedQuery(kind="semantic", terms="second")])
        with mock.patch(
            "radar.source.openalex._time.monotonic", side_effect=lambda: clock[0]
        ), mock.patch("radar.source.openalex._time.sleep", side_effect=advance):
            collect_with_feedback(first, transport)
            collect_with_feedback(second, transport)
        # Second wave waited for the first wave's completion.
        self.assertEqual(starts, [0.0, 1.0])

    def test_adapter_holds_pacing_for_slots_transport(self):
        from radar.source.openalex import SemanticPacingAdapter

        clock, starts = [0.0], []

        class SlotsSource:
            __slots__ = ()

            def get_json(self, url, params, headers, timeout):
                starts.append(clock[0])
                return {"results": []}

        def advance(seconds):
            clock[0] += seconds

        adapter = SemanticPacingAdapter(SlotsSource())
        first = QueryPlan(queries=[PlannedQuery(kind="semantic", terms="first")])
        second = QueryPlan(queries=[PlannedQuery(kind="semantic", terms="second")])
        with mock.patch(
            "radar.source.openalex._time.monotonic", side_effect=lambda: clock[0]
        ), mock.patch("radar.source.openalex._time.sleep", side_effect=advance):
            collect_with_feedback(first, adapter)
            collect_with_feedback(second, adapter)
        self.assertEqual(starts, [0.0, 1.0])


if __name__ == "__main__":
    unittest.main()
