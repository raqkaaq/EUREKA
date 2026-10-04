"""Research agenda behavior at the planner/source boundary."""

import datetime as dt
import unittest
from unittest import mock

from radar.config.interests import default_profile
from radar.config.searches import build_query_plan
from radar.source.openalex import build_request, collect, DictTransport, RetryingTransport, OpenAlexHttpError


class TestLearningSearches(unittest.TestCase):
    def test_agenda_has_historical_and_fresh_branches_for_each_question(self):
        plan = build_query_plan(default_profile(), today=dt.date(2026, 10, 4))
        self.assertEqual(len(plan.queries), 12)
        self.assertLessEqual(sum(q.per_page for q in plan.queries), 200)
        questions = {q.question_id for q in plan.queries}
        self.assertEqual(len(questions), 4)
        for question in questions:
            branches = [q for q in plan.queries if q.question_id == question]
            self.assertEqual({q.role for q in branches},
                             {"foundation", "frontier", "counterevidence"})
            for query in branches:
                self.assertEqual(query.from_date, "2026-07-06" if query.role == "frontier" else None)

    def test_frontier_uses_real_semantic_search_not_boolean_keyword_search(self):
        plan = build_query_plan(default_profile())
        query = next(q for q in plan.queries if q.role == "frontier")
        _, params, _, _ = build_request(query)
        self.assertIn("search.semantic", params)
        self.assertNotIn("search", params)
        self.assertNotIn("sort", params)
        self.assertGreater(len(params["search.semantic"]), 100)

    def test_question_and_role_provenance_survives_deduplication(self):
        plan = build_query_plan(default_profile())
        raw = {"id": "https://openalex.org/W1", "title": "Shared paper",
               "publication_date": dt.date.today().isoformat()}
        source = DictTransport({q.terms: {"results": [raw]} for q in plan.queries})
        with mock.patch("radar.source.openalex._time.sleep"):
            papers = collect(plan, source)
        self.assertEqual(len(papers), 1)
        self.assertEqual(len(papers[0].discovery_matches), 12)
        self.assertEqual({(m.question_id, m.role) for m in papers[0].discovery_matches},
                         {(q.question_id, q.role) for q in plan.queries})

    def test_smaller_request_cap_covers_different_questions_first(self):
        plan = build_query_plan(default_profile(), max_queries=4)
        self.assertEqual(len({q.question_id for q in plan.queries}), 4)
        self.assertTrue(all(q.role == "frontier" for q in plan.queries))

    def test_empty_domains_omit_only_cross_domain_questions(self):
        profile = default_profile().model_copy(update={"domains": []})
        plan = build_query_plan(profile)
        self.assertEqual(len(plan.queries), 6)
        self.assertEqual({q.question_id for q in plan.queries},
                         {"learning_and_generalization", "uncertainty_and_identification"})

    def test_semantic_retry_and_next_query_are_both_paced(self):
        from radar.schema.papers import PlannedQuery, QueryPlan
        clock, starts = [0.0], []

        class Source:
            def get_json(self, *args):
                starts.append(clock[0])
                if len(starts) == 1:
                    raise OpenAlexHttpError(429, "transient", retry_after=1)
                return {"results": []}

        def advance(seconds):
            clock[0] += seconds

        plan = QueryPlan(queries=[PlannedQuery(kind="semantic", terms=t) for t in ("first", "second")])
        with mock.patch("radar.source.openalex._time.monotonic", side_effect=lambda: clock[0]), \
             mock.patch("radar.source.openalex._time.sleep", side_effect=advance):
            collect(plan, RetryingTransport(Source()))
        self.assertEqual(starts, [0.0, 1.0, 2.0])

    def test_legacy_semantic_templates_keep_keyword_wire_behavior(self):
        from radar.config.yaml import parse_yaml
        from radar.schema.configuration import SearchConfig
        config = parse_yaml('''version: 1
queries:
  - name: legacy
    kind: semantic
    terms: '{keywords}'
''', SearchConfig)
        plan = build_query_plan(default_profile(), configuration=config)
        self.assertEqual(plan.queries[0].kind, "keyword")
        _, params, _, _ = build_request(plan.queries[0])
        self.assertIn("search", params)
        self.assertNotIn("search.semantic", params)

    def test_semantic_freshness_uses_supported_year_filter_and_exact_local_date(self):
        from radar.schema.papers import PlannedQuery, QueryPlan
        query = PlannedQuery(kind="semantic", terms="causal learning", from_date="2026-07-06")
        _, params, _, _ = build_request(query)
        self.assertEqual(params["filter"], "publication_year:2026-")
        self.assertIn("publication_date", params["select"])
        rows = [{"id": f"https://openalex.org/W{i}", "publication_date": date}
                for i, date in enumerate(("2026-07-05", "2026-07-06", "2026-08-10", None,
                                          "invalid", "2026-W28-1"))]
        papers = collect(QueryPlan(queries=[query]), DictTransport({query.terms: {"results": rows}}))
        self.assertEqual([p.openalex_id for p in papers],
                         ["https://openalex.org/W1", "https://openalex.org/W2"])
        self.assertEqual(papers[0].publication_date, "2026-07-06")
