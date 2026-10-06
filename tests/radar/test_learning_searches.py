"""Focused retrieval behavior at the planner/source boundary."""

import datetime as dt
import unittest
from unittest import mock

from radar.config.interests import default_profile
from radar.config.searches import build_query_plan
from radar.source.openalex import build_request, collect, DictTransport, RetryingTransport, OpenAlexHttpError


class TestLearningSearches(unittest.TestCase):
    def test_question_and_role_provenance_survives_deduplication(self):
        from radar.schema.papers import PlannedQuery, QueryPlan

        plan = QueryPlan(
            queries=[
                PlannedQuery(
                    kind="semantic", terms="transfer under shift",
                    question_id="transfer", role="foundation",
                ),
                PlannedQuery(
                    kind="keyword", terms="transfer evaluation",
                    question_id="transfer", role="counterevidence",
                ),
                PlannedQuery(
                    kind="recent", terms="transfer",
                    from_date=dt.date.today().isoformat(),
                    question_id="transfer", role="frontier",
                ),
            ]
        )
        raw = {"id": "https://openalex.org/W1", "title": "Shared paper",
               "publication_date": dt.date.today().isoformat()}
        pages = {}
        for q in plan.queries:
            key = q.terms
            pages[key] = {"results": [dict(raw)]}
        source = DictTransport(pages)
        with mock.patch("radar.source.openalex._time.sleep"):
            papers = collect(plan, source)
        self.assertEqual(len(papers), 1)
        self.assertEqual(len(papers[0].discovery_matches), 3)
        self.assertEqual({(m.question_id, m.role) for m in papers[0].discovery_matches},
                         {(q.question_id, q.role) for q in plan.queries})

    def test_cross_domain_role_provenance(self):
        from radar.schema.papers import PlannedQuery, QueryPlan

        plan = QueryPlan(
            queries=[
                PlannedQuery(
                    kind="keyword", terms="bridging methods",
                    question_id="bridge", role="cross_domain",
                ),
                PlannedQuery(
                    kind="semantic", terms="exploratory transfer probe",
                    question_id="bridge", role="exploration",
                ),
            ]
        )
        raw = {"id": "https://openalex.org/W9", "title": "Bridge paper"}
        source = DictTransport(
            {"bridging methods": {"results": [dict(raw)]},
             "exploratory transfer probe": {"results": [dict(raw)]}})
        with mock.patch("radar.source.openalex._time.sleep"):
            papers = collect(plan, source)
        self.assertEqual(len(papers), 1)
        self.assertEqual(
            {(m.question_id, m.role) for m in papers[0].discovery_matches},
            {("bridge", "cross_domain"), ("bridge", "exploration")},
        )

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
