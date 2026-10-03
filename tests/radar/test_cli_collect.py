"""Pipeline collection: the whole bounded plan executes; the output bound
applies only to the final ranked slice. Fake source, no network."""

from __future__ import annotations

import json
import unittest

from radar.pipeline import PipelineRequest, run


def _raw(wid: str, title: str, year: int = 2026) -> dict:
    return {
        "id": wid,
        "title": title,
        "abstract_inverted_index": {"x": [0]},
        "doi": "",
        "publication_year": year,
        "cited_by_count": 0,
    }


class FakeSource:
    """Serve one distinct work per planned query; record params."""

    def __init__(self):
        self.calls: list[dict] = []
        self.n = 0

    def get_json(self, url, params, headers, timeout):
        self.calls.append(dict(params))
        self.n += 1
        return {
            "results": [
                _raw(f"https://openalex.org/W{self.n}", f"Work {self.n}")
            ]
        }


class TestPipelineCollectionBounds(unittest.TestCase):
    def test_small_max_still_executes_whole_plan(self):
        fake = FakeSource()
        result = run(PipelineRequest(
            mode="collect", max_candidates=1, source_override=fake))
        self.assertEqual(result.exit_code, 0)
        # Whole bounded plan (five themes + recent query) executes despite
        # an output max of 1.
        self.assertEqual(len(fake.calls), 6)
        # Every discovery request carries the lookback date filter.
        for params in fake.calls:
            self.assertIn("from_publication_date:", params.get("filter", ""))
        # Sort only on the recent query (relevance preserved for semantic).
        sorts = [p.get("sort") for p in fake.calls]
        self.assertEqual(sorts.count("publication_date:desc"), 1)
        # Output bound respected.
        shown = json.loads(result.stdout)
        self.assertEqual(len(shown), 1)


if __name__ == "__main__":
    unittest.main()
