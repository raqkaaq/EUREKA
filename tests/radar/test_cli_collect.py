"""Regression: --max-candidates bounds final output, not plan execution."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from radar import cli as _cli


def _raw(wid: str, title: str, year: int = 2026) -> dict:
    return {
        "id": wid,
        "title": title,
        "abstract_inverted_index": {"x": [0]},
        "doi": "",
        "publication_year": year,
        "cited_by_count": 0,
    }


class FakeTransport:
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

    def close(self):
        pass


class TestCollectCandidatesBounds(unittest.TestCase):
    def test_small_max_still_executes_whole_plan(self):
        fake = FakeTransport()
        with patch.object(_cli, "HttpxTransport", lambda: fake):
            works = _cli.collect_candidates(
                max_candidates=1, lookback_days=90, timeout=10.0, keywords=None
            )
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
        self.assertEqual(len(works), 1)


if __name__ == "__main__":
    unittest.main()
