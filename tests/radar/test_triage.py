"""Pure triage selection policy tests (no network, no I/O)."""

from __future__ import annotations

import unittest

from radar.config.interests import RadarProfile
from radar.schema.papers import CollectedWork
from radar.schema.triage import TriageBatch, TriageResult


def _work(wid: str) -> CollectedWork:
    return CollectedWork(
        openalex_id=wid, title=f"Title {wid}", abstract="abstract",
        publication_year=2025, doi="", primary_url=wid, locations=[],
        cited_by_count=1, matched_queries=["q"], query_kinds=["semantic"],
        score=0.01,  # low keyword score: policy must use triage probs, not this
    )


def _batch(probs: dict[str, tuple[float, float]],
           unknowns: dict[str, str] | None = None) -> TriageBatch:
    results = [
        TriageResult(work_id=wid, status="scored",
                     ai_ml_relevance=a, cross_domain_potential=c)
        for wid, (a, c) in probs.items()
    ]
    for wid, status in (unknowns or {}).items():
        results.append(TriageResult(work_id=wid, status=status))  # type: ignore[arg-type]
    return TriageBatch(model_id="clef-flash", rubric_version="clef-triage-v1",
                       results=results)


class TestSelectCandidates(unittest.TestCase):
    def test_ranks_by_max_probability_id_tiebreak(self):
        from radar.processing.triage import select_candidates

        works = [_work("W2"), _work("W1"), _work("W3")]
        batch = _batch({"W1": (0.9, 0.1), "W2": (0.2, 0.9), "W3": (0.1, 0.1)})
        # max: W1=0.9, W2=0.9 tie -> ID order W1, W2; one unknown slot unused
        # (no unknowns) so top2 = W1, W2.
        got = select_candidates(works, batch, 2)
        self.assertEqual([w.openalex_id for w in got], ["W1", "W2"])

    def test_unknown_slot_reserved_when_available(self):
        from radar.processing.triage import select_candidates

        works = [_work("W1"), _work("W2"), _work("W3")]
        batch = _batch({"W1": (0.9, 0.9), "W2": (0.8, 0.8)},
                       {"W3": "missing_abstract"})
        got = select_candidates(works, batch, 2)
        # n-1 scored best + first unknown in input order.
        self.assertEqual([w.openalex_id for w in got], ["W1", "W3"])

    def test_no_reservation_for_n1(self):
        from radar.processing.triage import select_candidates

        works = [_work("W1"), _work("W2")]
        batch = _batch({"W1": (0.9, 0.9)}, {"W2": "missing_abstract"})
        got = select_candidates(works, batch, 1)
        self.assertEqual([w.openalex_id for w in got], ["W1"])

    def test_overflow_unknowns_fill_in_input_order(self):
        from radar.processing.triage import select_candidates

        works = [_work("W1"), _work("W2"), _work("W3")]
        batch = _batch({}, {"W1": "failed", "W2": "deadline",
                            "W3": "missing_abstract"})
        got = select_candidates(works, batch, 3)
        self.assertEqual([w.openalex_id for w in got], ["W1", "W2", "W3"])

    def test_empty_and_nonpositive_safe(self):
        from radar.processing.triage import select_candidates

        batch = _batch({})
        self.assertEqual(select_candidates([], batch, 5), [])
        self.assertEqual(select_candidates([_work("W1")], batch, 0), [])

    def test_cross_domain_low_keyword_paper_wins(self):
        from radar.processing.triage import select_candidates

        works = [_work("W-high-kw"), _work("W-cross")]
        works[0].score = 0.95  # keyword heuristic loves W-high-kw
        batch = _batch({"W-high-kw": (0.9, 0.0), "W-cross": (0.2, 0.95)})
        got = select_candidates(works, batch, 1)
        # max: W-high-kw=0.9 vs W-cross=0.95 -> cross-domain paper first.
        self.assertEqual([w.openalex_id for w in got], ["W-cross"])


if __name__ == "__main__":
    unittest.main()
