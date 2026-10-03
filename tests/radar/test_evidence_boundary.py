"""Evidence-boundary regressions; no external services or credentials."""

import unittest

from pydantic import ValidationError

from radar.processing.evidence import attach_evidence
from radar.schema.opportunities import OpportunityDraft, RadarDraft
from radar.schema.papers import CollectedWork
from radar.source.openalex import normalize_work


class TestEvidenceBoundary(unittest.TestCase):
    def test_citation_indices_reject_coercion(self):
        for index in (True, False, "0", "1", 0.0, 1.0):
            with self.subTest(input_type=type(index).__name__, index=index):
                with self.assertRaises(ValidationError):
                    OpportunityDraft.model_validate({"title": "Hypothesis", "evidence": [index]})
        self.assertEqual(OpportunityDraft(title="Hypothesis", evidence=[0, 1]).evidence, [0, 1])

    def test_bad_publisher_url_falls_back_without_crashing(self):
        urls = ("http://[malformed", "https://publisher.example:bad/paper",
                "https://publisher.example:70000/paper", "javascript:alert(1)",
                "https://user:password@publisher.example/paper",
                "https://publisher.example/paper\nInjected", "https://publisher.example/with space")
        for url in urls:
            with self.subTest(url=url):
                work = CollectedWork(openalex_id="https://openalex.org/W1", primary_url=url)
                draft = RadarDraft(opportunities=[OpportunityDraft(title="Hypothesis", evidence=[0])])
                report = attach_evidence(draft, [work])
                self.assertEqual(report.opportunities[0].evidence_links[0].url, work.openalex_id)

    def test_unsafe_fallback_is_omitted_while_other_citations_survive(self):
        bad = CollectedWork(openalex_id="javascript:openalex.org", primary_url="http://[bad")
        good = CollectedWork(openalex_id="https://openalex.org/W2",
                             primary_url="https://publisher.example/paper")
        draft = RadarDraft(opportunities=[OpportunityDraft(title="Hypothesis", evidence=[0, 1])])
        report = attach_evidence(draft, [bad, good])
        self.assertEqual(report.opportunities[0].draft.title, "Hypothesis")
        self.assertEqual([link.index for link in report.opportunities[0].evidence_links], [1])
        self.assertEqual(report.opportunities[0].evidence_links[0].url, good.primary_url)

    def test_source_identity_requires_a_real_openalex_work_url(self):
        rejected = ("javascript:openalex.org", "https://openalex.org.evil.example/W1",
                    "https://evil.example/openalex.org/W1", "https://openalex.org/works",
                    "https://openalex.org/W1?token=private", "https://openalex.org/W1#fragment",
                    "https://user:password@openalex.org/W1", "https://openalex.org:bad/W1",
                    "http://[openalex.org", "https://openalex.org/W1\nInjected")
        for identity in rejected:
            with self.subTest(identity=identity):
                self.assertIsNone(normalize_work({"id": identity, "title": "Paper"}, "learning"))
        for identity in ("https://openalex.org/W123", "http://openalex.org/W123"):
            with self.subTest(identity=identity):
                self.assertEqual(normalize_work({"id": identity}, "learning").openalex_id, identity)

    def test_valid_publisher_urls_and_integer_schema_are_unchanged(self):
        urls = ("https://doi.org/10.1234/test", "https://publisher.example/paper?q=ml%20economics",
                "https://publisher.example:8443/paper", "http://[::1]:8000/paper")
        for url in urls:
            with self.subTest(url=url):
                work = CollectedWork(openalex_id="https://openalex.org/W1", primary_url=url)
                draft = RadarDraft(opportunities=[OpportunityDraft(title="Hypothesis", evidence=[0])])
                self.assertEqual(attach_evidence(draft, [work]).opportunities[0].evidence_links[0].url, url)
        self.assertEqual(OpportunityDraft.model_json_schema()["properties"]["evidence"]["items"],
                         {"type": "integer"})


if __name__ == "__main__":
    unittest.main()
