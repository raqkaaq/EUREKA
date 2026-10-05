"""Adaptive plans are strict research intentions, not arbitrary network actions."""

import unittest
from radar.schema.discovery import SearchIntent, SearchWavePlan


class TestDiscoveryContracts(unittest.TestCase):
    def test_rejects_unstructured_network_actions(self):
        with self.assertRaises(ValueError):
            SearchWavePlan.model_validate({"summary": "Explore mechanisms", "intents": [],
                                           "url": "https://another-source.example"})

    def test_question_and_rationale_are_retained_with_actual_query(self):
        intent = SearchIntent.model_validate({
            "learning_goal_id": "generalization", "question": "Which assumption makes the transfer result hold?",
            "rationale": "Discriminate a structural mechanism from an evaluation artifact.",
            "expected_learning_value": "Reconstruct the assumption-dependent argument.",
            "origin": "agenda", "query": {"kind": "semantic", "terms": "transfer under distribution shift",
                                             "question_id": "transfer_assumptions", "role": "foundation"}})
        plan = SearchWavePlan(summary="Study the mechanism", intents=[intent])
        self.assertEqual(plan.query_plan.queries[0].terms, "transfer under distribution shift")
        self.assertEqual(plan.intents[0].question, intent.question)
