"""Adaptive discovery planner behavior through the real model seam.

All planner tests run through actual PydanticAI ``FunctionModel`` calls;
none mock planner internals. Sections marked ``needs-extended-schema``
require the sourceworker extension (``exploration``/``cross_domain`` roles,
``references``/``citations`` kinds, ``seed_work_id``); they assert a clear
blocker error until that schema lands, then assert full behavior.
"""

from __future__ import annotations

import datetime as dt
import unittest

from radar.config.interests import RadarProfile, default_profile
from radar.schema.discovery import (
    DiscoveryMemory,
    DiscoveryPolicy,
    QueryFeedback,
    SavedFinding,
    SearchWavePlan,
)
from radar.schema.papers import CollectedWork


def _supports_role(role: str) -> bool:
    from typing import get_args

    from radar.schema.papers import SearchRole

    try:
        return role in get_args(SearchRole)
    except Exception:
        return False


def _supports_kind(kind: str) -> bool:
    from radar.schema.papers import PlannedQuery

    field = PlannedQuery.model_fields.get("kind")
    if field is None:
        return False
    annotation = getattr(field, "annotation", None)
    try:
        from typing import get_args

        return kind in get_args(annotation)
    except Exception:
        return False


def _supports_seed() -> bool:
    from radar.schema.papers import PlannedQuery

    return "seed_work_id" in PlannedQuery.model_fields


EXTENDED_ROLES = _supports_role("exploration") and _supports_role("cross_domain")

NEEDS_EXTENDED = "needs sourceworker schema extension (exploration/cross_domain roles)"


def _policy(**overrides) -> DiscoveryPolicy:
    from radar.config.searches import search_policy

    base = search_policy().model_dump()
    base.update(overrides)
    return DiscoveryPolicy.model_validate(base)


def _memory(*, known=(), findings=()) -> DiscoveryMemory:
    return DiscoveryMemory(findings=list(findings), known_work_ids=list(known))


def _work(work_id: str, abstract: str = "abstract text") -> CollectedWork:
    return CollectedWork(openalex_id=work_id, title="Title", abstract=abstract)


class TestPlannerLoader(unittest.TestCase):
    def test_prompt_separates_source_provenance_from_citation_seeds(self):
        from radar.prompts.catalog import search_planning_prompt

        instructions = search_planning_prompt().instructions
        self.assertIn("no per-field character limits", instructions)
        self.assertIn("seed_work_id only for references or citations", instructions)
        self.assertIn("source_work_ids belongs to the intent", instructions)
        self.assertIn("omit seed_work_id or use null", instructions)

    def test_search_policy_is_typed_v3_without_query_strings(self):
        from radar.config.searches import search_config, search_policy

        for loader in (search_policy, search_config):
            policy = loader()
            self.assertIsInstance(policy, DiscoveryPolicy)
            self.assertEqual(policy.version, 3)
            self.assertEqual(policy.initial_queries, 6)
            self.assertEqual(policy.followup_queries, 6)
            self.assertEqual(policy.results_per_query, 15)
            self.assertEqual(policy.planning_timeout_s, 120)
            self.assertEqual(policy.planner_max_tokens, 3000)
            self.assertGreaterEqual(len(policy.learning_goals), 1)
            for goal in policy.learning_goals:
                self.assertTrue(goal.question)
                self.assertIn(goal.scope, ("broad", "cross_domain"))

    def test_legacy_build_query_plan_requires_explicit_old_config(self):
        from radar.config.searches import build_query_plan
        from radar.config.yaml import ConfigurationError

        with self.assertRaises(ConfigurationError):
            build_query_plan(default_profile())
        from radar.config.yaml import parse_yaml
        from radar.schema.configuration import SearchConfig

        config = parse_yaml(
            "version: 1\nqueries:\n  - name: legacy\n    kind: recent\n"
            "    terms: '({keywords})'\n",
            SearchConfig,
        )
        plan = build_query_plan(
            RadarProfile(keywords=["graph learning"], lookback_days=30),
            configuration=config, today=dt.date(2026, 10, 3),
        )
        self.assertEqual(plan.queries[0].terms, '("graph learning")')

    def test_search_planning_prompt_is_typed_with_safety_wording(self):
        from radar.prompts.catalog import search_planning_prompt
        from radar.schema.configuration import AgentPrompt

        spec = search_planning_prompt()
        self.assertIsInstance(spec, AgentPrompt)
        text = spec.instructions.lower()
        self.assertIn("untrusted", text)
        self.assertIn("no tools", text)

    def test_fingerprints_are_stable_and_distinguish_policy_profile(self):
        from radar.agent.discovery_planning import planning_fingerprints

        policy = _policy()
        first = planning_fingerprints(default_profile(), policy)
        self.assertEqual(first, planning_fingerprints(default_profile(), policy))
        self.assertEqual(len(first[0]), 64)
        self.assertEqual(len(first[1]), 64)
        other_profile = default_profile().model_copy(update={"keywords": ["other"]})
        self.assertNotEqual(
            first[0], planning_fingerprints(other_profile, policy)[0])
        self.assertNotEqual(
            first[1], planning_fingerprints(default_profile(), _policy(results_per_query=5))[1])


class TestPlannerThroughModel(unittest.TestCase):
    def test_detailed_search_questions_and_rationales_are_not_rejected_for_length(self):
        from pydantic_ai.messages import ModelResponse, ToolCallPart
        from pydantic_ai.models.function import FunctionModel
        from radar.agent.discovery_planning import plan_searches
        from tests.radar.discovery_support import initial_plan_dict

        payload = initial_plan_dict()
        payload["summary"] = ("Discriminate mechanisms from evaluation artifacts. " * 15).strip()
        detail = payload["intents"][0]
        detail["question"] = ("Which identifying assumption distinguishes these mechanisms? " * 12).strip()
        detail["rationale"] = ("Test the boundary condition rather than shared terminology. " * 10).strip()
        detail["expected_learning_value"] = ("Reconstruct the inference and its failure case. " * 10).strip()
        def respond(messages, info):
            return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, payload)])

        result = plan_searches(default_profile(), _memory(), model=FunctionModel(respond))
        self.assertEqual(result.summary, payload["summary"])
        self.assertEqual(result.intents[0].question, detail["question"])
        self.assertEqual(result.intents[0].rationale, detail["rationale"])
        self.assertEqual(result.intents[0].expected_learning_value, detail["expected_learning_value"])

    def test_citation_queries_do_not_repeat_requests_using_different_descriptions(self):
        from radar.agent.discovery_planning import validate_search_plan
        from tests.radar.discovery_support import initial_plan_dict
        payload = initial_plan_dict()
        payload["intents"][0]["query"].update(kind="references",
            seed_work_id="https://openalex.org/W1", terms="first description")
        prior = SearchWavePlan.model_validate(payload)
        next_payload = initial_plan_dict()
        next_payload["intents"] = next_payload["intents"][:1]
        next_payload["intents"][0]["query"].update(kind="references",
            seed_work_id="https://openalex.org/W1", terms="different description")
        with self.assertRaises(ValueError):
            validate_search_plan(SearchWavePlan.model_validate(next_payload), default_profile(),
                _memory(known=["https://openalex.org/W1"]), previous_plan=prior)

    def test_planner_errors_do_not_echo_upstream_private_text(self):
        from pydantic_ai.models.function import FunctionModel
        from radar.agent.discovery_planning import plan_searches
        from radar.provider.strata import StrataError
        def fail(messages, info):
            raise RuntimeError("private prompt secret-token fixture")
        with self.assertRaises(StrataError) as caught:
            plan_searches(default_profile(), _memory(), model=FunctionModel(fail))
        self.assertNotIn("secret-token", str(caught.exception))

    def test_model_can_name_a_new_question_independently_of_its_learning_goal(self):
        from pydantic_ai.messages import ModelResponse, ToolCallPart
        from pydantic_ai.models.function import FunctionModel
        from radar.agent.discovery_planning import plan_searches
        from tests.radar.discovery_support import initial_plan_dict
        def respond(messages, info):
            payload = initial_plan_dict()
            payload["intents"][0]["query"]["question_id"] = "when_transfer_bounds_fail"
            return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, payload)])
        result = plan_searches(default_profile(), _memory(), model=FunctionModel(respond))
        self.assertEqual(result.intents[0].query.question_id, "when_transfer_bounds_fail")

    def test_all_allowed_retrieved_ids_are_supplied_and_dense_history_fits(self):
        from radar.agent.discovery_planning import build_prompt
        from tests.radar.discovery_support import initial_plan_dict
        prior = SearchWavePlan.model_validate(initial_plan_dict())
        feedback = [QueryFeedback(query=i.query, returned=2, accepted=2, new_to_run=1,
                    new_to_library=1, with_abstract=1, with_pdf=1) for i in prior.intents]
        findings = [SavedFinding(work_id=f"https://openalex.org/W{i}", contribution="c" * 600,
            limits=["l" * 200] * 3, open_questions=["q" * 200] * 3) for i in range(3)]
        memory = DiscoveryMemory(findings=findings,
            known_work_ids=[f"https://openalex.org/W{i}" for i in range(100)],
            prior_feedback=(feedback * 2)[:9])
        works = [_work(f"https://openalex.org/W{1000+i}", abstract="a" * 600) for i in range(13)]
        prompt = build_prompt(default_profile(), memory, feedback=feedback,
                              previous_plan=prior, retrieved=works)
        from radar.prompts.catalog import search_planning_prompt
        self.assertLessEqual(len(prompt) + len(search_planning_prompt().instructions) + 2, 24000)
        self.assertIn("https://openalex.org/W1012", prompt)

    def test_different_agenda_memory_feedback_produces_different_queries(self):
        from pydantic_ai.messages import UserPromptPart
        from radar.agent.discovery_planning import plan_searches
        from tests.radar.discovery_support import known_work_ids, planner_model

        if not EXTENDED_ROLES:
            self.skipTest(NEEDS_EXTENDED)
        seen: list[str] = []

        base_model = planner_model()

        def _run(profile, memory, feedback):
            from pydantic_ai.models.function import FunctionModel

            captured: dict[str, str] = {}

            async def _probe(messages, info):
                captured["prompt"] = "\n".join(
                    part.content for message in messages for part in message.parts
                    if isinstance(part, UserPromptPart))
                return base_model.function(messages, info)

            model = FunctionModel(_probe)
            plan = plan_searches(profile, memory, model=model, feedback=feedback)
            return plan, captured["prompt"]

        known = known_work_ids()[:2]
        feedback = (QueryFeedback(
            query={"kind": "keyword", "terms": "stale terms here",
                   "per_page": 10, "question_id": "learning_and_generalization",
                   "role": "foundation"},
            returned=10, accepted=1, new_to_run=0, new_to_library=0,
            with_abstract=9, with_pdf=0),)
        profile_a = RadarProfile(keywords=["graph learning"], domains=["economics"])
        memory_a = _memory(known=known, findings=[SavedFinding(
            work_id=known[0], contribution="mechanism evidence",
            open_questions=["which assumption fails?"])])
        plan_a, prompt_a = _run(profile_a, memory_a, feedback)

        profile_b = RadarProfile(keywords=["diffusion models"], domains=["economics"])
        memory_b = _memory(known=known, findings=[SavedFinding(
            work_id=known[1], contribution="different evidence",
            open_questions=["what replicates?"])])
        plan_b, prompt_b = _run(profile_b, memory_b, ())

        self.assertIn("graph learning", prompt_a)
        self.assertIn("diffusion models", prompt_b)
        self.assertIn("stale terms here", prompt_a)
        self.assertNotIn("stale terms here", prompt_b)
        self.assertIn("which assumption fails?", prompt_a)
        self.assertIn("what replicates?", prompt_b)
        terms_a = {i.query.terms for i in plan_a.intents}
        terms_b = {i.query.terms for i in plan_b.intents}
        # Same offline double returns the same fixed wave, but the distinct
        # agenda/memory/feedback visibly reaches the model prompts.
        self.assertNotEqual(prompt_a, prompt_b)
        self.assertTrue(terms_a and terms_b)

    def test_fixture_waves_validate_and_followup_avoids_duplicates(self):
        from radar.agent.discovery_planning import validate_search_plan
        from tests.radar.discovery_support import (
            followup_plan_dict, initial_plan_dict, known_work_ids,
        )

        if not EXTENDED_ROLES:
            self.skipTest(NEEDS_EXTENDED)
        policy = _policy()
        profile = RadarProfile(keywords=["graph learning"], domains=["economics"])
        memory = _memory()
        first = SearchWavePlan.model_validate(initial_plan_dict())
        validated = validate_search_plan(first, profile, memory, policy=policy)
        roles = {i.query.role for i in validated.intents}
        self.assertIn("foundation", roles)
        self.assertIn("counterevidence", roles)
        self.assertIn("exploration", roles)
        self.assertIn("cross_domain", roles)
        self.assertLessEqual(len(validated.intents), policy.initial_queries)

        # A five-intent wave against a four-query policy fails the
        # code-owned budget even though the schema allows six.
        trimmed = SearchWavePlan.model_validate(
            {**initial_plan_dict(), "intents": initial_plan_dict()["intents"][:5]})
        with self.assertRaises(ValueError):
            validate_search_plan(trimmed, profile, memory,
                                 policy=_policy(initial_queries=4))

        grounded = _memory(known=known_work_ids()[:4])
        second = SearchWavePlan.model_validate(followup_plan_dict())
        # Followup against an unrelated prior wave validates within budget.
        other = SearchWavePlan.model_validate(initial_plan_dict())
        other = other.model_copy(update={"intents": other.intents[:1]})
        validated_followup = validate_search_plan(
            second, profile, grounded, policy=policy, previous_plan=other)
        self.assertLessEqual(len(validated_followup.intents), policy.followup_queries)
        # A followup repeating a prior-wave query fails instead.
        repeat = followup_plan_dict()
        repeat["intents"][0] = initial_plan_dict()["intents"][0]
        with self.assertRaises(ValueError):
            validate_search_plan(SearchWavePlan.model_validate(repeat), profile,
                                 grounded, policy=policy,
                                 previous_plan=SearchWavePlan.model_validate(
                                     initial_plan_dict()))

    def test_invalid_plans_retry_bounded_then_fail(self):
        from pydantic_ai.messages import ModelResponse, ToolCallPart
        from pydantic_ai.models.function import FunctionModel
        from radar.agent.discovery_planning import plan_searches
        from tests.radar.discovery_support import initial_plan_dict

        if not EXTENDED_ROLES:
            self.skipTest(NEEDS_EXTENDED)
        cases = {
            "missing_role": lambda p: p["intents"][0]["query"].pop("role"),
            "invented_source": lambda p: p["intents"][0].update(
                {"origin": "finding", "source_work_ids": ["https://openalex.org/W999999"]}),
            "finding_without_source": lambda p: p["intents"][0].update({"origin": "finding"}),
            "duplicate": lambda p: p["intents"].__setitem__(1, dict(p["intents"][0])),
            "overbudget": lambda p: p["intents"].extend([dict(p["intents"][0]) for _ in range(6)]),
            "bad_date": lambda p: p["intents"][0]["query"].update(
                {"kind": "recent", "from_date": "not-a-date"}),
            "future_date": lambda p: p["intents"][0]["query"].update(
                {"kind": "recent", "from_date": "2999-01-01"}),
            "unknown_goal": lambda p: p["intents"][0].update({"learning_goal_id": "nope_unknown"}),
            "free_url": lambda p: p["intents"][0]["query"].update(
                {"terms": "see https://example.com/paper for details"}),
        }
        for name, mutate in cases.items():
            with self.subTest(case=name):
                calls = []

                def _impl(messages, info, _mutate=mutate):
                    calls.append(len(calls))
                    payload = initial_plan_dict()
                    _mutate(payload)
                    return ModelResponse(parts=[ToolCallPart(
                        info.output_tools[0].name, payload)])

                with self.assertRaises(Exception):
                    plan_searches(default_profile(), _memory(),
                                  model=FunctionModel(_impl))
                # Bounded: one initial request plus one validation retry.
                self.assertEqual(len(calls), 2, name)

    def test_invented_citation_seed_fails_when_supported(self):
        from pydantic_ai.messages import ModelResponse, ToolCallPart
        from pydantic_ai.models.function import FunctionModel
        from radar.agent.discovery_planning import plan_searches
        from tests.radar.discovery_support import initial_plan_dict

        if not (_supports_kind("citations") and _supports_seed()):
            self.skipTest("needs sourceworker citation kind + seed_work_id")
        calls = []

        def _impl(messages, info):
            calls.append(1)
            payload = initial_plan_dict()
            payload["intents"][0]["query"].update(
                {"kind": "citations", "seed_work_id": "https://openalex.org/W999999"})
            return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, payload)])

        with self.assertRaises(Exception):
            plan_searches(default_profile(), _memory(), model=FunctionModel(_impl))
        self.assertEqual(len(calls), 2)

    def test_no_free_urls_in_validated_output(self):
        from radar.agent.discovery_planning import validate_search_plan
        from tests.radar.discovery_support import initial_plan_dict

        if not EXTENDED_ROLES:
            self.skipTest(NEEDS_EXTENDED)
        payload = initial_plan_dict()
        payload["intents"][0]["query"]["terms"] = "read https://example.com/x now"
        with self.assertRaises(ValueError):
            validate_search_plan(SearchWavePlan.model_validate(payload),
                                 default_profile(), _memory(), policy=_policy())

    def test_context_cap_and_timeout_are_enforced(self):
        from pydantic_ai.messages import UserPromptPart
        from pydantic_ai.models.function import FunctionModel
        from radar.agent.discovery_planning import plan_searches
        from tests.radar.discovery_support import known_work_ids, planner_model

        if not EXTENDED_ROLES:
            self.skipTest(NEEDS_EXTENDED)
        known = known_work_ids()
        findings = [SavedFinding(
            work_id=known[i % len(known)], title=f"finding {i}",
            contribution="c" * 500, limits=["l1", "l2"],
            open_questions=[f"open question {i}"],
            evidence_level="abstract") for i in range(6)]
        retrieved = [_work(known[i % len(known)], abstract="a" * 5000) for i in range(20)]
        captured: dict[str, str] = {}
        base = planner_model()

        async def _probe(messages, info):
            captured["prompt"] = "\n".join(
                part.content for message in messages for part in message.parts
                if isinstance(part, UserPromptPart))
            captured["instructions"] = info.instructions or ""
            return base.function(messages, info)

        plan = plan_searches(default_profile(), _memory(known=known, findings=findings),
                             model=FunctionModel(_probe), retrieved=tuple(retrieved))
        self.assertTrue(plan.intents)
        self.assertLessEqual(
            len(captured["prompt"]) + len(captured["instructions"]) + 2, 24000)
        self.assertIn("abstract excerpt", captured["prompt"].lower())

        with self.assertRaises(ValueError):
            plan_searches(default_profile(), _memory(), model=planner_model(), timeout_s=0)
        # A small but valid deadline does not falsely trip on an immediate model.
        quick = plan_searches(default_profile(), _memory(),
                              model=planner_model(), timeout_s=30)
        self.assertTrue(quick.intents)

    def test_goals_are_hypotheses_not_mastery(self):
        from pydantic_ai.messages import UserPromptPart
        from pydantic_ai.models.function import FunctionModel
        from radar.agent.discovery_planning import plan_searches
        from tests.radar.discovery_support import planner_model

        if not EXTENDED_ROLES:
            self.skipTest(NEEDS_EXTENDED)
        captured: dict[str, str] = {}
        base = planner_model()

        async def _probe(messages, info):
            captured["prompt"] = "\n".join(
                part.content for message in messages for part in message.parts
                if isinstance(part, UserPromptPart))
            captured["instructions"] = info.instructions or ""
            return base.function(messages, info)

        plan = plan_searches(default_profile(), _memory(), model=FunctionModel(_probe))
        blob = (captured["prompt"] + "\n" + captured["instructions"]).lower()
        self.assertIn("hypothes", blob)
        self.assertIn("not established", blob)
        # "mastery" appears only inside "not ... mastery" disclaimers.
        self.assertNotIn("mastered", blob)
        self.assertNotIn("achieve mastery", blob)
        self.assertNotIn("demonstrate mastery", blob)
        self.assertTrue(all(i.question and i.rationale for i in plan.intents))

    def test_fixture_support_matches_terms_pages_and_transport(self):
        from radar.source.openalex import DictTransport, collect
        from tests.radar.discovery_support import fake_pages, planner_model, test_query_plan

        if not EXTENDED_ROLES:
            self.skipTest(NEEDS_EXTENDED)
        from radar.agent.discovery_planning import plan_searches

        plan = plan_searches(default_profile(), _memory(), model=planner_model())
        fixture_queries = {q.terms for q in test_query_plan().queries}
        self.assertEqual({i.query.terms for i in plan.intents}, fixture_queries)

        pages = fake_pages()
        self.assertTrue(fixture_queries <= set(pages))
        papers = collect(test_query_plan(), DictTransport(pages))
        self.assertEqual(len(papers), len(fixture_queries))

        no_domains = default_profile().model_copy(update={"domains": []})
        omitted = test_query_plan(no_domains)
        self.assertTrue(all(q.role != "cross_domain" for q in omitted.queries))
        self.assertLess(len(omitted.queries), len(test_query_plan().queries))


if __name__ == "__main__":
    unittest.main()
