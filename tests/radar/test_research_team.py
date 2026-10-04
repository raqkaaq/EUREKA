"""Bounded specialist workflow through real PydanticAI and typed YAML."""

import unittest
import asyncio
import time
import tempfile
from types import SimpleNamespace
from unittest import mock

from pydantic_ai.messages import ModelResponse, ToolCallPart, UserPromptPart
from pydantic_ai.models.function import FunctionModel

from radar.schema.opportunities import OpportunityDraft, RadarDraft
from radar.schema.papers import CollectedWork


def works(n=2):
    return [CollectedWork(openalex_id=f"https://openalex.org/W{i}", title=f"Paper {i}",
                          abstract="Learning mechanisms and human incentives.")
            for i in range(n)]


class TestSpecialistPrompts(unittest.TestCase):
    def test_three_distinct_typed_specialists_have_operational_prompts(self):
        from radar.prompts.catalog import SPECIALIST_ROLES, specialist_prompt
        from radar.schema.configuration import SpecialistPrompt

        self.assertEqual(SPECIALIST_ROLES,
                         ("ml_methods", "behavioral_economics", "evidence_review"))
        for role in SPECIALIST_ROLES:
            prompt = specialist_prompt(role)
            self.assertIsInstance(prompt, SpecialistPrompt)
            self.assertEqual(prompt.role, role)
            self.assertIn("untrusted", prompt.instructions.lower())
            self.assertIn("abstract", prompt.instructions.lower())
            self.assertIn("{valid_range}", prompt.task_template)
            self.assertGreater(len(prompt.instructions), 700)

    def test_rubric_version_and_hash_track_the_expanded_question_configuration(self):
        from radar.prompts.catalog import screening_questions
        from radar.schema.configuration import ScreeningConfig

        rubric = screening_questions()
        self.assertEqual(rubric.rubric_version, "clef-triage-v2")
        self.assertEqual(len(rubric.fingerprint), 64)
        changed = rubric.model_dump(by_alias=True)
        changed["questions"]["ai_ml_relevance"]["instructions"] += " More guidance."
        modified = ScreeningConfig.model_validate(changed)
        self.assertNotEqual(modified.fingerprint, rubric.fingerprint)

    def test_invalid_specialist_configuration_stops_before_external_calls(self):
        from radar.config.yaml import ConfigurationError
        from radar.pipeline import PipelineRequest, run
        from radar.source.openalex import DictTransport

        source = DictTransport({})
        with mock.patch("radar.agent.research_team.specialist_prompt",
                        side_effect=ConfigurationError("invalid specialist YAML")):
            result = run(PipelineRequest(mode="analyze", source_override=source))
            collected = run(PipelineRequest(mode="collect", source_override=DictTransport({})))
        self.assertEqual(result.exit_code, 4)
        self.assertEqual(source.calls, [])
        self.assertEqual(collected.exit_code, 0)


class TestResearchTeam(unittest.TestCase):
    def test_three_real_specialists_feed_one_synthesis_with_shared_evidence(self):
        from radar.agent.research_team import research_candidates
        from radar.prompts.catalog import SPECIALIST_ROLES, specialist_prompt

        roles = {specialist_prompt(role).instructions: role for role in SPECIALIST_ROLES}
        seen = []

        def respond(messages, info):
            prompt = next(part.content for message in messages for part in message.parts
                          if isinstance(part, UserPromptPart))
            role = roles.get(info.instructions, "synthesis")
            seen.append((role, prompt))
            if role == "synthesis":
                for name in SPECIALIST_ROLES:
                    self.assertIn(name, prompt)
                    self.assertIn(f"Hypothesis from {name}", prompt)
            draft = RadarDraft(opportunities=[OpportunityDraft(
                title=role, wow=f"Hypothesis from {role}",
                investigate="Compare against a control.", reproduce="Check missing data.",
                evidence=[0])], next_move="Inspect details.")
            return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, draft.model_dump())])

        result = research_candidates(works(), model=FunctionModel(respond))
        self.assertEqual(len(seen), 4)
        self.assertEqual(seen[-1][0], "synthesis")
        self.assertEqual({role for role, _ in seen[:-1]}, set(SPECIALIST_ROLES))
        self.assertEqual([report.role for report in result.specialist_reports], list(SPECIALIST_ROLES))
        self.assertEqual(result.draft.opportunities[0].evidence, [0])
        self.assertEqual(len(result.included), 2)
        for _, prompt in seen:
            self.assertIn("[0] Paper 0", prompt)
            self.assertIn("[1] Paper 1", prompt)
            self.assertIn("0..1", prompt)

    def test_bounded_retries_can_recover_with_at_most_eight_requests(self):
        from radar.agent.research_team import research_candidates
        from radar.prompts.catalog import SPECIALIST_ROLES, specialist_prompt

        roles = {specialist_prompt(role).instructions: role for role in SPECIALIST_ROLES}
        calls = {}

        def respond(messages, info):
            role = roles.get(info.instructions, "synthesis")
            calls[role] = calls.get(role, 0) + 1
            draft = RadarDraft(opportunities=[OpportunityDraft(
                title="Hypothesis", evidence=[199 if calls[role] == 1 else 0])])
            return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, draft.model_dump())])

        result = research_candidates(works(), model=FunctionModel(respond))
        self.assertEqual(sum(calls.values()), 8)
        self.assertEqual(set(calls.values()), {2})
        self.assertEqual(result.draft.opportunities[0].evidence, [0])

    def test_invalid_or_oversized_specialist_reports_block_synthesis(self):
        from radar.agent.research_team import research_candidates
        from radar.prompts.catalog import SPECIALIST_ROLES, specialist_prompt
        from radar.provider.freetoken import FreeTokenError

        roles = {specialist_prompt(role).instructions: role for role in SPECIALIST_ROLES}
        variants = (
            RadarDraft(opportunities=[OpportunityDraft(title="bad", evidence=[99])]),
            RadarDraft(next_move="x" * 1500),
            RadarDraft(opportunities=[OpportunityDraft(title="A"), OpportunityDraft(title="B")]),
        )
        for draft in variants:
            calls = {}

            def respond(messages, info):
                role = roles.get(info.instructions, "synthesis")
                calls[role] = calls.get(role, 0) + 1
                return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, draft.model_dump())])

            with self.subTest(draft=draft.model_dump()), self.assertRaises(FreeTokenError):
                research_candidates(works(), model=FunctionModel(respond))
            self.assertNotIn("synthesis", calls)
            self.assertTrue(all(count <= 2 for count in calls.values()))

    def test_specialist_report_1000_to_1500_accepted_first_pass(self):
        from radar.agent.research_team import research_candidates
        from radar.prompts.catalog import SPECIALIST_ROLES, specialist_prompt

        draft = RadarDraft(next_move="x" * 1400)
        size = len(draft.model_dump_json())
        self.assertGreater(size, 1000)
        self.assertLessEqual(size, 1500)
        roles = {specialist_prompt(role).instructions: role for role in SPECIALIST_ROLES}
        calls = {}

        def respond(messages, info):
            role = roles.get(info.instructions, "synthesis")
            calls[role] = calls.get(role, 0) + 1
            return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, draft.model_dump())])

        result = research_candidates(works(), model=FunctionModel(respond))
        self.assertEqual(sum(calls.values()), 4)
        self.assertEqual(set(calls.values()), {1})
        self.assertEqual(len(result.specialist_reports), 3)
        for report in result.specialist_reports:
            self.assertGreater(len(report.draft.model_dump_json()), 1000)
            self.assertLessEqual(len(report.draft.model_dump_json()), 1500)

    def test_oversized_specialist_report_above_1500_blocked_without_synthesis(self):
        from radar.agent.research_team import research_candidates
        from radar.prompts.catalog import SPECIALIST_ROLES, specialist_prompt
        from radar.provider.freetoken import FreeTokenError

        draft = RadarDraft(next_move="x" * 1600)
        self.assertGreater(len(draft.model_dump_json()), 1500)
        roles = {specialist_prompt(role).instructions: role for role in SPECIALIST_ROLES}
        calls = {}

        def respond(messages, info):
            role = roles.get(info.instructions, "synthesis")
            calls[role] = calls.get(role, 0) + 1
            return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, draft.model_dump())])

        with self.assertRaises(FreeTokenError):
            research_candidates(works(), model=FunctionModel(respond))
        self.assertNotIn("synthesis", calls)
        self.assertTrue(all(count <= 2 for count in calls.values()))

    def test_three_near_max_reports_fit_synthesis_budget(self):
        from radar.agent.research_team import research_candidates
        from radar.config.runtime import MAX_PROMPT_CHARS, SPECIALIST_CONTEXT_CHARS
        from radar.prompts.catalog import SPECIALIST_ROLES, opportunity_analysis_prompt, specialist_prompt

        self.assertEqual(SPECIALIST_CONTEXT_CHARS, 5000)
        draft = RadarDraft(next_move="x" * 1400)
        self.assertGreater(len(draft.model_dump_json()), 1000)
        self.assertLessEqual(len(draft.model_dump_json()), 1500)
        roles = {specialist_prompt(role).instructions: role for role in SPECIALIST_ROLES}
        seen = {}
        synthesis_prompt = []

        def respond(messages, info):
            prompt = next(part.content for message in messages for part in message.parts
                          if isinstance(part, UserPromptPart))
            role = roles.get(info.instructions, "synthesis")
            seen[role] = prompt
            if role == "synthesis":
                synthesis_prompt.append(prompt)
                small = RadarDraft(next_move="Check.")
                return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, small.model_dump())])
            return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, draft.model_dump())])

        result = research_candidates(works(), model=FunctionModel(respond))
        self.assertEqual(len(result.specialist_reports), 3)
        self.assertEqual(result.prompt.count(draft.next_move), 3)
        synthesis_instructions = opportunity_analysis_prompt().instructions
        self.assertEqual(len(synthesis_prompt), 1)
        self.assertLessEqual(len(synthesis_instructions) + len(synthesis_prompt[0]) + 2, MAX_PROMPT_CHARS)
        self.assertLessEqual(len(synthesis_instructions) + len(synthesis_prompt[0]) + 2, 12000)
        for role in SPECIALIST_ROLES:
            self.assertIn(role, synthesis_prompt[0])
        cohort_blocks = ["[0] Paper 0", "[1] Paper 1"]
        for prompt in list(seen.values()) + [result.prompt]:
            for block in cohort_blocks:
                self.assertIn(block, prompt)
            self.assertIn("0..1", prompt)
        self.assertIn("not source evidence", result.prompt)

    def test_shared_deadline_cancels_all_work_and_closes_the_owned_session(self):
        from radar.agent.research_team import research_candidates
        from radar.provider.freetoken import FreeTokenError, FreeTokenSession

        started, settled = [], []

        async def stall(messages, info):
            started.append(1)
            try:
                await asyncio.Event().wait()
            finally:
                settled.append(1)

        client = SimpleNamespace(aclose=mock.AsyncMock())
        session = FreeTokenSession(FunctionModel(stall), client)
        start = time.monotonic()
        with self.assertRaises(FreeTokenError) as caught:
            research_candidates(works(), session=session, analysis_timeout_s=0.03)
        self.assertIn("overall", str(caught.exception))
        self.assertLess(time.monotonic() - start, 1)
        self.assertLessEqual(len(started), 2)
        self.assertEqual(len(started), len(settled))
        client.aclose.assert_awaited_once()

    def test_model_errors_are_redacted_and_owned_session_closes(self):
        from radar.agent.research_team import research_candidates
        from radar.provider.freetoken import FreeTokenError, FreeTokenSession

        def fail(messages, info):
            raise ValueError("SECRET-UPSTREAM-BODY")

        client = SimpleNamespace(aclose=mock.AsyncMock())
        with self.assertRaises(FreeTokenError) as caught:
            research_candidates(works(), session=FreeTokenSession(FunctionModel(fail), client))
        self.assertNotIn("SECRET-UPSTREAM-BODY", str(caught.exception))
        client.aclose.assert_awaited_once()

    def test_common_cohort_fits_every_stage_including_intermediate_reports(self):
        from radar.agent.research_team import research_candidates
        from radar.config.runtime import MAX_PROMPT_CHARS

        seen = []
        pool = works(25)
        for work in pool:
            work.abstract = "abstract word " * 1000

        def respond(messages, info):
            prompt = next(part.content for message in messages for part in message.parts
                          if isinstance(part, UserPromptPart))
            seen.append((info.instructions, prompt))
            return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name,
                                                     RadarDraft(next_move="Check.").model_dump())])

        result = research_candidates(pool, max_candidates=25, model=FunctionModel(respond))
        self.assertGreater(len(result.included), 0)
        self.assertLess(len(result.included), 25)
        for instructions, prompt in seen:
            self.assertLessEqual(len(instructions) + len(prompt) + 2, MAX_PROMPT_CHARS)
            for i in range(len(result.included)):
                self.assertIn(f"[{i}] Paper {i}", prompt)
            self.assertNotIn(f"[{len(result.included)}]", prompt)
            self.assertIn(f"0..{len(result.included) - 1}", prompt)
        self.assertIn("not source evidence", result.prompt)
        self.assertIn("TASK:", result.prompt.split("end untrusted specialist data ---")[-1])

    def test_pipeline_runs_the_specialists_and_uses_actual_common_coverage(self):
        from pathlib import Path
        from radar.pipeline import PipelineRequest, run
        from radar.storage.snapshots import refresh_pool
        from radar.schema.triage import TriageBatch, TriageResult

        calls = []

        def respond(messages, info):
            calls.append(info.instructions)
            return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name,
                                                     RadarDraft(next_move="Check.").model_dump())])

        def score(pool, profile):
            return TriageBatch(model_id="test", rubric_version="test", results=[
                TriageResult(work_id=w.openalex_id, status="scored",
                             ai_ml_relevance=0.9, cross_domain_potential=0.8) for w in pool])

        with tempfile.TemporaryDirectory() as tmp:
            snapshot = refresh_pool(works(), tmp)["snapshot"]
            before = Path(snapshot).read_bytes()
            result = run(PipelineRequest(mode="analyze", from_snapshot=snapshot,
                                        triage_scorer=score, model_override=FunctionModel(respond)))
            self.assertEqual(Path(snapshot).read_bytes(), before)
        self.assertEqual(result.exit_code, 0, result.stderr_notes)
        self.assertEqual(len(calls), 4)
        self.assertIn("analyzed=2", " ".join(result.stderr_notes))
        self.assertIn("ml_methods,behavioral_economics,evidence_review", " ".join(result.stderr_notes))

    def test_empty_pool_needs_no_model_and_closes_any_supplied_session(self):
        from radar.agent.research_team import research_candidates
        from radar.provider.freetoken import FreeTokenSession

        client = SimpleNamespace(aclose=mock.AsyncMock())
        model = FunctionModel(lambda messages, info: self.fail("no inference expected"))
        result = research_candidates([], session=FreeTokenSession(model, client))
        self.assertEqual(result.included, ())
        self.assertEqual(result.specialist_reports, ())
        client.aclose.assert_awaited_once()

    def test_nonempty_pool_that_does_not_fit_reports_zero_analysis_honestly(self):
        from radar.pipeline import PipelineRequest, run
        from radar.schema.configuration import AnalysisPrompt
        from radar.schema.triage import TriageBatch, TriageResult
        from radar.storage.snapshots import refresh_pool

        spec = AnalysisPrompt(version=1, instructions="x" * 6300,
                              candidate_header="CANDIDATES",
                              task_template="TASK: {max_opportunities}; evidence {valid_range}")
        pool = works(1)
        pool[0].abstract = "word " * 1000
        model = FunctionModel(lambda messages, info: self.fail("no inference expected"))

        def score(pool, profile):
            return TriageBatch(model_id="test", rubric_version="test", results=[
                TriageResult(work_id=w.openalex_id, status="scored", ai_ml_relevance=0.9,
                             cross_domain_potential=0.8) for w in pool])

        with tempfile.TemporaryDirectory() as tmp:
            snapshot = refresh_pool(pool, tmp)["snapshot"]
            with mock.patch("radar.agent.research_team.opportunity_analysis_prompt", return_value=spec):
                result = run(PipelineRequest(mode="analyze", from_snapshot=snapshot,
                                            triage_scorer=score, model_override=model))
        self.assertEqual(result.exit_code, 0, result.stderr_notes)
        self.assertIn("prompt budget", result.stdout)
        self.assertIn("pool=1", " ".join(result.stderr_notes))
        self.assertIn("analyzed=0", " ".join(result.stderr_notes))

    def test_synthesis_deadline_has_no_fresh_budget_and_closes_successful_specialists(self):
        from radar.agent.research_team import research_candidates
        from radar.prompts.catalog import opportunity_analysis_prompt
        from radar.provider.freetoken import FreeTokenError, FreeTokenSession

        roles_done, synthesis_done = [], []

        async def respond(messages, info):
            if info.instructions == opportunity_analysis_prompt().instructions:
                try:
                    await asyncio.Event().wait()
                finally:
                    synthesis_done.append(1)
            roles_done.append(1)
            return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name,
                                                     RadarDraft(next_move="Check.").model_dump())])

        client = SimpleNamespace(aclose=mock.AsyncMock())
        with self.assertRaises(FreeTokenError):
            research_candidates(works(), session=FreeTokenSession(FunctionModel(respond), client),
                                analysis_timeout_s=0.15)
        self.assertEqual(len(roles_done), 3)
        self.assertEqual(synthesis_done, [1])
        client.aclose.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
