"""Configuration behavior through safe parsing, packaged loading and planning."""

from __future__ import annotations

import datetime as dt
import unittest
from unittest import mock

from radar.config.yaml import ConfigurationError, parse_yaml
from radar.schema.configuration import AgentPrompt, AnalysisPrompt, ScreeningConfig, SearchConfig


SCREENING = '''
version: 1
rubric_version: clef-importance-v1
questions:
  research_importance:
    type: noul
    instructions: "Important for {keywords}?"
    criteria:
      "true": "Important."
      "false": "Not important."
  cross_domain_potential:
    type: noul
    instructions: "Transfers to {domains}?"
    criteria:
      "true": "Transfers."
      "false": "No transfer."
'''


class TestTypedYaml(unittest.TestCase):
    def test_unsupported_importance_rubric_fails_configuration_preflight(self):
        with self.assertRaises(ConfigurationError):
            parse_yaml(SCREENING.replace("clef-importance-v1", "clef-importance-v2"), ScreeningConfig)

    def test_questions_are_typed_and_render_without_changing_the_contract(self):
        config = parse_yaml(SCREENING, ScreeningConfig)
        self.assertEqual(config.questions.research_importance.type, "noul")
        rendered = config.render(keywords="diffusion", domains="economics")
        self.assertEqual(rendered["research_importance"], {
            "type": "noul", "instructions": "Important for diffusion?",
            "criteria": {"true": "Important.", "false": "Not important."},
        })
        self.assertEqual(rendered["cross_domain_potential"]["instructions"],
                         "Transfers to economics?")

    def test_invalid_yaml_or_schema_fails_without_echoing_values(self):
        documents = [
            "!!python/object/apply:os.system ['SECRET-CONFIG-VALUE']",
            SCREENING.replace("version: 1", "version: 2"),
            SCREENING.replace("version: 1", "version: true"),
            SCREENING.replace("clef-importance-v1", "x" * 201),
            SCREENING + "unknown: SECRET-CONFIG-VALUE\n",
            SCREENING.replace("version: 1", "version: 1\nversion: 1"),
            SCREENING.replace('"Important."', 'false'),
            SCREENING.replace('"Important."', '"   "'),
            SCREENING.replace("{keywords}", "{keywords.__class__}"),
            SCREENING.replace("{keywords}", "{keywords!r}"),
            SCREENING.replace("{keywords}", "{keywords:>30}"),
            SCREENING.replace("{keywords}", "{unknown}"),
            SCREENING.replace("{keywords}", "{keywords"),
            "questions: [",
            "[]",
            "",
            SCREENING.replace('"true"', 'true'),
        ]
        for document in documents:
            with self.subTest(document=document[:25]):
                with self.assertRaises(ConfigurationError) as caught:
                    parse_yaml(document, ScreeningConfig)
                self.assertNotIn("SECRET-CONFIG-VALUE", str(caught.exception))

    def test_resources_are_typed_and_cannot_be_modified_in_place(self):
        from pydantic import ValidationError
        from radar.config.searches import search_config
        from radar.prompts.catalog import (
            opportunity_analysis_prompt, paper_triage_prompt, screening_questions,
        )

        self.assertIsInstance(search_config(), SearchConfig)
        self.assertIsInstance(opportunity_analysis_prompt(), AnalysisPrompt)
        self.assertIsInstance(paper_triage_prompt(), AgentPrompt)
        self.assertIsInstance(screening_questions(), ScreeningConfig)
        with self.assertRaises(ValidationError):
            screening_questions().questions.research_importance.instructions = "changed"
        # Rendered wire dictionaries are independent of cached defaults.
        first = screening_questions().render(keywords="ml", domains="economics")
        first["research_importance"]["criteria"]["true"] = "changed"
        second = screening_questions().render(keywords="ml", domains="economics")
        self.assertNotEqual(second["research_importance"]["criteria"]["true"], "changed")

    def test_config_size_bound(self):
        with self.assertRaises(ConfigurationError):
            parse_yaml("#" * 65537, ScreeningConfig)


class TestConfiguredSearches(unittest.TestCase):
    def test_default_plan_has_broad_and_cross_domain_searches(self):
        from radar.config.interests import default_profile
        from radar.config.searches import build_query_plan

        plan = build_query_plan(default_profile())
        self.assertEqual(len(plan.queries), 12)
        self.assertIn("machine learning", plan.queries[0].terms)
        intersections = [query for query in plan.queries if " AND " in query.terms]
        self.assertEqual(len(intersections), 8)
        self.assertTrue(any("behavioral economics" in q.terms for q in plan.queries))
        frontier = [query for query in plan.queries if query.role == "frontier"]
        self.assertEqual(len(frontier), 4)
        self.assertNotIn("behavioral", frontier[0].terms)

    def test_config_changes_actual_queries_and_preserves_profile_dates_and_caps(self):
        from radar.config.interests import RadarProfile
        from radar.config.searches import build_query_plan
        from radar.source.openalex import build_request

        config = parse_yaml('''
version: 1
queries:
  - name: configured
    kind: recent
    terms: '({keywords}) AND "decision making"'
    per_page: 50
''', SearchConfig)
        plan = build_query_plan(RadarProfile(keywords=["graph learning"], lookback_days=30),
                                configuration=config, today=dt.date(2026, 10, 3))
        self.assertEqual(plan.queries[0].terms, '("graph learning") AND "decision making"')
        self.assertEqual(plan.queries[0].from_date, "2026-09-03")
        _, params, _, _ = build_request(plan.queries[0])
        self.assertEqual(params["filter"], "from_publication_date:2026-09-03")
        self.assertEqual(params["sort"], "publication_date:desc")
        self.assertEqual(params["per-page"], "50")

    def test_empty_domains_skip_cross_domain_but_keep_broad_searches(self):
        from radar.config.interests import RadarProfile
        from radar.config.searches import build_query_plan

        plan = build_query_plan(RadarProfile(keywords=["diffusion"], domains=[]))
        self.assertEqual([query.kind for query in plan.queries],
                         ["semantic", "semantic", "keyword", "keyword", "keyword", "keyword"])
        self.assertIn("diffusion", plan.queries[0].terms)
        self.assertTrue(all("behavioral" not in query.terms and "generative AI" not in query.terms
                            for query in plan.queries))

    def test_request_cap_cannot_be_raised_by_configuration(self):
        from radar.config.interests import RadarProfile
        from radar.config.searches import build_query_plan

        config = parse_yaml('''
version: 1
queries:
  - name: first
    kind: semantic
    terms: '{keyword}'
    repeat: keywords
    limit: 6
  - name: second
    kind: recent
    terms: '{keyword}'
    repeat: keywords
    limit: 6
''', SearchConfig)
        profile = RadarProfile(keywords=[f"topic {i}" for i in range(10)])
        self.assertEqual(len(build_query_plan(profile, 99, configuration=config).queries), 6)
        self.assertEqual(len(build_query_plan(profile, 1, configuration=config).queries), 1)

    def test_invalid_definitions_reject_bad_placeholders_and_bounds(self):
        template = '''
version: 1
queries:
  - name: broad
    kind: semantic
    terms: '{keywords}'
'''
        documents = [
            template.replace("{keywords}", "{keyword}"),
            template.replace("{keywords}", "{unknown}"),
            template.replace("kind: semantic", "kind: arbitrary"),
            template + "    per_page: 51\n",
            template + "    per_page: true\n",
            template + "    limit: 7\n",
            template + "    limit: 2\n",
            template + "    endpoint: https://example.com\n",
        ]
        for document in documents:
            with self.subTest(document=document):
                with self.assertRaises(ConfigurationError):
                    parse_yaml(document, SearchConfig)

    def test_oversized_query_is_not_silently_truncated(self):
        from radar.config.interests import RadarProfile
        from radar.config.searches import build_query_plan

        with self.assertRaises(ConfigurationError):
            build_query_plan(RadarProfile(keywords=["x" * 301]))


class TestYamlAgentInstructions(unittest.TestCase):
    def test_opportunity_instructions_are_agent_instructions_not_user_data(self):
        from pydantic_ai.messages import ModelResponse, ToolCallPart, UserPromptPart
        from pydantic_ai.models.function import FunctionModel
        from radar.agent.opportunity_analysis import analyze_candidates, build_prompt
        from radar.prompts.catalog import opportunity_analysis_prompt
        from radar.schema.opportunities import RadarDraft
        from radar.schema.papers import CollectedWork

        work = CollectedWork(openalex_id="https://openalex.org/W1", title="Paper",
                             abstract="An experiment about learning.")
        expected = opportunity_analysis_prompt().instructions
        seen = {}

        def respond(messages, info):
            seen["instructions"] = info.instructions
            seen["users"] = [part.content for message in messages
                             for part in getattr(message, "parts", [])
                             if isinstance(part, UserPromptPart)]
            return ModelResponse(parts=[ToolCallPart(
                "final_result", RadarDraft(next_move="Explore.").model_dump())])

        draft, prompt = analyze_candidates([work], model=FunctionModel(respond))
        self.assertEqual(draft.next_move, "Explore.")
        self.assertEqual(seen["instructions"], expected)
        self.assertNotIn(expected, "\n".join(seen["users"]))
        self.assertEqual(prompt, build_prompt([work]))
        self.assertIn("An experiment about learning.", prompt)
        self.assertIn("Valid evidence indices for this run: 0..0", prompt)

    def test_qwen_instructions_and_questions_come_from_the_shared_yaml(self):
        import json
        from pydantic_ai.messages import ModelResponse, ToolCallPart, UserPromptPart
        from pydantic_ai.models.function import FunctionModel
        from radar.agent.paper_triage import screen_works
        from radar.config.interests import RadarProfile
        from radar.prompts.catalog import paper_triage_prompt, screening_questions
        from radar.schema.papers import CollectedWork

        seen = {}

        def respond(messages, info):
            seen["instructions"] = info.instructions
            prompt = next(part.content for message in messages for part in message.parts
                          if isinstance(part, UserPromptPart))
            paper = json.loads(prompt)["papers"][0]
            seen["questions"] = paper["input"]["questions"]
            return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, {"responses": [{
                "work_id": paper["work_id"], "model": paper["input"]["model"],
                "answers": {"research_importance": {"type": "noul", "noul": 0.8},
                            "cross_domain_potential": {"type": "noul", "noul": 0.6}},
            }]})])

        batch = screen_works([CollectedWork(openalex_id="https://openalex.org/W1",
                              title="Title", abstract="Full abstract")],
                             RadarProfile(keywords=["graph learning"], domains=["economics"]),
                             model=FunctionModel(respond), model_id="test")
        self.assertEqual(batch.results[0].status, "scored")
        self.assertEqual(seen["instructions"], paper_triage_prompt().instructions)
        self.assertEqual(seen["questions"], screening_questions().render(
            keywords="graph learning", domains="economics"))

    def test_combined_instructions_and_user_data_stay_within_budget(self):
        from radar.agent.opportunity_analysis import build_prompt, select_for_prompt
        from radar.config.runtime import MAX_PROMPT_CHARS
        from radar.prompts.catalog import opportunity_analysis_prompt
        from radar.schema.papers import CollectedWork

        works = [CollectedWork(openalex_id=f"https://openalex.org/W{i}",
                               title=f"Paper {i}", abstract="word " * 4000)
                 for i in range(25)]
        included = select_for_prompt(works, 25)
        prompt = build_prompt(works, 25)
        self.assertLess(len(included), 25)
        self.assertLessEqual(len(opportunity_analysis_prompt().instructions) + len(prompt) + 2,
                             MAX_PROMPT_CHARS)
        self.assertIn(f"Valid evidence indices for this run: 0..{len(included) - 1}", prompt)
        for i in range(len(included)):
            self.assertIn(f"[{i}] Paper {i}", prompt)


class TestConfigurationPreflight(unittest.TestCase):
    def test_invalid_agent_config_is_exit4_before_source_or_model_calls(self):
        from radar.pipeline import PipelineRequest, run
        from radar.source.openalex import DictTransport

        source = DictTransport({})
        with mock.patch("radar.prompts.catalog.paper_triage_prompt", side_effect=ConfigurationError(
                "paper_triage.yaml: invalid configuration")):
            result = run(PipelineRequest(mode="analyze", source_override=source))
        self.assertEqual(result.exit_code, 4)
        self.assertEqual(source.calls, [])
        self.assertIn("paper_triage.yaml", " ".join(result.stderr_notes))

    def test_collect_only_does_not_need_agent_prompts(self):
        from radar.pipeline import PipelineRequest, run
        from radar.source.openalex import DictTransport

        with mock.patch("radar.prompts.catalog.paper_triage_prompt", side_effect=ConfigurationError()):
            result = run(PipelineRequest(mode="collect", source_override=DictTransport({})))
        self.assertEqual(result.exit_code, 0)


if __name__ == "__main__":
    unittest.main()
