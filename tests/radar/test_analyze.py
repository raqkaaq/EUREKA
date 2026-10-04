"""Tests for structured PydanticAI analysis wiring, evidence attachment,
Markdown rendering, prompt bounds, and the CLI seams.

LLM behavior is exercised via PydanticAI's injected TestModel/FunctionModel
only — no network, no real inference.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import unittest
from unittest import mock

from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel

from radar.agent.opportunity_analysis import (
    analyze_candidates,
    build_agent,
    build_prompt,
)
from radar.cli import main
from radar.config.runtime import MAX_CANDIDATES_IN_PROMPT, MAX_PROMPT_CHARS
from radar.output.json import work_to_json
from radar.output.markdown import render_markdown
from radar.processing.evidence import attach_evidence
from radar.processing.ranking import bound_candidates
from radar.schema.opportunities import (
    OpportunityDraft,
    RadarDraft,
)
from radar.schema.papers import CollectedWork


def _candidate(i: int = 0) -> CollectedWork:
    return CollectedWork(
        openalex_id=f"https://openalex.org/W{i}",
        title=f"Paper {i} on diffusion models",
        abstract="Abstract about diffusion models and behavioral incentives. " * 5,
        publication_year=2024,
        doi=f"https://doi.org/10.1234/{i}",
        primary_url=f"https://example.org/paper-{i}",
        cited_by_count=10 + i,
        matched_queries=["diffusion models"],
        query_kinds=["semantic"],
        score=0.9,
    )


class TestStructuredOutputWiring(unittest.TestCase):
    def test_testmodel_returns_validated_radar_draft(self):
        """PydanticAI output wiring: Agent(output_type=RadarDraft) yields RadarDraft."""
        candidates = [_candidate(0), _candidate(1)]
        draft, prompt = analyze_candidates(candidates, model=TestModel())
        self.assertIsInstance(draft, RadarDraft)
        self.assertIsInstance(prompt, str)
        # Round-trip through validation (proves Pydantic-structured output).
        self.assertEqual(RadarDraft.model_validate(draft.model_dump()), draft)

    def test_function_model_fixed_draft_flows_through(self):
        from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart

        fixed = RadarDraft(
            opportunities=[
                OpportunityDraft(
                    title="Cross-domain transfer",
                    wow="Surprising link.",
                    investigate="Run an RCT.",
                    reproduce="Re-run notebook.",
                    evidence=[0, 7, -1],  # 7/-1 invalid -> dropped deterministically
                )
            ],
            ignore=["duplicate of W0"],
            next_move="Fetch more recent works.",
        )
        seen: dict = {}

        def _impl(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            seen["n"] = len(messages)
            # Structured-output path: TestModel machinery validates via tool
            # calls; here echo the fixed draft as JSON text is NOT enough, so
            # delegate: use TestModel-like validation by returning via agent?
            # Instead emulate the output tool call the agent expects.
            from pydantic_ai.messages import ToolCallPart

            return ModelResponse(
                parts=[ToolCallPart(tool_name="final_result", args=fixed.model_dump())]
            )

        agent = build_agent(FunctionModel(_impl))
        result = asyncio.run(agent.run("hello"))
        self.assertIsInstance(result.output, RadarDraft)
        self.assertEqual(result.output.next_move, "Fetch more recent works.")
        report = attach_evidence(result.output, [_candidate(0)])
        self.assertEqual(len(report.opportunities), 1)
        # Only index 0 survives; 7 and -1 are dropped.
        self.assertEqual([l.index for l in report.opportunities[0].evidence_links], [0])
        self.assertEqual(
            report.opportunities[0].evidence_links[0].url, "https://example.org/paper-0"
        )

    def test_inference_failure_becomes_actionable(self):
        def _boom(messages, info):
            raise ConnectionRefusedError("refused")

        with self.assertRaises(Exception) as ctx:
            analyze_candidates([_candidate(0)], model=FunctionModel(_boom))
        self.assertIn("FREETOKEN_BASE_URL", str(ctx.exception))


class TestEvidenceAndRendering(unittest.TestCase):
    def test_evidence_links_are_deterministic(self):
        draft = RadarDraft(
            opportunities=[
                OpportunityDraft(title="T", wow="W", investigate="I", reproduce="R", evidence=[1, 1, 0])
            ],
            ignore=[],
            next_move="N",
        )
        report = attach_evidence(draft, [_candidate(0), _candidate(1)])
        links = report.opportunities[0].evidence_links
        # Deterministic order preserved, duplicates removed.
        self.assertEqual([l.index for l in links], [1, 0])
        self.assertEqual(links[0].url, "https://example.org/paper-1")

    def test_markdown_sections_present(self):
        draft = RadarDraft(
            opportunities=[
                OpportunityDraft(title="T", wow="W", investigate="I", reproduce="R", evidence=[0])
            ],
            ignore=["weak paper"],
            next_move="Do X.",
        )
        md = render_markdown(attach_evidence(draft, [_candidate(0)]))
        for section in ("The Wow", "Investigate", "Reproduce", "Ignore", "Next move", "Evidence"):
            self.assertIn(section, md)
        self.assertIn("https://example.org/paper-0", md)

    def test_empty_report_renders(self):
        md = render_markdown(attach_evidence(RadarDraft(), []))
        for section in ("Ignore", "Next move"):
            self.assertIn(section, md)

    def test_untrusted_evidence_title_cannot_inject_markdown_link(self):
        candidate = _candidate(0).model_copy(
            update={"title": "Paper](https://evil.example) [label"}
        )
        draft = RadarDraft(
            opportunities=[
                OpportunityDraft(
                    title="T", wow="W", investigate="I", reproduce="R", evidence=[0]
                )
            ]
        )
        md = render_markdown(attach_evidence(draft, [candidate]))
        self.assertIn(r"Paper\](https://evil.example) \[label", md)
        self.assertNotIn("[Paper](https://evil.example)", md)

    def test_unsafe_evidence_url_falls_back_to_openalex(self):
        candidate = _candidate(0).model_copy(update={"primary_url": "javascript:alert(1)"})
        draft = RadarDraft(
            opportunities=[
                OpportunityDraft(
                    title="T", wow="W", investigate="I", reproduce="R", evidence=[0]
                )
            ]
        )
        report = attach_evidence(draft, [candidate])
        self.assertEqual(
            report.opportunities[0].evidence_links[0].url,
            candidate.openalex_id,
        )


class TestPromptBounds(unittest.TestCase):
    def test_prompt_bounded(self):
        candidates = [_candidate(i) for i in range(50)]
        prompt = build_prompt(candidates, max_candidates=50)
        self.assertLessEqual(len(prompt), MAX_PROMPT_CHARS)
        # Candidate cap enforced (indices only up to cap-1).
        self.assertNotIn("[25]", prompt)
        self.assertIn("[0]", prompt)

    def test_prompt_cap_covers_cli_max(self):
        from radar.config.runtime import MAX_MAX_CANDIDATES

        self.assertGreaterEqual(MAX_CANDIDATES_IN_PROMPT, MAX_MAX_CANDIDATES)

    def test_prompt_never_empty(self):
        prompt = build_prompt([], max_candidates=8)
        self.assertIn("CANDIDATES", prompt)

    def test_prompt_marks_candidates_untrusted_and_no_instruction_following(self):
        from radar.prompts.catalog import opportunity_analysis_prompt

        lowered = opportunity_analysis_prompt().instructions.lower()
        self.assertIn("untrusted", lowered)
        self.assertIn("never follow", lowered)
        prompt = build_prompt([_candidate(0)], max_candidates=8)
        self.assertIn("untrusted", prompt.lower())
        self.assertIn("begin untrusted candidate 0 data", prompt.lower())
        self.assertIn("end untrusted candidate 0 data", prompt.lower())

    def test_prompt_and_evidence_share_bounded_set_above_index_11(self):
        candidates = [_candidate(i) for i in range(15)]
        prompt = build_prompt(candidates, max_candidates=15)
        self.assertIn("[14]", prompt)
        bounded = bound_candidates(candidates, 15)
        self.assertEqual(len(bounded), 15)
        draft = RadarDraft(
            opportunities=[
                OpportunityDraft(
                    title="T", wow="W", investigate="I", reproduce="R", evidence=[14]
                )
            ],
            ignore=[],
            next_move="N",
        )
        report = attach_evidence(draft, bounded)
        links = report.opportunities[0].evidence_links
        self.assertEqual([l.index for l in links], [14])
        self.assertEqual(links[0].url, "https://example.org/paper-14")
        self.assertIn("paper 14", links[0].title.lower())

def _all_scored_scorer(pool, profile):
    from types import SimpleNamespace

    assert profile.keywords, "active profile keywords must reach the scorer"
    scored = []
    for pos, w in enumerate(pool):
        scored.append(SimpleNamespace(
            work_id=w.openalex_id, status="scored",
            ai_ml_relevance=round(0.99 - 0.01 * pos, 4),
            cross_domain_potential=0.1))
    return SimpleNamespace(
        model_id="test-clef", rubric_version="r1", results=scored)


class TestCliSeams(unittest.TestCase):
    def test_collect_bounds_validated(self):
        from radar.pipeline import PipelineRequest, run

        for bad in (dict(max_candidates=0), dict(max_candidates=26),
                    dict(timeout_s=999.0), dict(lookback_days=0)):
            result = run(PipelineRequest(mode="collect", **bad))
            self.assertEqual(result.exit_code, 4)

    def test_work_to_json_keys(self):
        payload = work_to_json(_candidate(3))
        self.assertEqual(payload["openalex_id"], "https://openalex.org/W3")
        self.assertLessEqual(len(payload["abstract"]), 2000)

    def test_collect_only_prints_bounded_json(self):
        import contextlib
        import io

        from radar.pipeline import PipelineRequest, run
        from radar.config.searches import build_query_plan
        from radar.source.openalex import DictTransport
        from radar.config.interests import default_profile

        plan = build_query_plan(default_profile())
        first_terms = plan.queries[0].terms
        pages = {first_terms: {"results": [{
            "id": f"https://openalex.org/W{i}",
            "title": f"Work {i}",
            "abstract_inverted_index": {"x": [0]},
            "doi": "", "publication_year": 2026,
            "publication_date": dt.date.today().isoformat(), "cited_by_count": 0,
        } for i in range(3)]}}
        result = run(PipelineRequest(
            mode="collect", max_candidates=8, source_override=DictTransport(pages)))
        self.assertEqual(result.exit_code, 0)
        parsed = json.loads(result.stdout)
        self.assertEqual(len(parsed), 3)
        self.assertEqual(parsed[0]["openalex_id"], "https://openalex.org/W0")
        # Thin CLI prints the pipeline payload unchanged.
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = main(["--collect-only", "--from-snapshot",
                         "/nonexistent.json"])
        self.assertEqual(code, 4)

    def test_openalex_failure_is_clear_external_error(self):
        from radar.pipeline import PipelineRequest, run

        class _Down:
            def get_json(self, url, params, headers, timeout):
                raise TimeoutError("slow")

        result = run(PipelineRequest(mode="collect", source_override=_Down()))
        self.assertEqual(result.exit_code, 2)
        self.assertTrue(any("OpenAlex" in note for note in result.stderr_notes))

    def test_public_base_url_rejected_before_inference(self):
        import socket

        from radar.pipeline import PipelineRequest, run
        from radar.config.searches import build_query_plan
        from radar.source.openalex import DictTransport
        from radar.config.interests import default_profile

        plan = build_query_plan(default_profile())
        pages = {plan.queries[0].terms: {"results": [{
            "id": "https://openalex.org/W0", "title": "T",
            "abstract_inverted_index": {"x": [0]},
            "doi": "", "publication_year": 2026,
            "publication_date": dt.date.today().isoformat(), "cited_by_count": 0}]}}
        public_answer = [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 0))
        ]
        with mock.patch("socket.getaddrinfo", return_value=public_answer):
            result = run(PipelineRequest(
                mode="analyze", max_candidates=1,
                base_url="https://api.openai.com/v1",
                source_override=DictTransport(pages),
                triage_scorer=_all_scored_scorer))
        self.assertEqual(result.exit_code, 3)

    def test_analysis_report_flow_retains_evidence_above_index_11(self):
        """Regression: evidence attachment uses the same bounded set as the LLM.

        With >12 candidates and --max-candidates 15, a selection of index 14
        must retain its deterministic URL/title (old works[:12] dropped it).
        """
        from pydantic_ai.messages import ModelResponse, ToolCallPart

        from radar.pipeline import PipelineRequest, run
        from radar.config.searches import build_query_plan
        from radar.source.openalex import DictTransport
        from radar.config.interests import default_profile
        from pydantic_ai.models.function import FunctionModel

        draft = RadarDraft(
            opportunities=[
                OpportunityDraft(
                    title="T", wow="W", investigate="I", reproduce="R", evidence=[14]
                )
            ],
            ignore=[],
            next_move="N",
        )

        def _impl(messages, info):
            return ModelResponse(parts=[ToolCallPart(
                tool_name="final_result", args=draft.model_dump())])

        plan = build_query_plan(default_profile(), max_queries=6)
        pages = {}
        for qi, q in enumerate(plan.queries):
            pages.setdefault(q.terms, {"results": []})["results"].extend([{
                "id": f"https://openalex.org/W{qi * 3 + j}",
                "title": f"Paper {qi * 3 + j}",
                "abstract_inverted_index": {"x": [0]},
                # Distinct citations force ranked order == numeric order,
                # so fixed evidence index 14 resolves to W14.
                "doi": "", "publication_year": 2026,
                "publication_date": dt.date.today().isoformat(),
                "cited_by_count": 100 - (qi * 3 + j),
            } for j in range(3)])
        result = run(PipelineRequest(
            mode="analyze", max_candidates=15,
            source_override=DictTransport(pages),
            model_override=FunctionModel(_impl),
            triage_scorer=_all_scored_scorer))
        self.assertEqual(result.exit_code, 0)
        self.assertIn("https://openalex.org/W14", result.stdout)

    def test_report_generation_failure_is_clear(self):
        from radar.pipeline import PipelineRequest, run
        from radar.config.searches import build_query_plan
        from radar.source.openalex import DictTransport
        from radar.config.interests import default_profile
        from pydantic_ai.models.function import FunctionModel

        def _empty(messages, info):
            from pydantic_ai.messages import ModelResponse, ToolCallPart
            return ModelResponse(parts=[ToolCallPart(
                tool_name="final_result",
                args=RadarDraft().model_dump())])

        plan = build_query_plan(default_profile())
        pages = {plan.queries[0].terms: {"results": [{
            "id": "https://openalex.org/W0", "title": "T",
            "abstract_inverted_index": {"x": [0]},
            "doi": "", "publication_year": 2026,
            "publication_date": dt.date.today().isoformat(), "cited_by_count": 0}]}}
        import radar.pipeline as _pipeline

        with mock.patch.object(_pipeline, "attach_evidence",
                               side_effect=RuntimeError("boom")):
            result = run(PipelineRequest(
                mode="analyze", max_candidates=1,
                source_override=DictTransport(pages),
                model_override=FunctionModel(_empty),
                triage_scorer=_all_scored_scorer))
        self.assertEqual(result.exit_code, 3)
        self.assertTrue(any("Report generation failed" in note
                            for note in result.stderr_notes))


class TestRetryingTransport(unittest.TestCase):
    def test_429_then_success(self):
        import urllib.error

        from radar.source.openalex import RetryingTransport

        calls = {"n": 0}

        class Flaky:
            def get_json(self, url, params, headers, timeout):
                calls["n"] += 1
                if calls["n"] == 1:
                    raise urllib.error.HTTPError(
                        url, 429, "throttled", {"Retry-After": "0"}, None
                    )
                return {"results": []}

        with mock.patch("time.sleep") as nap:
            out = RetryingTransport(Flaky()).get_json("http://x", {}, {}, 5.0)
        self.assertEqual(out, {"results": []})
        self.assertEqual(calls["n"], 2)
        nap.assert_called_once()

    def test_non_429_propagates_immediately(self):
        import urllib.error

        from radar.source.openalex import RetryingTransport

        class Bad:
            def get_json(self, url, params, headers, timeout):
                raise urllib.error.HTTPError(url, 400, "bad", {}, None)

        with mock.patch("time.sleep") as nap:
            with self.assertRaises(urllib.error.HTTPError):
                RetryingTransport(Bad()).get_json("http://x", {}, {}, 5.0)
        nap.assert_not_called()

    def test_retries_bounded(self):
        import urllib.error

        from radar.source.openalex import RetryingTransport

        calls = {"n": 0}

        class Always429:
            def get_json(self, url, params, headers, timeout):
                calls["n"] += 1
                raise urllib.error.HTTPError(url, 429, "t", {"Retry-After": "0"}, None)

        with mock.patch("time.sleep"):
            with self.assertRaises(urllib.error.HTTPError):
                RetryingTransport(Always429(), max_retries=2).get_json(
                    "http://x", {}, {}, 5.0
                )
        self.assertEqual(calls["n"], 3)  # initial + 2 retries

    def test_429_missing_retry_after_uses_small_backoff(self):
        import urllib.error

        from radar.source.openalex import DEFAULT_RETRY_AFTER_S, MAX_RETRY_AFTER_S, RetryingTransport

        class MissingHeader:
            def get_json(self, url, params, headers, timeout):
                raise urllib.error.HTTPError(url, 429, "t", {}, None)

        with mock.patch("time.sleep") as nap:
            with self.assertRaises(urllib.error.HTTPError):
                RetryingTransport(MissingHeader(), max_retries=1).get_json(
                    "http://x", {}, {}, 5.0
                )
        waits = [c.args[0] for c in nap.call_args_list]
        self.assertEqual(len(waits), 1)
        self.assertEqual(waits[0], DEFAULT_RETRY_AFTER_S)
        self.assertLess(waits[0], MAX_RETRY_AFTER_S)

    def test_429_malformed_retry_after_uses_small_backoff(self):
        import urllib.error

        from radar.source.openalex import DEFAULT_RETRY_AFTER_S, RetryingTransport

        for bad in ("not-a-number", "", "abc", "NaN", "-5x"):
            with self.subTest(header=bad):

                class Malformed:
                    def get_json(self, url, params, headers, timeout):
                        raise urllib.error.HTTPError(
                            url, 429, "t", {"Retry-After": bad}, None
                        )

                with mock.patch("time.sleep") as nap:
                    with self.assertRaises(urllib.error.HTTPError):
                        RetryingTransport(Malformed(), max_retries=1).get_json(
                            "http://x", {}, {}, 5.0
                        )
                self.assertEqual(nap.call_args.args[0], DEFAULT_RETRY_AFTER_S)

    def test_429_valid_retry_after_capped(self):
        import urllib.error

        from radar.source.openalex import MAX_RETRY_AFTER_S, RetryingTransport

        class Capped:
            def get_json(self, url, params, headers, timeout):
                raise urllib.error.HTTPError(
                    url, 429, "t", {"Retry-After": "120"}, None
                )

        with mock.patch("time.sleep") as nap:
            with self.assertRaises(urllib.error.HTTPError):
                RetryingTransport(Capped(), max_retries=1).get_json(
                    "http://x", {}, {}, 5.0
                )
        self.assertEqual(nap.call_args.args[0], MAX_RETRY_AFTER_S)

    def test_429_valid_retry_after_honored(self):
        import urllib.error

        from radar.source.openalex import RetryingTransport

        class Honest:
            calls = 0

            def get_json(self, url, params, headers, timeout):
                type(self).calls += 1
                if type(self).calls == 1:
                    raise urllib.error.HTTPError(
                        url, 429, "t", {"Retry-After": "3"}, None
                    )
                return {"results": []}

        Honest.calls = 0
        with mock.patch("time.sleep") as nap:
            out = RetryingTransport(Honest()).get_json("http://x", {}, {}, 5.0)
        self.assertEqual(out, {"results": []})
        self.assertEqual(nap.call_args.args[0], 3.0)


if __name__ == "__main__":
    unittest.main()
