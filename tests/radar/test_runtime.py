"""Runtime reliability: deadlines, per-run bounds, opt-in thinking key,
cached snapshot synthesis, prompt-fit coverage. New caller-facing seams;
TestModel/FunctionModel doubles; stdout captured, never mocked."""

from __future__ import annotations

import asyncio
import contextlib
import io
import json
import os
import tempfile
import unittest
from unittest import mock

from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.models.test import TestModel

from radar.agent import opportunity_analysis as _agent
from radar.cli import main as _main
from radar.pipeline import PipelineRequest, run as _run_pipeline
from radar.processing import ranking as _ranking
from radar.provider import freetoken as _ft
from radar.schema.opportunities import RadarDraft
from radar.schema.papers import CollectedWork
from radar.storage import snapshots as _refresh


def _work(wid: str, title: str = "T", score: float = 0.1,
          abstract: str = "abstract text") -> CollectedWork:
    return CollectedWork(
        openalex_id=wid, title=title, abstract=abstract,
        publication_year=2025, doi="",
        primary_url=wid, locations=[], cited_by_count=1,
        matched_queries=["q"], query_kinds=["semantic"], score=score)


def _seed_snapshot(tmp: str, works: list[CollectedWork]) -> str:
    summary = _refresh.refresh_pool(works, tmp)
    return summary["snapshot"]


def _scored_batch(pool) -> object:
    from types import SimpleNamespace

    return SimpleNamespace(
        model_id="test-clef", rubric_version="r1",
        results=[SimpleNamespace(
            work_id=w.openalex_id, status="scored",
            ai_ml_relevance=0.9, cross_domain_potential=0.1) for w in pool])


def _fixed_draft_model() -> FunctionModel:
    fixed = RadarDraft(opportunities=[], ignore=[], next_move="done")

    def _impl(messages, info):
        return ModelResponse(parts=[ToolCallPart(
            tool_name="final_result", args=fixed.model_dump())])

    return FunctionModel(_impl)


class TestRunBounds(unittest.TestCase):
    def test_defaults_are_bounded(self):
        settings, limits, retries = _agent.run_bounds()
        self.assertEqual(settings["max_tokens"], 2000)
        self.assertLessEqual(settings["timeout"], 90.0)
        self.assertGreater(settings["timeout"], 0)
        self.assertIn(limits.request_limit, (1, 2))
        self.assertEqual(retries, 1)
        self.assertNotIn("extra_body", settings)

    def test_opt_in_thinking_key_only_when_enabled(self):
        settings, _, _ = _agent.run_bounds(disable_thinking=True)
        self.assertEqual(
            settings["extra_body"],
            {"chat_template_kwargs": {"enable_thinking": False}})
        settings2, _, _ = _agent.run_bounds(disable_thinking=False)
        self.assertNotIn("extra_body", settings2)

    def test_invalid_bounds_rejected(self):
        with self.assertRaises(ValueError):
            _agent.run_bounds(max_tokens=0)
        with self.assertRaises(ValueError):
            _agent.run_bounds(request_limit=3)
        with self.assertRaises(ValueError):
            _agent.run_bounds(analysis_timeout_s=0)
        with self.assertRaises(ValueError):
            _agent.run_bounds(analysis_timeout_s=301)

    def test_single_validation_retry_recovers_malformed_envelope(self):
        from pydantic_ai.messages import ModelResponse, ToolCallPart

        calls = []

        def _flaky(messages, info):
            calls.append(1)
            if len(calls) == 1:
                args = {"opportunities": None, "ignore": [],
                        "next_move": "n"}
            else:
                args = {"opportunities": [], "ignore": [], "next_move": "n"}
            return ModelResponse(parts=[ToolCallPart(
                tool_name="final_result", args=args)])

        draft, _ = _agent.analyze_candidates(
            [_work("https://openalex.org/W1")],
            model=FunctionModel(_flaky))
        self.assertEqual(draft.opportunities, [])
        self.assertEqual(len(calls), 2)


class TestDeadline(unittest.TestCase):
    def test_slow_model_hits_overall_deadline(self):
        async def _slow(messages, info):
            await asyncio.sleep(5)
            return RadarDraft(opportunities=[], ignore=[], next_move="x")

        with self.assertRaises(_ft.FreeTokenError) as ctx:
            _agent.analyze_candidates(
                [_work("https://openalex.org/W1")],
                model=FunctionModel(_slow), analysis_timeout_s=0.3)
        self.assertIn("deadline", str(ctx.exception).lower())

    def test_fast_model_succeeds_with_test_model(self):
        draft, prompt = _agent.analyze_candidates(
            [_work("https://openalex.org/W1")], model=TestModel())
        self.assertIsInstance(draft, RadarDraft)
        self.assertIn("at most 2 opportunities", prompt)


class TestDisableThinkingResolution(unittest.TestCase):
    def test_env_opt_in(self):
        with mock.patch.dict(os.environ, {"FREETOKEN_DISABLE_THINKING": "yes"}):
            self.assertTrue(_ft.resolve_disable_thinking(False))
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("FREETOKEN_DISABLE_THINKING", None)
            self.assertFalse(_ft.resolve_disable_thinking(False))
            self.assertTrue(_ft.resolve_disable_thinking(True))

    def test_server_rejection_is_clear(self):
        def _reject(messages, info):
            raise RuntimeError("400 Bad Request: unrecognized extra_body key")

        with self.assertRaises(_ft.FreeTokenError) as ctx:
            _agent.analyze_candidates(
                [_work("https://openalex.org/W1")],
                model=FunctionModel(_reject), disable_thinking=True)
        self.assertIn("--disable-thinking", str(ctx.exception))


class TestCachedSnapshot(unittest.TestCase):
    def test_load_snapshot_returns_ranked_works_and_meta(self):
        with tempfile.TemporaryDirectory() as tmp:
            works = [_work("https://openalex.org/W2", score=0.2),
                     _work("https://openalex.org/W1", score=0.9)]
            path = _seed_snapshot(tmp, works)
            loaded, meta = _refresh.load_snapshot(path)
            self.assertEqual(len(loaded), 2)
            self.assertTrue(meta["collected_at_utc"])
            top = _ranking.select_topn(loaded, 1)
            self.assertEqual(top[0].openalex_id, "https://openalex.org/W1")

    def test_load_snapshot_rejects_invalid(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "snapshot.json")
            with open(path, "w") as fh:
                fh.write('{"schema_version": 999, "works": []}')
            with self.assertRaises(_refresh.RefreshError):
                _refresh.load_snapshot(path)

    def test_from_snapshot_collect_only_needs_no_network(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _seed_snapshot(tmp, [_work("https://openalex.org/W1"),
                                        _work("https://openalex.org/W2", abstract="")])
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                code = _main(["--collect-only", "--from-snapshot", path,
                              "--max-candidates", "1"])
            self.assertEqual(code, 0)
            shown = json.loads(buf.getvalue())
            self.assertEqual(shown["coverage"]["collected"], 2)
            self.assertEqual(len(shown["works"]), 1)

    def test_from_snapshot_plus_refresh_dir_rejected(self):
        self.assertEqual(
            _main(["--collect-only", "--from-snapshot", "x.json",
                   "--refresh-dir", "y"]), 4)

    def test_missing_snapshot_is_usage_error(self):
        self.assertEqual(
            _main(["--collect-only", "--from-snapshot",
                   "/nonexistent/snap.json"]), 4)

    def test_invalid_snapshot_fails_before_model_and_preserves(self):
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "snapshot.json"
            path.write_text("{broken", encoding="utf-8")
            before = path.read_text(encoding="utf-8")
            code = _main(["--from-snapshot", str(path)])
            self.assertNotEqual(code, 0)
            self.assertEqual(path.read_text(encoding="utf-8"), before)

    def test_invalid_analysis_timeout_flag(self):
        self.assertEqual(_main(["--analysis-timeout", "0"]), 4)
        self.assertEqual(_main(["--analysis-timeout", "301"]), 4)

    def test_cached_analysis_never_rewrites_snapshot(self):
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            path = _seed_snapshot(tmp, [_work("https://openalex.org/W1")])
            before = Path(path).read_bytes()
            result = _run_pipeline(PipelineRequest(
                mode="analyze", max_candidates=1, from_snapshot=path,
                model_override=_fixed_draft_model(),
                triage_scorer=lambda pool, profile: _scored_batch(pool)))
            self.assertEqual(result.exit_code, 0)
            self.assertIn("Next move", result.stdout)
            self.assertEqual(Path(path).read_bytes(), before)

    def test_pipeline_reports_full_cache_pool_and_actual_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _seed_snapshot(tmp, [_work(f"https://openalex.org/W{i}")
                                        for i in range(5)])
            result = _run_pipeline(PipelineRequest(
                mode="analyze", max_candidates=2, from_snapshot=path,
                model_override=_fixed_draft_model(),
                triage_scorer=lambda pool, profile: _scored_batch(pool)))
            self.assertEqual(result.exit_code, 0)
            coverage = " ".join(result.stderr_notes)
            self.assertIn("pool=5", coverage)
            self.assertIn("selected=2", coverage)
            self.assertIn("analyzed=2", coverage)


class TestCoverageGuard(unittest.TestCase):
    def test_stale_snapshot_coverage_update_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            _refresh.refresh_pool([_work("https://openalex.org/W1")], tmp,
                                  collected_at="2026-01-01T00:00:00+00:00")
            with self.assertRaises(_refresh.RefreshError):
                _refresh.update_llm_coverage(
                    tmp, 1, 0, expect_collected_at="2099-01-01T00:00:00+00:00")


class TestValidationCategories(unittest.TestCase):
    def test_invalid_model_output_reports_safe_categories(self):
        def _invalid(messages, info):
            return ModelResponse(parts=[ToolCallPart(
                tool_name="final_result",
                args={"opportunities": [{
                    "title": "T", "wow": "W", "investigate": "I",
                    "reproduce": "R",
                    "evidence": ["MARKER-BAD-VALUE-9z"]}],
                    "ignore": [], "next_move": "n"})])

        with self.assertRaises(_ft.FreeTokenError) as ctx:
            _agent.analyze_candidates(
                [_work("https://openalex.org/W1")],
                model=FunctionModel(_invalid))
        text = str(ctx.exception)
        self.assertIn("evidence", text)
        self.assertNotIn("MARKER-BAD-VALUE-9z", text)

    def test_prompt_states_valid_evidence_range(self):
        prompt = _agent.build_prompt(
            [_work(f"https://openalex.org/W{i}") for i in range(3)],
            max_candidates=3)
        self.assertIn("0..2", prompt)

    def test_non_string_ignore_items_coerced(self):
        def _ints_in_ignore(messages, info):
            return ModelResponse(parts=[ToolCallPart(
                tool_name="final_result",
                args={"opportunities": [], "ignore": [7, None, "dup"],
                      "next_move": "n"})])

        draft, _ = _agent.analyze_candidates(
            [_work("https://openalex.org/W1")],
            model=FunctionModel(_ints_in_ignore))
        self.assertEqual(draft.ignore, ["7", "dup"])


class TestSessionLifecycle(unittest.TestCase):
    def _session(self, model):
        import httpx2

        from radar.provider.freetoken import FreeTokenSession

        return FreeTokenSession(
            model=model,
            http_client=httpx2.AsyncClient(timeout=5.0))

    def test_deadline_closes_session_in_loop(self):
        async def _slow(messages, info):
            await asyncio.sleep(5)
            return RadarDraft(opportunities=[], ignore=[], next_move="x")

        session = self._session(FunctionModel(_slow))
        with self.assertRaises(_ft.FreeTokenError):
            _agent.analyze_candidates(
                [_work("https://openalex.org/W1")],
                analysis_timeout_s=0.3, session=session)
        self.assertTrue(session.http_client.is_closed)

    def test_early_bounds_error_closes_session(self):
        session = self._session(TestModel())
        with self.assertRaises(ValueError):
            _agent.analyze_candidates(
                [_work("https://openalex.org/W1")],
                max_tokens=0, session=session)
        self.assertTrue(session.http_client.is_closed)

    def test_success_closes_session(self):
        session = self._session(TestModel())
        _agent.analyze_candidates(
            [_work("https://openalex.org/W1")], session=session)
        self.assertTrue(session.http_client.is_closed)


class TestPromptFit(unittest.TestCase):
    def test_footer_survives_and_only_complete_blocks_count(self):
        long_abstract = "word " * 4000
        works = [_work(f"https://openalex.org/W{i}", title=f"Paper {i}",
                       abstract=long_abstract) for i in range(25)]
        included = _agent.select_for_prompt(works, 25)
        self.assertLess(len(included), 25)
        self.assertGreater(len(included), 0)
        prompt = _agent.build_prompt(works, 25)
        self.assertIn("TASK:", prompt)
        self.assertIn(f"0..{len(included) - 1}", prompt)
        # Every claimed index resolves against the included list.
        for i in range(len(included)):
            self.assertIn(f"[{i}]", prompt)


if __name__ == "__main__":
    unittest.main()
