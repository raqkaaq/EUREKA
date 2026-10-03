"""Runtime reliability slice: deadlines, per-run bounds, opt-in thinking key,
cached snapshot synthesis. Public seams only; TestModel/FunctionModel doubles."""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import unittest
from unittest import mock

from pydantic_ai.models.function import FunctionModel
from pydantic_ai.models.test import TestModel

from radar import analyze as _analyze
from radar import cli as _cli
from radar import freetoken as _ft
from radar import refresh as _refresh
from radar.models import CollectedWork, LocationInfo, RadarDraft


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


class TestRunBounds(unittest.TestCase):
    def test_defaults_are_bounded(self):
        settings, limits, retries = _analyze.run_bounds()
        self.assertEqual(settings["max_tokens"], 2000)
        self.assertLessEqual(settings["timeout"], 90.0)
        self.assertGreater(settings["timeout"], 0)
        self.assertIn(limits.request_limit, (1, 2))
        self.assertEqual(retries, 0)
        self.assertNotIn("extra_body", settings)

    def test_opt_in_thinking_key_only_when_enabled(self):
        settings, _, _ = _analyze.run_bounds(disable_thinking=True)
        self.assertEqual(
            settings["extra_body"],
            {"chat_template_kwargs": {"enable_thinking": False}})
        settings2, _, _ = _analyze.run_bounds(disable_thinking=False)
        self.assertNotIn("extra_body", settings2)

    def test_invalid_bounds_rejected(self):
        with self.assertRaises(ValueError):
            _analyze.run_bounds(max_tokens=0)
        with self.assertRaises(ValueError):
            _analyze.run_bounds(request_limit=3)
        with self.assertRaises(ValueError):
            _analyze.run_bounds(analysis_timeout_s=0)
        with self.assertRaises(ValueError):
            _analyze.run_bounds(analysis_timeout_s=301)


class TestDeadline(unittest.TestCase):
    def test_slow_model_hits_overall_deadline(self):
        async def _slow(messages, info):
            await asyncio.sleep(5)
            return _analyze.RadarDraft(opportunities=[], ignore=[], next_move="x")

        with self.assertRaises(_ft.FreeTokenError) as ctx:
            _analyze.analyze_candidates(
                [_work("https://openalex.org/W1")],
                model=FunctionModel(_slow), analysis_timeout_s=0.3)
        self.assertIn("deadline", str(ctx.exception).lower())

    def test_fast_model_succeeds_with_test_model(self):
        draft, prompt = _analyze.analyze_candidates(
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
            _analyze.analyze_candidates(
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
            top = _refresh.select_topn(loaded, 1)
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
            with mock.patch.object(_cli, "collect_pool",
                                   side_effect=AssertionError("no network")):
                with mock.patch("builtins.print") as fake_print:
                    code = _cli.main(["--collect-only", "--from-snapshot", path,
                                      "--max-candidates", "1"])
            self.assertEqual(code, 0)
            shown = json.loads(fake_print.call_args[0][0])
            self.assertEqual(shown["coverage"]["collected"], 2)
            self.assertEqual(len(shown["works"]), 1)

    def test_from_snapshot_plus_refresh_dir_rejected(self):
        code = _cli.main(["--collect-only", "--from-snapshot", "x.json",
                          "--refresh-dir", "y"])
        self.assertEqual(code, 4)

    def test_missing_snapshot_is_usage_error(self):
        code = _cli.main(["--collect-only", "--from-snapshot",
                          "/nonexistent/snap.json"])
        self.assertEqual(code, 4)

    def test_invalid_snapshot_fails_before_model_and_preserves(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "snapshot.json")
            with open(path, "w") as fh:
                fh.write("{broken")
            before = open(path).read()
            with mock.patch.object(_cli, "analyze_candidates",
                                   side_effect=AssertionError("no model")):
                code = _cli.main(["--from-snapshot", path])
            self.assertNotEqual(code, 0)
            self.assertEqual(open(path).read(), before)

    def test_invalid_analysis_timeout_flag(self):
        self.assertEqual(_cli.main(["--analysis-timeout", "0"]), 4)
        self.assertEqual(_cli.main(["--analysis-timeout", "301"]), 4)

    def test_cached_analysis_never_rewrites_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _seed_snapshot(tmp, [_work("https://openalex.org/W1")])
            before = open(path, "rb").read()
            draft = RadarDraft(opportunities=[], ignore=[], next_move="done")
            with mock.patch.object(_cli, "analyze_candidates",
                                   return_value=(draft, "p")):
                with mock.patch.object(_cli._freetoken.FreeTokenConfig,
                                       "resolve", return_value=mock.Mock()):
                    with mock.patch.object(_cli._freetoken, "build_session",
                                           return_value=mock.Mock(model=object())):
                        with mock.patch.object(_cli._freetoken, "close_session"):
                            with mock.patch("builtins.print"):
                                code = _cli.main(["--from-snapshot", path])
            self.assertEqual(code, 0)
            self.assertEqual(open(path, "rb").read(), before)


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
        from pydantic_ai.messages import ModelResponse, ToolCallPart

        def _invalid(messages, info):
            return ModelResponse(parts=[ToolCallPart(
                tool_name="final_result",
                args={"opportunities": [{
                    "title": "T", "wow": "W", "investigate": "I",
                    "reproduce": "R",
                    "evidence": ["MARKER-BAD-VALUE-9z"]}],
                    "ignore": [], "next_move": "n"})])

        with self.assertRaises(_ft.FreeTokenError) as ctx:
            _analyze.analyze_candidates(
                [_work("https://openalex.org/W1")],
                model=FunctionModel(_invalid))
        text = str(ctx.exception)
        self.assertIn("evidence", text)
        self.assertNotIn("MARKER-BAD-VALUE-9z", text)

    def test_prompt_states_valid_evidence_range(self):
        from radar.prompt import build_prompt

        prompt = build_prompt(
            [_work(f"https://openalex.org/W{i}") for i in range(3)],
            max_candidates=3)
        self.assertIn("0..2", prompt)

    def test_non_string_ignore_items_coerced(self):
        from pydantic_ai.messages import ModelResponse, ToolCallPart

        def _ints_in_ignore(messages, info):
            return ModelResponse(parts=[ToolCallPart(
                tool_name="final_result",
                args={"opportunities": [], "ignore": [7, None, "dup"],
                      "next_move": "n"})])

        draft, _ = _analyze.analyze_candidates(
            [_work("https://openalex.org/W1")],
            model=FunctionModel(_ints_in_ignore))
        self.assertEqual(draft.ignore, ["7", "dup"])


if __name__ == "__main__":
    unittest.main()
