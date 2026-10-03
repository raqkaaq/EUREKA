"""Mandatory routing integration: preferred CLEF, typed Qwen chat fallback,
config validation, sidecars, coverage, and failed-both-backend safeguards.
External HTTP/model boundaries are substituted; no services or .env reads.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models.function import FunctionModel

from radar.output import triage as _out_triage
from radar.pipeline import PipelineRequest, run as _run_pipeline
from radar.schema.opportunities import RadarDraft
from radar.schema.papers import CollectedWork
from radar.source.openalex import DictTransport, build_query_plan
from radar.config.interests import default_profile
from radar.storage import triage as _triage_store


def _work(wid: str, **kwargs) -> CollectedWork:
    params = dict(title=f"Title {wid}", abstract="abstract text about models",
                  publication_year=2025, doi="", primary_url=wid,
                  locations=[], cited_by_count=1, matched_queries=["q"],
                  query_kinds=["semantic"], score=0.1)
    params.update(kwargs)
    return CollectedWork(openalex_id=wid, **params)


def _result(wid: str, status: str, rel=None, cross=None) -> SimpleNamespace:
    return SimpleNamespace(work_id=wid, status=status,
                           ai_ml_relevance=rel, cross_domain_potential=cross)


def _batch(results, model_id="test-clef", rubric="r1") -> SimpleNamespace:
    return SimpleNamespace(model_id=model_id, rubric_version=rubric,
                           results=list(results))


def _pages(n: int) -> dict:
    plan = build_query_plan(default_profile(), max_queries=6)
    pages: dict = {}
    i = 0
    for q in plan.queries:
        rows = []
        for _ in range(3):
            if i >= n:
                break
            rows.append({
                "id": f"https://openalex.org/W{i}", "title": f"Work {i}",
                "abstract_inverted_index": {"x": [0]},
                "doi": "", "publication_year": 2026, "cited_by_count": i})
            i += 1
        pages[q.terms] = {"results": rows}
    return pages


def _write_snapshot(tmp: str, works: list) -> str:
    """Write a minimal valid v1 snapshot with pathlib (handles closed)."""
    import json as _json
    from pathlib import Path

    snap = str(Path(tmp) / "snapshot.json")
    n = len(works)
    with open(snap, "w", encoding="utf-8") as fh:
        _json.dump({
            "schema_version": 1,
            "collected_at_utc": "2026-10-04T00:00:00+00:00",
            "discovery": {"source": "OpenAlex"},
            "coverage": {"collected": n, "with_abstracts": n,
                         "missing_abstracts": 0, "llm_selected": 0,
                         "llm_analyzed": 0},
            "works": [w.model_dump() for w in works],
        }, fh)
    return snap


def _read_bytes(path: str) -> bytes:
    from pathlib import Path

    return Path(path).read_bytes()


def _fixed_model() -> FunctionModel:
    fixed = RadarDraft(opportunities=[], ignore=[], next_move="done")

    def _impl(messages, info):
        return ModelResponse(parts=[ToolCallPart(
            tool_name="final_result", args=fixed.model_dump())])

    return FunctionModel(_impl)


def _failed_routing_model() -> FunctionModel:
    def unavailable(messages, info):
        raise RuntimeError("simulated provider outage")
    return FunctionModel(unavailable)


class TestClefConfigRequired(unittest.TestCase):
    def test_both_endpoints_unconfigured_is_exit4_with_zero_model_calls(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            result = _run_pipeline(PipelineRequest(
                mode="analyze", max_candidates=2,
                source_override=DictTransport(_pages(6))))
        self.assertEqual(result.exit_code, 4)
        self.assertTrue(any("FREETOKEN_BASE_URL" in n for n in result.stderr_notes))

    def test_missing_endpoint_checked_even_with_model_override(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            result = _run_pipeline(PipelineRequest(
                mode="analyze", max_candidates=2,
                source_override=DictTransport(_pages(6)),
                model_override=_fixed_model()))
        self.assertEqual(result.exit_code, 4)

    def test_bad_clef_path_redacted_in_pipeline_stderr(self):
        secret_path = "/v1/secret-token-abc-xyz"
        result = _run_pipeline(PipelineRequest(
            mode="analyze", max_candidates=1,
            clef_base_url=f"http://127.0.0.1:11434{secret_path}",
            source_override=DictTransport(_pages(1)),
            model_override=_fixed_model()))
        self.assertEqual(result.exit_code, 4)
        joined = " ".join(result.stderr_notes)
        self.assertNotIn("secret-token-abc-xyz", joined)

    def test_no_triage_mode_flag(self):
        from radar.cli import main

        with self.assertRaises(SystemExit):
            main(["--triage", "off"])

    def test_bad_clef_timeouts_are_usage_errors(self):
        with mock.patch.dict(os.environ, {"CLEF_BASE_URL": "http://127.0.0.1:9"}):
            bad_request = _run_pipeline(PipelineRequest(
                mode="analyze", max_candidates=1, clef_timeout_s=0,
                source_override=DictTransport(_pages(1)),
                model_override=_fixed_model(),
                triage_scorer=lambda pool, profile: _batch([])))
            self.assertEqual(bad_request.exit_code, 4)
            bad_overall = _run_pipeline(PipelineRequest(
                mode="analyze", max_candidates=1, triage_timeout_s=301,
                source_override=DictTransport(_pages(1)),
                model_override=_fixed_model(),
                triage_scorer=lambda pool, profile: _batch([])))
            self.assertEqual(bad_overall.exit_code, 4)


class TestCollectOnlyIgnoresClef(unittest.TestCase):
    def test_collect_only_needs_no_clef_config(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("CLEF_BASE_URL", None)
            result = _run_pipeline(PipelineRequest(
                mode="collect", max_candidates=2,
                clef_timeout_s=-1, triage_timeout_s=-1,
                source_override=DictTransport(_pages(6))))
        self.assertEqual(result.exit_code, 0)
        self.assertEqual(len(json.loads(result.stdout)), 2)


class TestSidecar(unittest.TestCase):
    def test_write_sidecar_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "triage-out")
            batch = _batch([_result("https://openalex.org/W1", "scored", 0.9, 0.2)])
            summary = _triage_store.write_sidecar(out, batch)
            self.assertEqual(summary["results"], 1)
            with open(os.path.join(out, "triage.json")) as fh:
                payload = json.load(fh)
            self.assertEqual(payload["schema"], "triage-sidecar/v1")
            self.assertEqual(payload["model_id"], "test-clef")
            self.assertTrue(payload["generated_at_utc"])

    def test_sidecar_failure_preserves_previous(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "triage-out")
            batch = _batch([_result("https://openalex.org/W1", "scored", 0.9, 0.2)])
            _triage_store.write_sidecar(out, batch)
            with open(os.path.join(out, "triage.json"), "rb") as fh:
                before = fh.read()
            blocker = os.path.join(out, "triage.json")
            os.remove(blocker)
            os.mkdir(blocker)  # a directory at the file path breaks staging
            with self.assertRaises(_triage_store.TriageSidecarError):
                _triage_store.write_sidecar(out, batch)


class TestTriageCoverageFormat(unittest.TestCase):
    def test_buckets_and_provenance(self):
        batch = _batch([
            _result("w1", "scored", 0.9, 0.1),
            _result("w2", "missing_abstract"),
            _result("w3", "oversized"),
            _result("w4", "deadline"),
            _result("w5", "failed"),
        ], model_id="clef-flash", rubric="rubric-7")
        summary = _out_triage.summarize_batch(batch)
        self.assertEqual(summary, {
            "considered": 5, "scored": 1, "unknown": 3, "failed": 1,
            "model_id": "clef-flash", "rubric_version": "rubric-7",
            "backend": "clef", "probability_kind": "native_noul", "fallback_reason": None})
        line = _out_triage.triage_coverage_line(
            pool_total=9, summary=summary, selected=2, opportunities=1)
        for token in ("pool=9", "considered=5", "scored=1", "unknown=3",
                      "failed=1", "selected=2", "analyzed=2", "opportunities=1",
                      "model=clef-flash", "rubric=rubric-7"):
            self.assertIn(token, line)


class TestMandatoryRoutingIntegration(unittest.TestCase):
    def test_scorer_override_drives_shortlist_and_coverage(self):
        from radar.processing.triage import select_candidates

        works = [_work(f"https://openalex.org/W{i}") for i in range(6)]

        def _scorer(pool, profile):
            self.assertEqual(len(pool), 6)
            self.assertTrue(profile.keywords)
            return _batch([
                _result(f"https://openalex.org/W{i}", "scored",
                        0.9 - 0.1 * i, 0.1) for i in range(6)])

        with tempfile.TemporaryDirectory() as tmp:
            snap = _write_snapshot(tmp, works)
            result = _run_pipeline(PipelineRequest(
                mode="analyze", max_candidates=3, from_snapshot=snap,
                model_override=_fixed_model(), triage_scorer=_scorer))
            self.assertEqual(result.exit_code, 0)
            expected = select_candidates(
                works,
                _batch([_result(f"https://openalex.org/W{i}", "scored",
                                0.9 - 0.1 * i, 0.1) for i in range(6)]), 3)
            expected_ids = {w.openalex_id for w in expected}
            self.assertTrue(expected_ids)
            line = " ".join(result.stderr_notes)
            self.assertIn("considered=6", line)
            self.assertIn("selected=3", line)

    def test_failed_batch_statuses_stop_before_synthesis(self):
        seen = []

        def _all_failed(pool, profile):
            return _batch([
                _result(w.openalex_id, "failed") for w in pool])

        def _model(messages, info):
            seen.append(1)
            raise AssertionError("synthesis must not run on failed batch")

        with tempfile.TemporaryDirectory() as tmp:
            snap = _write_snapshot(tmp, [_work("https://openalex.org/W0")])
            before = _read_bytes(snap)
            result = _run_pipeline(PipelineRequest(
                mode="analyze", max_candidates=1, from_snapshot=snap,
                model_override=FunctionModel(_model),
                triage_model_override=_failed_routing_model(),
                triage_scorer=_all_failed))
            self.assertEqual(result.exit_code, 3)
            self.assertEqual(seen, [])
            self.assertEqual(_read_bytes(snap), before)

    def test_service_failure_stops_before_synthesis(self):
        seen = []

        def _failing_scorer(pool, profile):
            raise RuntimeError("connection refused")

        def _model(messages, info):
            seen.append(1)
            raise AssertionError("synthesis must not run after triage failure")

        with tempfile.TemporaryDirectory() as tmp:
            snap = _write_snapshot(tmp, [_work("https://openalex.org/W0")])
            before = _read_bytes(snap)
            result = _run_pipeline(PipelineRequest(
                mode="analyze", max_candidates=1, from_snapshot=snap,
                model_override=FunctionModel(_model),
                triage_model_override=_failed_routing_model(),
                triage_scorer=_failing_scorer))
            self.assertEqual(result.exit_code, 3)
            self.assertEqual(seen, [])
            self.assertEqual(_read_bytes(snap), before)


if __name__ == "__main__":
    unittest.main()
