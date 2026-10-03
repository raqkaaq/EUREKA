"""Same native SystemOne request contract at CLEF and Qwen endpoints."""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import httpx
from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models.function import FunctionModel

from radar.pipeline import PipelineRequest, run
from radar.schema.opportunities import RadarDraft
from radar.schema.papers import CollectedWork
from radar.schema.triage import TriageBatch, TriageResult
from radar.storage.snapshots import refresh_pool


def works(n=3):
    return [CollectedWork(openalex_id=f"https://openalex.org/W{i}",
                          title=f"Paper {i}", abstract=f"Full abstract {i}")
            for i in range(n)]


def synth_model(calls):
    def respond(messages, info):
        calls.append(1)
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name,
            RadarDraft(opportunities=[], ignore=[], next_move="done").model_dump())])
    return FunctionModel(respond)


def scored(requests):
    def respond(request):
        body = json.loads(request.content)
        requests.append((str(request.url), body))
        return httpx.Response(200, json={"model": body["model"], "answers": {
            "ai_ml_relevance": {"type": "noul", "noul": 0.8},
            "cross_domain_potential": {"type": "noul", "noul": 0.4}}})
    return httpx.MockTransport(respond)


class TestQwenSystemOne(unittest.TestCase):
    def pipeline(self, pool, *, clef_transport=None, qwen_transport=None, **kwargs):
        """Exercise the real pipeline; only external HTTP/model are doubled."""
        synthesis = []
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ, {}, clear=True):
            snapshot = refresh_pool(pool, tmp)["snapshot"]
            before = Path(snapshot).read_bytes()
            request = dict(
                mode="analyze", from_snapshot=snapshot, max_candidates=2,
                qwen_systemone_base_url="http://127.0.0.1:1919/v1",
                qwen_systemone_model="test-qwen", qwen_systemone_transport=qwen_transport,
                model_override=synth_model(synthesis))
            if clef_transport is not None:
                request.update(clef_base_url="http://127.0.0.1:11434/v1",
                               clef_model="test-clef", clef_transport=clef_transport)
            request.update(kwargs)
            result = run(PipelineRequest(**request))
            self.assertEqual(Path(snapshot).read_bytes(), before)
        return result, synthesis

    def test_missing_clef_native_fallback_full_pool_then_synthesis(self):
        requests, synthesis = [], []
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ, {}, clear=True):
            snapshot = refresh_pool(works(30), tmp)["snapshot"]
            before = Path(snapshot).read_bytes()
            result = run(PipelineRequest(
                mode="analyze", from_snapshot=snapshot, max_candidates=2,
                qwen_systemone_base_url="http://127.0.0.1:1919/v1",
                qwen_systemone_model="test-qwen",
                qwen_systemone_transport=scored(requests),
                model_override=synth_model(synthesis), triage_output=tmp + "/triage"))
            self.assertEqual(result.exit_code, 0, result.stderr_notes)
            self.assertEqual(len(requests), 30)
            self.assertEqual(synthesis, [1])
            self.assertEqual(Path(snapshot).read_bytes(), before)
            self.assertTrue(all(url.endswith("/v1/systemone") for url, _ in requests))
            self.assertTrue(all(set(body) == {"model", "state", "questions"}
                                for _, body in requests))
            notes = " ".join(result.stderr_notes)
            self.assertIn("backend=qwen", notes)
            self.assertIn("fallback=missing_endpoint", notes)
            self.assertIn("probabilities=native_noul", notes)
            sidecar = json.loads(Path(tmp + "/triage/triage.json").read_text())
            self.assertEqual(sidecar["backend"], "qwen")
            self.assertEqual(sidecar["fallback_reason"], "missing_endpoint")

    def test_payloads_are_identical_except_endpoint_model_selection(self):
        native, fallback = [], []
        def down(request):
            native.append((str(request.url), json.loads(request.content)))
            return httpx.Response(503)
        result, synthesis = self.pipeline(
            works(), clef_transport=httpx.MockTransport(down),
            qwen_transport=scored(fallback), keywords=("psychometrics",))
        self.assertEqual(result.exit_code, 0, result.stderr_notes)
        self.assertEqual(synthesis, [1])
        self.assertEqual(len(native), 1)
        self.assertEqual(len(fallback), 3)
        clef = dict(native[0][1])
        qwen = next(body for _, body in fallback if body["state"] == clef["state"])
        clef.pop("model")
        qwen = dict(qwen)
        qwen.pop("model")
        self.assertEqual(qwen, clef)
        self.assertIn("psychometrics", qwen["questions"]["ai_ml_relevance"]["instructions"])
        self.assertIn("fallback=screening_failed", " ".join(result.stderr_notes))

    def test_full_106_pool_missing_abstracts_are_unknown(self):
        pool = works(106)
        for index in (0, 27, 54, 81):
            pool[index].abstract = ""
        requests = []
        result, synthesis = self.pipeline(pool, qwen_transport=scored(requests), max_candidates=8)
        self.assertEqual(result.exit_code, 0, result.stderr_notes)
        self.assertEqual(len(requests), 102)
        self.assertEqual(synthesis, [1])
        notes = " ".join(result.stderr_notes)
        for token in ("pool=106", "considered=106", "scored=102", "unknown=4", "selected=8"):
            self.assertIn(token, notes)

    def test_successful_clef_never_calls_fallback(self):
        native, fallback = [], []
        result, synthesis = self.pipeline(works(), clef_transport=scored(native),
                                          qwen_transport=scored(fallback))
        self.assertEqual(result.exit_code, 0, result.stderr_notes)
        self.assertEqual(len(native), 3)
        self.assertEqual(fallback, [])
        self.assertEqual(synthesis, [1])
        self.assertIn("backend=clef", " ".join(result.stderr_notes))

    def test_malformed_clef_reroutes_entire_pool(self):
        requests = []
        result, synthesis = self.pipeline(works(), qwen_transport=scored(requests),
            clef_transport=httpx.MockTransport(lambda r: httpx.Response(200, json={
                "model": "wrong", "answers": {}})))
        self.assertEqual(result.exit_code, 0, result.stderr_notes)
        self.assertEqual(len(requests), 3)
        self.assertEqual(synthesis, [1])

    def test_clef_deadline_still_uses_native_qwen_fallback(self):
        async def stall(request):
            await asyncio.sleep(1)
            return httpx.Response(200)
        requests = []
        result, synthesis = self.pipeline(works(), qwen_transport=scored(requests),
            clef_transport=httpx.MockTransport(stall), triage_timeout_s=0.03)
        self.assertEqual(result.exit_code, 0, result.stderr_notes)
        self.assertEqual(len(requests), 3)
        self.assertEqual(synthesis, [1])
        self.assertIn("fallback=deadline", " ".join(result.stderr_notes))

    def test_chat_only_404_aborts_after_one_native_call_without_synthesis(self):
        calls = []
        def missing(request):
            calls.append(request.url.path)
            return httpx.Response(404, text="sensitive upstream body")
        result, synthesis = self.pipeline(works(30),
                                         qwen_transport=httpx.MockTransport(missing))
        self.assertEqual(result.exit_code, 3)
        self.assertEqual(calls, ["/v1/systemone"])
        self.assertEqual(synthesis, [])
        notes = " ".join(result.stderr_notes)
        self.assertIn("Chat Completions-only", notes)
        self.assertNotIn("sensitive", notes)

    def test_malformed_qwen_probabilities_block_synthesis(self):
        for value in (True, "0.8", None, float("nan"), float("inf"), -0.1, 1.1):
            with self.subTest(value=repr(value)):
                def invalid(request):
                    body = json.loads(request.content)
                    return httpx.Response(200, content=json.dumps({
                        "model": body["model"], "answers": {
                            "ai_ml_relevance": {"type": "noul", "noul": value},
                            "cross_domain_potential": {"type": "noul", "noul": 0.3}}}).encode())
                result, synthesis = self.pipeline(works(),
                    qwen_transport=httpx.MockTransport(invalid))
                self.assertEqual(result.exit_code, 3)
                self.assertEqual(synthesis, [])

    def test_qwen_deadline_cancels_and_blocks_synthesis(self):
        cancelled = []
        async def stall(request):
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.append(1)
        result, synthesis = self.pipeline(works(), qwen_transport=httpx.MockTransport(stall),
                                          triage_timeout_s=0.02)
        self.assertEqual(result.exit_code, 3)
        self.assertEqual(synthesis, [])
        self.assertEqual(cancelled, [1])

    def test_invalid_clef_or_qwen_endpoint_never_calls_other_services(self):
        for endpoint in ("http://8.8.8.8/v1", "http://user:secret@127.0.0.1/v1",
                         "http://127.0.0.1/v1?key=secret", "http://127.0.0.1/bad-secret"):
            for field in ("clef_base_url", "qwen_systemone_base_url"):
                with self.subTest(field=field, endpoint=endpoint):
                    requests = []
                    result, synthesis = self.pipeline(works(), qwen_transport=scored(requests),
                                                      **{field: endpoint})
                    self.assertEqual(result.exit_code, 4)
                    self.assertEqual(requests, [])
                    self.assertEqual(synthesis, [])
                    self.assertNotIn("secret", " ".join(result.stderr_notes))

    def test_oversized_and_missing_papers_are_not_sent_to_qwen(self):
        pool = works(4)
        pool[0].abstract = ""
        pool[1].abstract = "😀" * 20000
        requests = []
        result, synthesis = self.pipeline(pool, qwen_transport=scored(requests))
        self.assertEqual(result.exit_code, 0, result.stderr_notes)
        self.assertEqual(len(requests), 2)
        self.assertEqual(synthesis, [1])
        self.assertIn("unknown=2", " ".join(result.stderr_notes))

    def test_collect_only_and_empty_pool_call_no_models(self):
        for pool, options in ((works(), {"mode": "collect"}), ([], {})):
            requests = []
            result, synthesis = self.pipeline(pool, qwen_transport=scored(requests), **options)
            self.assertEqual(result.exit_code, 0, result.stderr_notes)
            self.assertEqual(requests, [])
            self.assertEqual(synthesis, [])

    def test_configuration_precedence_and_no_clef_setting_leak(self):
        from radar.provider.qwen_systemone import resolve_config

        with mock.patch.dict(os.environ, {
            "FREETOKEN_BASE_URL": "http://127.0.0.1:1919/v1", "FREETOKEN_MODEL": "qwen-default",
            "CLEF_BASE_URL": "http://127.0.0.1:11434/v1", "CLEF_MODEL": "clef-default"}, clear=True):
            config = resolve_config()
            self.assertEqual(config.base_url, "http://127.0.0.1:1919/v1")
            self.assertEqual(config.model, "qwen-default")
            with mock.patch.dict(os.environ, {"QWEN_SYSTEMONE_BASE_URL": "http://127.0.0.1:1920/v1",
                                              "QWEN_SYSTEMONE_MODEL": "qwen-override"}):
                config = resolve_config(freetoken_base_url="http://127.0.0.1:1930/v1", freetoken_model="other")
                self.assertEqual(config.base_url, "http://127.0.0.1:1920/v1")
                self.assertEqual(config.model, "qwen-override")
                config = resolve_config(base_url="http://127.0.0.1:1940/v1", model="explicit")
                self.assertEqual(config.model, "explicit")
                self.assertEqual(config.base_url, "http://127.0.0.1:1940/v1")

    def test_cli_forwards_native_endpoint_and_model_overrides(self):
        from radar.cli import main
        from radar.pipeline import PipelineResult

        with mock.patch("radar.cli.run", return_value=PipelineResult(0)) as pipeline:
            self.assertEqual(main(["--qwen-systemone-base-url", "http://127.0.0.1:1919/v1",
                                   "--qwen-systemone-model", "native-qwen"]), 0)
        request = pipeline.call_args.args[0]
        self.assertEqual(request.qwen_systemone_base_url, "http://127.0.0.1:1919/v1")
        self.assertEqual(request.qwen_systemone_model, "native-qwen")

    def test_blank_explicit_settings_cannot_inherit_clef_endpoint_or_model(self):
        from radar.provider.qwen_systemone import resolve_config

        with mock.patch.dict(os.environ, {
            "CLEF_BASE_URL": "http://127.0.0.1:11434/v1", "CLEF_MODEL": "clef-only"}, clear=True):
            for options in ({"base_url": "  ", "model": "qwen"},
                            {"base_url": "http://127.0.0.1:1919/v1", "model": "  "}):
                with self.subTest(options=options), self.assertRaises(ValueError):
                    resolve_config(**options)

    def test_partial_clef_scores_are_not_mixed_with_qwen_scores(self):
        native, fallback = [], []
        def partial(request):
            native.append(1)
            if len(native) > 1:
                return httpx.Response(503)
            body = json.loads(request.content)
            return httpx.Response(200, json={"model": body["model"], "answers": {
                "ai_ml_relevance": {"type": "noul", "noul": 0.99},
                "cross_domain_potential": {"type": "noul", "noul": 0.99}}})
        with tempfile.TemporaryDirectory() as tmp:
            result, synthesis = self.pipeline(works(6), clef_transport=httpx.MockTransport(partial),
                qwen_transport=scored(fallback), triage_output=tmp)
            self.assertEqual(result.exit_code, 0, result.stderr_notes)
            sidecar = json.loads(Path(tmp + "/triage.json").read_text())
        self.assertEqual(len(fallback), 6)
        self.assertEqual(synthesis, [1])
        self.assertTrue(all(r["ai_ml_relevance"] == 0.8 for r in sidecar["results"]))

    def test_missing_duplicate_and_foreign_batch_ids_require_complete_fallback(self):
        for ids in ([], ["https://openalex.org/W0"] * 3,
                    ["foreign", "https://openalex.org/W1", "https://openalex.org/W2"]):
            with self.subTest(ids=ids):
                requests = []
                def bad_batch(pool, profile):
                    return TriageBatch(model_id="test-clef", rubric_version="r1", results=[
                        TriageResult(work_id=wid, status="scored", ai_ml_relevance=0.9,
                                     cross_domain_potential=0.2) for wid in ids])
                result, synthesis = self.pipeline(works(), qwen_transport=scored(requests),
                                                  triage_scorer=bad_batch)
                self.assertEqual(result.exit_code, 0, result.stderr_notes)
                self.assertEqual(len(requests), 3)
                self.assertEqual(synthesis, [1])
                self.assertIn("fallback=screening_error", " ".join(result.stderr_notes))

    def test_native_fallback_transport_closes_on_success_failure_and_timeout(self):
        for outcome in ("success", "failure", "deadline"):
            with self.subTest(outcome=outcome):
                closed = []
                class TrackingTransport(httpx.AsyncBaseTransport):
                    async def handle_async_request(self, request):
                        if outcome == "deadline":
                            await asyncio.Event().wait()
                        if outcome == "failure":
                            return httpx.Response(503)
                        body = json.loads(request.content)
                        return httpx.Response(200, json={"model": body["model"], "answers": {
                            "ai_ml_relevance": {"type": "noul", "noul": 0.8},
                            "cross_domain_potential": {"type": "noul", "noul": 0.4}}})
                    async def aclose(self):
                        closed.append(1)
                result, _ = self.pipeline(works(), qwen_transport=TrackingTransport(),
                                          triage_timeout_s=0.02)
                self.assertEqual(result.exit_code, 0 if outcome == "success" else 3)
                self.assertEqual(closed, [1])
