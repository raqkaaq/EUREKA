"""CLEF/SystemOne screening adapter + triage schema tests (HTTPX MockTransport;
no network, no model, no filesystem)."""

from __future__ import annotations

import json
import os
import unittest
from unittest import mock

import httpx

from radar.config.interests import RadarProfile
from radar.schema.papers import CollectedWork


def _work(wid: str, abstract: str = "abstract text about learning") -> CollectedWork:
    return CollectedWork(
        openalex_id=wid, title=f"Title {wid}", abstract=abstract,
        publication_year=2025, doi="", primary_url=wid, locations=[],
        cited_by_count=1, matched_queries=["q"], query_kinds=["semantic"],
        score=0.1,
    )


def _profile() -> RadarProfile:
    return RadarProfile(
        keywords=["machine learning", "deep learning"],
        domains=["behavioral science", "mechanism design"],
        lookback_days=90,
    )


def _answers(relevance: object, cross: object, model: str = "clef-flash") -> dict:
    return {
        "model": model,
        "answers": {
            "ai_ml_relevance": {"type": "noul", "noul": relevance},
            "cross_domain_potential": {"type": "noul", "noul": cross},
        },
        "usage": {"input_tokens": 100, "output_tokens": 2},
    }


class TestTriageSchema(unittest.TestCase):
    def test_valid_result_and_batch(self):
        from radar.schema.triage import TriageBatch, TriageResult

        batch = TriageBatch(
            model_id="clef-flash", rubric_version="clef-triage-v1",
            results=[TriageResult(
                work_id="https://openalex.org/W1", status="scored",
                ai_ml_relevance=0.9, cross_domain_potential=0.2)],
        )
        self.assertEqual(batch.results[0].status, "scored")

    def test_scored_requires_probabilities(self):
        from pydantic import ValidationError

        from radar.schema.triage import TriageResult

        with self.assertRaises(ValidationError):
            TriageResult(work_id="W1", status="scored",
                         ai_ml_relevance=0.5, cross_domain_potential=None)

    def test_unknown_statuses_require_no_probabilities(self):
        from pydantic import ValidationError

        from radar.schema.triage import TriageResult

        with self.assertRaises(ValidationError):
            TriageResult(work_id="W1", status="missing_abstract",
                         ai_ml_relevance=0.5, cross_domain_potential=None)

    def test_rejects_bool_string_nan_out_of_range(self):
        from pydantic import ValidationError

        from radar.schema.triage import TriageResult

        for bad in (True, "0.9", float("nan"), 1.5, -0.1):
            with self.assertRaises(ValidationError, msg=f"{bad!r}"):
                TriageResult(work_id="W1", status="scored",
                             ai_ml_relevance=bad, cross_domain_potential=0.5)

    def test_rejects_bad_status(self):
        from pydantic import ValidationError

        from radar.schema.triage import TriageResult

        with self.assertRaises(ValidationError):
            TriageResult(work_id="W1", status="maybe")


class TestClefConfig(unittest.TestCase):
    def _env(self, base=None, model=None):
        env = {}
        if base is not None:
            env["CLEF_BASE_URL"] = base
        if model is not None:
            env["CLEF_MODEL"] = model
        return mock.patch.dict(os.environ, env, clear=False)

    def test_resolve_from_env_and_defaults(self):
        from radar.provider.clef import ClefConfig

        with self._env("http://192.168.0.166:1919/v1"):
            cfg = ClefConfig.resolve()
        self.assertEqual(cfg.model, "clef-flash")
        self.assertEqual(cfg.request_timeout_s, 10)
        self.assertEqual(cfg.overall_timeout_s, 60)
        self.assertEqual(cfg.concurrency, 4)

    def test_missing_base_url_rejected(self):
        from radar.provider.clef import ClefConfig, ClefError

        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ClefError):
                ClefConfig.resolve()

    def test_public_url_rejected(self):
        from radar.provider.clef import ClefConfig, ClefError

        with self._env("https://api.example.com/v1"):
            with self.assertRaises(ClefError):
                ClefConfig.resolve()

    def test_bad_bounds_rejected(self):
        from radar.provider.clef import ClefConfig, ClefError

        with self._env("http://127.0.0.1:11434/v1"):
            for kwargs in ({"request_timeout_s": 0}, {"request_timeout_s": 31},
                           {"overall_timeout_s": 301}, {"concurrency": 0},
                           {"concurrency": 9}):
                with self.assertRaises(ClefError, msg=str(kwargs)):
                    ClefConfig.resolve(**kwargs)

    def test_missing_config_names_base_not_endpoint(self):
        from radar.provider.clef import ClefConfig, ClefError

        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ClefError) as ctx:
                ClefConfig.resolve()
        text = str(ctx.exception)
        self.assertIn("CLEF_BASE_URL", text)
        self.assertNotIn("/v1/systemone", text)

    def test_base_url_normalization(self):
        from radar.provider.clef import ClefConfig

        cases = {
            "http://127.0.0.1:11434": "http://127.0.0.1:11434/v1",
            "http://127.0.0.1:11434/": "http://127.0.0.1:11434/v1",
            "http://127.0.0.1:11434/v1": "http://127.0.0.1:11434/v1",
            "http://127.0.0.1:11434/v1/systemone":
                "http://127.0.0.1:11434/v1",
            "http://127.0.0.1:11434/prefix/v1":
                "http://127.0.0.1:11434/prefix/v1",
            "http://127.0.0.1:11434/prefix/v1/systemone":
                "http://127.0.0.1:11434/prefix/v1",
        }
        for raw, expected in cases.items():
            with self._env(raw):
                cfg = ClefConfig.resolve()
            self.assertEqual(cfg.base_url, expected, msg=raw)

    def test_unsupported_path_echoes_nothing(self):
        from radar.provider.clef import ClefConfig, ClefError

        secret_path = "/v1/secret-token-abc-xyz"
        with self._env(f"http://127.0.0.1:11434{secret_path}"):
            with self.assertRaises(ClefError) as ctx:
                ClefConfig.resolve()
        text = str(ctx.exception)
        self.assertNotIn("secret-token-abc-xyz", text)
        self.assertNotIn(secret_path, text)

    def test_bad_base_shapes_rejected_without_leak(self):
        from radar.provider.clef import ClefConfig, ClefError

        bad = [
            "http://user:s3cret@127.0.0.1:11434/v1",
            "http://127.0.0.1:11434/v1?api_key=S3CRET",
            "http://127.0.0.1:11434/v1#frag",
            "http://127.0.0.1:11434/api/v2",
            "http://127.0.0.1:11434/systemone",
        ]
        for raw in bad:
            with self._env(raw):
                with self.assertRaises(ClefError, msg=raw) as ctx:
                    ClefConfig.resolve()
            text = str(ctx.exception)
            self.assertNotIn("s3cret", text.lower())
            self.assertNotIn("S3CRET", text)


class TestScreenWorks(unittest.TestCase):
    def _screen(self, works, handler, **cfg_kwargs):
        from radar.provider.clef import ClefConfig, screen_works

        with mock.patch.dict(
            os.environ, {"CLEF_BASE_URL": "http://127.0.0.1:11434/v1"},
            clear=False,
        ):
            config = ClefConfig.resolve(**cfg_kwargs)
        calls: list = []

        def wrapped(request):
            calls.append(request)
            return handler(request)

        batch = screen_works(
            works, _profile(), config=config,
            transport=httpx.MockTransport(wrapped),
        )
        return batch, calls

    def test_success_payload_shape_and_result(self):
        def handler(request):
            self.assertEqual(request.url.path, "/v1/systemone")
            self.assertEqual(request.method, "POST")
            body = json.loads(request.content.decode())
            lowered = json.dumps(body).lower()
            self.assertNotIn("chat/completions", lowered)
            self.assertNotIn("tools", lowered)
            self.assertEqual(set(body["questions"]),
                             {"ai_ml_relevance", "cross_domain_potential"})
            self.assertEqual(body["model"], "clef-flash")
            return httpx.Response(200, json=_answers(0.8, 0.3))

        batch, calls = self._screen([_work("https://openalex.org/W1")], handler)
        self.assertEqual(len(calls), 1)
        (res,) = batch.results
        self.assertEqual(res.status, "scored")
        self.assertEqual(res.ai_ml_relevance, 0.8)
        self.assertEqual(batch.model_id, "clef-flash")
        self.assertEqual(batch.rubric_version, "clef-triage-v1")

    def test_missing_abstract_makes_no_call(self):
        def handler(request):
            return httpx.Response(200, json=_answers(0.5, 0.5))

        works = [_work("W1", abstract=""), _work("W2", abstract="  "),
                 _work("W3")]
        batch, calls = self._screen(works, handler)
        self.assertEqual(len(calls), 1)
        by_id = {r.work_id: r.status for r in batch.results}
        self.assertEqual(by_id, {"W1": "missing_abstract",
                                 "W2": "missing_abstract", "W3": "scored"})

    def test_oversized_payload_makes_no_call(self):
        def handler(request):
            return httpx.Response(200, json=_answers(0.5, 0.5))

        # 4-byte chars: within the 20k-char model bound but over 64 KiB UTF-8.
        works = [_work("W1", abstract="\U00010348" * 20000)]
        batch, calls = self._screen(works, handler)
        self.assertEqual(calls, [])
        self.assertEqual(batch.results[0].status, "oversized")

    def test_malformed_probabilities_fail(self):
        for bad in (True, "0.9", 1.5):
            def handler(request, _bad=bad):
                return httpx.Response(200, json=_answers(_bad, 0.5))
            batch, _ = self._screen([_work("W1")], handler)
            self.assertEqual(batch.results[0].status, "failed")

    def test_nan_probability_fails(self):
        def handler(request):
            return httpx.Response(
                200,
                content=b'{"model":"clef-flash","answers":'
                b'{"ai_ml_relevance":{"type":"noul","noul":NaN},'
                b'"cross_domain_potential":{"type":"noul","noul":0.5}}}',
                headers={"Content-Type": "application/json"},
            )

        batch, _ = self._screen([_work("W1")], handler)
        self.assertEqual(batch.results[0].status, "failed")

    def test_missing_question_and_model_mismatch_fail(self):
        def handler_missing(request):
            payload = _answers(0.5, 0.5)
            del payload["answers"]["cross_domain_potential"]
            return httpx.Response(200, json=payload)

        batch, _ = self._screen([_work("W1")], handler_missing)
        self.assertEqual(batch.results[0].status, "failed")

        def handler_model(request):
            return httpx.Response(200, json=_answers(0.5, 0.5, model="other"))

        batch, _ = self._screen([_work("W1")], handler_model)
        self.assertEqual(batch.results[0].status, "failed")

    def test_service_error_fail_fast_no_retry(self):
        def handler(request):
            return httpx.Response(500, text="internal boom")

        works = [_work(f"W{i}") for i in range(3)]
        batch, calls = self._screen(works, handler)
        self.assertEqual(len(calls), 1)
        self.assertEqual([r.status for r in batch.results],
                         ["failed", "failed", "failed"])

    def test_concurrency_bound_respected(self):
        import threading
        import time

        live = 0
        peak = 0
        lock = threading.Lock()

        def handler(request):
            nonlocal live, peak
            with lock:
                live += 1
                peak = max(peak, live)
            try:
                time.sleep(0.02)
                return httpx.Response(200, json=_answers(0.5, 0.5))
            finally:
                with lock:
                    live -= 1

        works = [_work(f"W{i}") for i in range(8)]
        batch, _ = self._screen(works, handler)
        self.assertLessEqual(peak, 4)
        self.assertEqual([r.status for r in batch.results], ["scored"] * 8)

    def test_overall_deadline_marks_unattempted(self):
        import time

        def handler(request):
            time.sleep(0.3)
            return httpx.Response(200, json=_answers(0.5, 0.5))

        works = [_work(f"W{i}") for i in range(3)]
        batch, _ = self._screen(works, handler, overall_timeout_s=0.05)
        statuses = [r.status for r in batch.results]
        self.assertEqual(len(statuses), 3)
        self.assertIn("deadline", statuses)
        self.assertNotIn("scored", statuses)

    def test_wire_url_exact_no_doubled_path(self):
        for raw, expected_path in (
            ("http://127.0.0.1:11434", "/v1/systemone"),
            ("http://127.0.0.1:11434/v1", "/v1/systemone"),
            ("http://127.0.0.1:11434/v1/systemone", "/v1/systemone"),
            ("http://127.0.0.1:11434/prefix/v1", "/prefix/v1/systemone"),
        ):
            paths: list = []

            async def handler(request, _paths=paths):
                _paths.append(request.url.path)
                return httpx.Response(200, json=_answers(0.5, 0.5))

            from radar.provider.clef import ClefConfig, screen_works

            with mock.patch.dict(os.environ, {"CLEF_BASE_URL": raw},
                                 clear=False):
                config = ClefConfig.resolve()
            screen_works([_work("W1")], _profile(), config=config,
                         transport=httpx.MockTransport(handler))
            self.assertEqual(paths, [expected_path], msg=raw)

    def test_overall_deadline_bounds_wallclock_and_closes(self):
        import asyncio
        import time

        closed = 0
        calls = 0

        class RecordingTransport(httpx.MockTransport):
            async def aclose(self):
                nonlocal closed
                closed += 1
                await super().aclose()

        async def handler(request):
            nonlocal calls
            calls += 1
            await asyncio.sleep(1.0)
            return httpx.Response(200, json=_answers(0.5, 0.5))

        from radar.provider.clef import ClefConfig, screen_works

        with mock.patch.dict(
            os.environ, {"CLEF_BASE_URL": "http://127.0.0.1:11434/v1"},
            clear=False,
        ):
            config = ClefConfig.resolve(overall_timeout_s=0.1,
                                        request_timeout_s=10)
        works = [_work(f"W{i}") for i in range(3)]
        transport = RecordingTransport(handler)
        start = time.monotonic()
        batch = screen_works(works, _profile(), config=config,
                             transport=transport)
        elapsed = time.monotonic() - start
        # Async overall deadline cancels in-flight calls: all deadline,
        # wall clock near the 0.1s budget (far below the 1.0s handler
        # delay a blocking join would wait out), client closed, and no
        # further calls issued after cancellation. (No guarantee holds
        # for custom transports whose handlers block the loop itself.)
        self.assertEqual([r.status for r in batch.results],
                         ["deadline"] * 3)
        self.assertLess(elapsed, 0.5)
        self.assertEqual(closed, 1)
        self.assertEqual(calls, 1)

    def test_client_closed_on_service_failure(self):
        closed = 0

        class RecordingTransport(httpx.MockTransport):
            async def aclose(self):
                nonlocal closed
                closed += 1
                await super().aclose()

        async def handler(request):
            return httpx.Response(500, text="boom")

        from radar.provider.clef import ClefConfig, screen_works

        with mock.patch.dict(
            os.environ, {"CLEF_BASE_URL": "http://127.0.0.1:11434/v1"},
            clear=False,
        ):
            config = ClefConfig.resolve()
        batch = screen_works([_work("W1")], _profile(), config=config,
                             transport=RecordingTransport(handler))
        self.assertEqual(batch.results[0].status, "failed")
        self.assertEqual(closed, 1)


if __name__ == "__main__":
    unittest.main()
