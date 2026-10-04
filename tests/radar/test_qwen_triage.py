"""Same screening inputs/typed answers through Qwen's PydanticAI chat path."""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import httpx

from pydantic_ai.messages import ModelResponse, ToolCallPart, UserPromptPart
from pydantic_ai.models.function import FunctionModel

from radar.config.interests import default_profile
from radar.schema.papers import CollectedWork
from radar.pipeline import PipelineRequest, run
from radar.storage.snapshots import refresh_pool


def works(n=3):
    return [CollectedWork(openalex_id=f"https://openalex.org/W{i}",
                          title=f"Paper {i}", abstract=f"Full abstract {i}")
            for i in range(n)]


def answers_model(calls, mutate=None):
    def respond(messages, info):
        prompt = next(p.content for m in reversed(messages) for p in m.parts
                      if isinstance(p, UserPromptPart))
        papers = json.loads(prompt)["papers"]
        calls.append(papers)
        rows = [{"work_id": p["work_id"], "model": p["input"]["model"],
                 "answers": {"ai_ml_relevance": {"type": "noul", "noul": 0.8},
                             "cross_domain_potential": {"type": "noul", "noul": 0.4}}}
                for p in papers]
        if mutate:
            rows = mutate(rows)
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, {"responses": rows})])
    return FunctionModel(respond)


def synth_model(calls):
    from radar.schema.opportunities import RadarDraft
    def respond(messages, info):
        calls.append(1)
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name,
            RadarDraft(opportunities=[], ignore=[], next_move="done").model_dump())])
    return FunctionModel(respond)


class TestQwenTriage(unittest.TestCase):
    def test_same_screening_input_and_pydantic_answers(self):
        from radar.agent.paper_triage import screen_works

        calls = []
        batch = screen_works(works(), default_profile(),
                             model=answers_model(calls), model_id="test-qwen")
        self.assertTrue(all(r.status == "scored" for r in batch.results))
        self.assertEqual(batch.backend, "qwen")
        self.assertEqual(batch.probability_kind, "prompted_estimate")
        self.assertEqual([r.work_id for r in batch.results], [w.openalex_id for w in works()])
        self.assertEqual(set(calls[0][0]["input"]), {"model", "state", "questions"})
        self.assertEqual(calls[0][0]["input"]["state"]["abstract"], works()[0].abstract)

    def test_106_pool_batched_without_shortlist_loss(self):
        from radar.agent.paper_triage import screen_works
        pool, calls = works(106), []
        for index in (0, 27, 54, 81):
            pool[index].abstract = ""
        batch = screen_works(pool, default_profile(), model=answers_model(calls), model_id="test-qwen")
        self.assertEqual(len(batch.results), 106)
        self.assertEqual(sum(r.status == "scored" for r in batch.results), 102)
        self.assertEqual(sum(r.status == "missing_abstract" for r in batch.results), 4)
        self.assertEqual(len(calls), 5)
        self.assertEqual({p["work_id"] for chunk in calls for p in chunk},
                         {w.openalex_id for w in pool if w.abstract})

    def test_invalid_probability_values_are_rejected_with_two_call_cap(self):
        from radar.agent.paper_triage import screen_works
        for value in (True, "0.8", None, float("nan"), float("inf"), -0.1, 1.1):
            with self.subTest(value=repr(value)):
                calls = []
                def invalid(rows):
                    rows[0]["answers"]["ai_ml_relevance"]["noul"] = value
                    return rows
                batch = screen_works(works(1), default_profile(), model=answers_model(calls, invalid))
                self.assertEqual(batch.results[0].status, "failed")
                self.assertEqual(len(calls), 2)

    def test_missing_duplicate_foreign_ids_and_wrong_model_are_rejected(self):
        from radar.agent.paper_triage import screen_works
        changes = (lambda rows: rows[:-1], lambda rows: [rows[0]] * len(rows),
                   lambda rows: [dict(r, work_id="foreign") for r in rows],
                   lambda rows: [dict(r, model="wrong") for r in rows])
        for change in changes:
            calls = []
            batch = screen_works(works(3), default_profile(), model=answers_model(calls, change))
            self.assertTrue(all(r.status == "failed" for r in batch.results))
            self.assertEqual(len(calls), 2)

    def test_one_validation_retry_can_recover(self):
        from radar.agent.paper_triage import screen_works
        calls = []
        batch = screen_works(works(3), default_profile(), model=answers_model(
            calls, lambda rows: rows[:-1] if len(calls) == 1 else rows))
        self.assertTrue(all(r.status == "scored" for r in batch.results))
        self.assertEqual(len(calls), 2)

    def test_complete_blocks_split_on_byte_budget(self):
        from radar.agent.paper_triage import screen_works
        pool, calls = works(6), []
        for work in pool:
            work.abstract = "x" * 20000
        batch = screen_works(pool, default_profile(), model=answers_model(calls))
        self.assertTrue(all(r.status == "scored" for r in batch.results))
        # Richer shared questions consume bytes too; preserve complete papers.
        self.assertEqual([len(chunk) for chunk in calls], [2, 2, 2])
        self.assertTrue(all(len(p["input"]["state"]["abstract"]) == 20000 for c in calls for p in c))

    def test_missing_and_unicode_oversized_papers_need_no_model(self):
        from radar.agent.paper_triage import screen_works
        pool, calls = works(2), []
        pool[0].abstract = ""
        pool[1].abstract = "😀" * 20000
        batch = screen_works(pool, default_profile(), model=answers_model(calls))
        self.assertEqual([r.status for r in batch.results], ["missing_abstract", "oversized"])
        self.assertEqual(calls, [])

    def test_deadline_cancels_tasks_and_closes_owned_client(self):
        from radar.agent.paper_triage import screen_works
        from radar.provider import strata as freetoken
        active, settled = [], []
        async def stall(messages, info):
            active.append(1)
            try:
                await asyncio.Event().wait()
            finally:
                settled.append(1)
        client = SimpleNamespace(aclose=mock.AsyncMock())
        session = freetoken.FreeTokenSession(FunctionModel(stall), client)
        with mock.patch.object(freetoken.FreeTokenConfig, "resolve", return_value=SimpleNamespace(model="test-qwen")), \
             mock.patch.object(freetoken, "build_session", return_value=session):
            batch = screen_works(works(50), default_profile(), overall_timeout_s=0.02)
        self.assertLessEqual(len(active), 2)
        self.assertEqual(len(settled), len(active))
        self.assertTrue(all(r.status == "deadline" for r in batch.results))
        client.aclose.assert_awaited_once()

    def test_failed_inference_is_redacted_and_owned_client_closed(self):
        from radar.agent.paper_triage import screen_works
        from radar.provider import strata as freetoken
        def fail(messages, info):
            raise RuntimeError("sensitive-body")
        client = SimpleNamespace(aclose=mock.AsyncMock())
        session = freetoken.FreeTokenSession(FunctionModel(fail), client)
        with mock.patch.object(freetoken.FreeTokenConfig, "resolve", return_value=SimpleNamespace(model="test-qwen")), \
             mock.patch.object(freetoken, "build_session", return_value=session):
            batch = screen_works(works(2), default_profile())
        self.assertTrue(all(r.status == "failed" for r in batch.results))
        self.assertNotIn("sensitive", batch.model_dump_json())
        client.aclose.assert_awaited_once()

    def test_real_pydanticai_provider_uses_chat_completions_and_shared_inputs(self):
        import httpx2
        from radar.agent.paper_triage import screen_works
        from radar.provider import strata as freetoken
        requests = []
        def chat(request):
            body = json.loads(request.content)
            requests.append((request.url.path, body))
            prompt = next(m["content"] for m in body["messages"] if m["role"] == "user")
            papers = json.loads(prompt)["papers"]
            responses = [{"work_id": p["work_id"], "model": p["input"]["model"], "answers": {
                "ai_ml_relevance": {"type": "noul", "noul": 0.8},
                "cross_domain_potential": {"type": "noul", "noul": 0.4}}} for p in papers]
            return httpx2.Response(200, json={"id": "chat-test", "object": "chat.completion", "created": 1,
                "model": "test-qwen", "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
                    "role": "assistant", "tool_calls": [{"id": "call1", "type": "function", "function": {
                        "name": body["tools"][0]["function"]["name"],
                        "arguments": json.dumps({"responses": responses})}}]}}],
                "usage": {"prompt_tokens": 20, "completion_tokens": 40, "total_tokens": 60}})
        clients = []
        def create_client(timeout_s):
            client = httpx2.AsyncClient(transport=httpx2.MockTransport(chat), trust_env=False)
            clients.append(client)
            return client
        with mock.patch.dict(os.environ, {
            "STRATA_BASE_URL": "http://127.0.0.1:8080/v1",
            "STRATA_MODEL": "test-qwen",
        }, clear=True), mock.patch.object(freetoken, "_provider_http_client", side_effect=create_client):
            batch = screen_works(works(), default_profile())
        self.assertTrue(all(r.status == "scored" for r in batch.results))
        self.assertEqual([path for path, _ in requests], ["/v1/chat/completions"])
        self.assertTrue(clients[0].is_closed)
        self.assertEqual(requests[0][1]["model"], "test-qwen")


class TestQwenPipeline(unittest.TestCase):
    def pipeline(self, pool, *, routing=None, **kwargs):
        synthesis = []
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ, {}, clear=True):
            snapshot = refresh_pool(pool, tmp)["snapshot"]
            before = Path(snapshot).read_bytes()
            args = dict(mode="analyze", from_snapshot=snapshot, max_candidates=2,
                        model_override=synth_model(synthesis), triage_model_override=routing)
            args.update(kwargs)
            result = run(PipelineRequest(**args))
            self.assertEqual(Path(snapshot).read_bytes(), before)
        return result, synthesis

    def test_absent_clef_full_pool_then_synthesis_with_honest_sidecar(self):
        pool, calls = works(106), []
        for index in (0, 27, 54, 81):
            pool[index].abstract = ""
        with tempfile.TemporaryDirectory() as out:
            result, synthesis = self.pipeline(pool, routing=answers_model(calls), max_candidates=8, triage_output=out)
            sidecar = json.loads(Path(out + "/triage.json").read_text())
        self.assertEqual(result.exit_code, 0, result.stderr_notes)
        self.assertEqual(synthesis, [1, 1, 1, 1])  # three specialists + synthesis
        self.assertEqual(sum(len(chunk) for chunk in calls), 102)
        for token in ("pool=106", "considered=106", "scored=102", "unknown=4", "selected=8", "backend=qwen"):
            self.assertIn(token, " ".join(result.stderr_notes))
        self.assertEqual(sidecar["probability_kind"], "prompted_estimate")
        self.assertEqual(sidecar["rubric_version"], "clef-triage-v2")
        self.assertEqual(len(sidecar["rubric_hash"]), 64)
        self.assertEqual(sidecar["fallback_reason"], "missing_endpoint")

    def test_native_and_chat_inputs_identical_except_served_model(self):
        native, calls = [], []
        def down(request):
            native.append(json.loads(request.content))
            return httpx.Response(503)
        result, synthesis = self.pipeline(works(), routing=answers_model(calls),
            keywords=("psychometrics",), clef_base_url="http://127.0.0.1:9", clef_model="test-clef",
            clef_transport=httpx.MockTransport(down))
        self.assertEqual(result.exit_code, 0, result.stderr_notes)
        self.assertEqual(synthesis, [1, 1, 1, 1])
        chat = dict(calls[0][0]["input"])
        native = dict(native[0])
        chat.pop("model")
        native.pop("model")
        self.assertEqual(chat, native)
        self.assertIn("psychometrics", chat["questions"]["ai_ml_relevance"]["instructions"])

    def test_successful_clef_never_calls_qwen_routing(self):
        calls = []
        def clef(request):
            body = json.loads(request.content)
            return httpx.Response(200, json={"model": body["model"], "answers": {
                "ai_ml_relevance": {"type": "noul", "noul": 0.9},
                "cross_domain_potential": {"type": "noul", "noul": 0.3}}})
        result, synthesis = self.pipeline(works(), routing=answers_model(calls),
            clef_base_url="http://127.0.0.1:9", clef_transport=httpx.MockTransport(clef))
        self.assertEqual(result.exit_code, 0, result.stderr_notes)
        self.assertEqual(calls, [])
        self.assertEqual(synthesis, [1, 1, 1, 1])

    def test_malformed_clef_and_clef_deadline_fall_back(self):
        async def stall(request):
            await asyncio.Event().wait()
        for transport in (httpx.MockTransport(lambda r: httpx.Response(200, json={})), httpx.MockTransport(stall)):
            calls = []
            result, synthesis = self.pipeline(works(), routing=answers_model(calls),
                clef_base_url="http://127.0.0.1:9", clef_transport=transport, triage_timeout_s=0.04)
            self.assertEqual(result.exit_code, 0, result.stderr_notes)
            self.assertEqual(sum(len(c) for c in calls), 3)
            self.assertEqual(synthesis, [1, 1, 1, 1])

    def test_bad_qwen_answers_block_all_synthesis(self):
        calls = []
        result, synthesis = self.pipeline(works(), routing=answers_model(calls, lambda rows: []))
        self.assertEqual(result.exit_code, 3)
        self.assertEqual(synthesis, [])
        self.assertEqual(len(calls), 2)

    def test_collect_only_and_empty_pool_call_no_models(self):
        for pool, options in ((works(), {"mode": "collect"}), ([], {})):
            calls = []
            result, synthesis = self.pipeline(pool, routing=answers_model(calls), **options)
            self.assertEqual(result.exit_code, 0)
            self.assertEqual(calls, [])
            self.assertEqual(synthesis, [])

    def test_invalid_clef_config_stays_an_error_not_a_fallback(self):
        for endpoint in ("http://8.8.8.8/v1", "http://user:secret@127.0.0.1/v1",
                         "http://127.0.0.1/v1?key=secret", "http://127.0.0.1/bad-secret"):
            calls = []
            result, synthesis = self.pipeline(works(), routing=answers_model(calls), clef_base_url=endpoint)
            self.assertEqual(result.exit_code, 4)
            self.assertEqual(calls, [])
            self.assertEqual(synthesis, [])
            self.assertNotIn("secret", " ".join(result.stderr_notes))

    def test_unsafe_chat_endpoint_refused_and_redacted(self):
        for endpoint in ("http://8.8.8.8/v1", "http://user:secret@127.0.0.1/v1"):
            result, synthesis = self.pipeline(works(), base_url=endpoint)
            self.assertEqual(result.exit_code, 4)
            self.assertEqual(synthesis, [])
            self.assertNotIn("secret", " ".join(result.stderr_notes))

    def test_partial_clef_scores_are_not_mixed_into_chat_results(self):
        native, calls = [], []
        def partial(request):
            native.append(1)
            if len(native) > 1:
                return httpx.Response(503)
            body = json.loads(request.content)
            return httpx.Response(200, json={"model": body["model"], "answers": {
                "ai_ml_relevance": {"type": "noul", "noul": 0.99},
                "cross_domain_potential": {"type": "noul", "noul": 0.99}}})
        with tempfile.TemporaryDirectory() as out:
            result, synthesis = self.pipeline(works(6), routing=answers_model(calls),
                clef_base_url="http://127.0.0.1:9", clef_transport=httpx.MockTransport(partial), triage_output=out)
            sidecar = json.loads(Path(out + "/triage.json").read_text())
        self.assertEqual(result.exit_code, 0, result.stderr_notes)
        self.assertEqual(sum(len(c) for c in calls), 6)
        self.assertEqual(synthesis, [1, 1, 1, 1])
        self.assertTrue(all(r["ai_ml_relevance"] == 0.8 for r in sidecar["results"]))
