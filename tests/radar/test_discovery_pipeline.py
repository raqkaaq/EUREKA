"""Planner/executor/history behavior through the normal public pipeline."""

import tempfile
import unittest
import json
from unittest import mock
import httpx
import httpx2
from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models.function import FunctionModel
from radar.pipeline import PipelineRequest, run
from radar.source.openalex import HttpxTransport
from radar.storage.sqlite import SQLiteStore
from tests.radar.discovery_support import initial_plan_dict, followup_plan_dict


def adaptive_model(prompts, *, fail_followup=False):
    def respond(messages, info):
        prompt = "\n".join(p.content for m in messages for p in m.parts
                           if isinstance(getattr(p, "content", None), str))
        prompts.append(prompt)
        followup = "WAVE: followup" in prompt
        if followup and fail_followup:
            raise ConnectionRefusedError("offline fixture")
        payload = followup_plan_dict() if followup else initial_plan_dict()
        for intent in payload["intents"]:
            intent.update(origin="agenda", source_work_ids=[])
            intent["query"].update(kind="keyword", from_date=None)
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, payload)])
    return FunctionModel(respond)


def source_client(calls, *, fail_followup=False):
    def handle(request):
        calls.append(request)
        if fail_followup and len(calls) > 6:
            return httpx.Response(503)
        # Every query rediscovers one common paper and contributes a new one.
        results = [{"id": f"https://openalex.org/W{index}", "title": f"Study {index}",
                    "publication_year": 2026, "publication_date": "2026-09-15"}
                   for index in (1000, 1000 + len(calls))]
        return httpx.Response(200, json={"results": results})
    return httpx.Client(transport=httpx.MockTransport(handle))


class NoSource:
    calls = 0
    def get_json(self, *args):
        self.calls += 1
        raise AssertionError("No OpenAlex calls before a valid plan")


class TestDiscoveryPipeline(unittest.TestCase):
    def test_planning_uses_real_strata_chat_protocol_and_closes_each_client(self):
        from radar.provider import strata
        chat_requests, clients, calls = [], [], []
        def chat(request):
            self.assertEqual(request.url.path, "/v1/chat/completions")
            body = json.loads(request.content)
            chat_requests.append(body)
            prompt = next(m["content"] for m in body["messages"] if m["role"] == "user")
            payload = followup_plan_dict() if "WAVE: followup" in prompt else initial_plan_dict()
            for intent in payload["intents"]:
                intent.update(origin="agenda", source_work_ids=[])
                intent["query"].update(kind="keyword", from_date=None)
            return httpx2.Response(200, json={"id": "offline-planning", "object": "chat.completion",
                "created": 1, "model": "offline-qwen", "choices": [{"index": 0,
                "finish_reason": "tool_calls", "message": {"role": "assistant", "tool_calls": [{
                    "id": "plan", "type": "function", "function": {
                        "name": body["tools"][0]["function"]["name"],
                        "arguments": json.dumps(payload)}}]}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 100, "total_tokens": 200}})
        def client_factory(timeout_s):
            client = httpx2.AsyncClient(transport=httpx2.MockTransport(chat), trust_env=False,
                                       follow_redirects=False, timeout=timeout_s)
            clients.append(client)
            return client
        with source_client(calls) as client, mock.patch.object(strata, "_provider_http_client",
                                                               side_effect=client_factory):
            result = run(PipelineRequest(source_override=HttpxTransport(client),
                base_url="http://127.0.0.1:1919/v1", model="offline-qwen"))
        self.assertEqual(result.exit_code, 0, result.stderr_notes)
        self.assertEqual(len(chat_requests), 2)
        self.assertEqual(len(calls), 12)
        self.assertTrue(all(c.is_closed for c in clients))
        self.assertTrue(all(b["model"] == "offline-qwen" for b in chat_requests))
        self.assertIn("intents", chat_requests[0]["tools"][0]["function"]["parameters"]["properties"])

    def test_cached_plan_preserves_original_date_and_profile_compatibility(self):
        calls, prompts = [], []
        def offline(messages, info):
            raise ConnectionRefusedError("offline fixture")
        with tempfile.TemporaryDirectory() as directory, source_client(calls) as client:
            source = HttpxTransport(client)
            first = run(PipelineRequest(storage_dir=directory, source_override=source,
                planner_model_override=adaptive_model(prompts)))
            self.assertEqual(first.exit_code, 0, first.stderr_notes)
            with SQLiteStore(directory) as store:
                original = store._db.execute("SELECT id,started_at FROM runs ORDER BY rowid LIMIT 1").fetchone()
            calls.clear()
            cached = run(PipelineRequest(storage_dir=directory, source_override=source,
                planner_model_override=FunctionModel(offline)))
            self.assertEqual(cached.exit_code, 0, cached.stderr_notes)
            self.assertEqual(len(calls), 6)
            self.assertIn(original[1], " ".join(cached.stderr_notes))
            self.assertIn("stale", " ".join(cached.stderr_notes))
            with SQLiteStore(directory) as store:
                recent = store._db.execute("SELECT id FROM runs ORDER BY rowid DESC LIMIT 1").fetchone()[0]
                self.assertEqual(store.search_waves(recent)[0].origin, "cached")
            calls.clear()
            incompatible = run(PipelineRequest(storage_dir=directory, source_override=source,
                keywords=("unrelated changed profile",), planner_model_override=FunctionModel(offline)))
            self.assertEqual(incompatible.exit_code, 3)
            self.assertEqual(calls, [])

    def test_qwen_choices_execute_in_two_waves_with_feedback_and_sqlite_history(self):
        calls, prompts = [], []
        with tempfile.TemporaryDirectory() as directory, source_client(calls) as client:
            request = PipelineRequest(mode="collect", storage_dir=directory,
                source_override=HttpxTransport(client), planner_model_override=adaptive_model(prompts))
            result = run(request)
            self.assertEqual(result.exit_code, 0, result.stderr_notes)
            self.assertEqual(len(calls), 12)
            self.assertEqual(len(prompts), 2)
            self.assertIn("new_to_run=", prompts[1])
            self.assertIn("Study 1000", prompts[1])
            self.assertEqual(calls[0].url.params["search"], initial_plan_dict()["intents"][0]["query"]["terms"])
            with SQLiteStore(directory) as store:
                run_id = store._db.execute("SELECT id FROM runs").fetchone()[0]
                records = store.search_waves(run_id)
                self.assertEqual([r.wave for r in records], [1, 2])
                self.assertEqual([len(r.feedback) for r in records], [6, 6])
                self.assertEqual(len(store.known_work_ids()), 13)

    def test_failed_followup_planning_retains_first_wave(self):
        calls, prompts = [], []
        with source_client(calls) as client:
            result = run(PipelineRequest(mode="collect", source_override=HttpxTransport(client),
                planner_model_override=adaptive_model(prompts, fail_followup=True)))
        self.assertEqual(result.exit_code, 0, result.stderr_notes)
        self.assertEqual(len(calls), 6)
        self.assertIn("retaining initial", " ".join(result.stderr_notes))

    def test_failed_followup_retrieval_retains_first_wave(self):
        calls, prompts = [], []
        with tempfile.TemporaryDirectory() as directory, source_client(calls, fail_followup=True) as client:
            result = run(PipelineRequest(mode="collect", storage_dir=directory, source_override=HttpxTransport(client),
                planner_model_override=adaptive_model(prompts)))
            with SQLiteStore(directory) as store:
                run_id = store._db.execute("SELECT id FROM runs").fetchone()[0]
                records = store.search_waves(run_id)
                self.assertEqual(records[1].retrieval_failure, "source_error")
                self.assertEqual(records[1].feedback, [])
                self.assertEqual(len(store.known_work_ids()), 7)
        self.assertEqual(result.exit_code, 0, result.stderr_notes)
        self.assertEqual(len(calls), 7)
        self.assertIn("retaining initial", " ".join(result.stderr_notes))

    def test_first_boot_planning_failure_does_not_use_static_searches(self):
        source = NoSource()
        def failed_model(messages, info):
            raise ConnectionRefusedError("offline fixture")
        result = run(PipelineRequest(mode="collect", source_override=source,
                                     planner_model_override=FunctionModel(failed_model)))
        self.assertEqual(result.exit_code, 3)
        self.assertEqual(source.calls, 0)
        self.assertIn("planning", " ".join(result.stderr_notes).lower())
