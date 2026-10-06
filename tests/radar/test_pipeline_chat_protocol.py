"""Offline acceptance through real PydanticAI and the HTTP chat protocol.

Only the HTTP boundary is substituted. No model overrides, .env reads,
network sockets, external inference or model-quality claims are involved.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

import httpx
import httpx2

from radar.config.interests import default_profile
from radar.pipeline import PipelineRequest, run
from radar.processing.triage_input import build_input
from radar.prompts.catalog import SPECIALIST_ROLES, specialist_prompt
from radar.provider import strata as freetoken
from radar.schema.papers import CollectedWork
from radar.source import openalex
from radar.storage.snapshots import refresh_pool
from tests.radar.discovery_support import initial_plan_dict, followup_plan_dict

MODEL = "offline-chat-model"
BASE = "http://127.0.0.1:1919/v1"


def paper_pool(count: int = 106) -> list[CollectedWork]:
    return [CollectedWork(
        openalex_id=f"https://openalex.org/W{i}", title=f"Paper {i}",
        abstract="" if i % 27 == 0 else f"Paper {i}: learning and incentive mechanisms.",
        primary_url=f"https://openalex.org/W{i}", cited_by_count=i,
    ) for i in range(count)]


def user_prompt(body: dict) -> str:
    return next(message["content"] for message in body["messages"]
                if message["role"] == "user")


def is_triage(body: dict) -> bool:
    schema = body["tools"][0]["function"]["parameters"]
    return "responses" in schema["properties"]


def is_planning(body: dict) -> bool:
    schema = body["tools"][0]["function"]["parameters"]
    return "intents" in schema["properties"]


def research_role(body: dict) -> str:
    instructions = next(message["content"] for message in body["messages"]
                        if message["role"] in ("system", "developer"))
    for role in SPECIALIST_ROLES:
        if instructions == specialist_prompt(role).instructions:
            return role
    return "synthesis"


def completion(body: dict, output: dict) -> httpx2.Response:
    return httpx2.Response(200, json={
        "id": "offline-completion", "object": "chat.completion", "created": 1,
        "model": MODEL, "choices": [{"index": 0, "finish_reason": "tool_calls",
            "message": {"role": "assistant", "tool_calls": [{
                "id": "offline-tool-call", "type": "function", "function": {
                    "name": body["tools"][0]["function"]["name"],
                    "arguments": json.dumps(output),
                }}]}}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 100, "total_tokens": 200},
    })


def valid_output(body: dict) -> dict:
    if is_planning(body):
        prompt = user_prompt(body)
        payload = followup_plan_dict() if "WAVE: followup" in prompt else initial_plan_dict()
        ids = list(dict.fromkeys(re.findall(r"https://openalex\.org/W\d+", prompt)))
        for intent in payload["intents"]:
            if intent["source_work_ids"]:
                if ids:
                    intent["source_work_ids"] = [ids[0]]
                else:
                    intent.update(origin="agenda", source_work_ids=[])
        return payload
    if is_triage(body):
        papers = json.loads(user_prompt(body))["papers"]
        return {"responses": [{"work_id": paper["work_id"],
            "model": paper["input"]["model"], "answers": {
                "research_importance": {"type": "noul", "noul": 0.8},
                "cross_domain_potential": {"type": "noul", "noul": 0.6},
            }} for paper in papers]}
    return {"opportunities": [{"title": "Offline hypothesis",
        "wow": "A proposed mechanism, not a verified finding.",
        "investigate": "Compare against a control; reject if unchanged.",
        "reproduce": "Inspect the missing methods before replication.",
        "evidence": [0]}], "ignore": [], "next_move": "Check the original paper."}


@contextmanager
def chat_boundary(handler):
    """Replace just the owned HTTP client, preserving real provider creation."""
    clients = []

    def client_factory(timeout_s):
        client = httpx2.AsyncClient(transport=httpx2.MockTransport(handler),
                                   trust_env=False, follow_redirects=False,
                                   timeout=timeout_s)
        clients.append(client)
        return client

    with mock.patch.dict(os.environ, {}, clear=True), mock.patch.object(
            freetoken, "_provider_http_client", side_effect=client_factory):
        yield clients


class TestPipelineChatProtocol(unittest.TestCase):
    def run_cached(self, pool, handler, **options):
        with tempfile.TemporaryDirectory() as tmp, chat_boundary(handler) as clients:
            snapshot = refresh_pool(pool, tmp)["snapshot"]
            before = Path(snapshot).read_bytes()
            sidecar_dir = str(Path(tmp) / "triage")
            from pdf_support import empty_notes_model, make_any_loader
            settings = dict(mode="analyze", from_snapshot=snapshot, max_candidates=2,
                            base_url=BASE, model=MODEL, triage_output=sidecar_dir,
                            document_loader=make_any_loader(),
                            document_model_override=empty_notes_model())
            settings.update(options)
            result = run(PipelineRequest(**settings))
            self.assertEqual(Path(snapshot).read_bytes(), before)
            self.assertTrue(all(client.is_closed for client in clients))
            sidecar = Path(sidecar_dir) / "triage.json"
            metadata = json.loads(sidecar.read_text()) if sidecar.exists() else None
            return result, metadata, len(clients)

    def test_full_pool_routes_and_researches_over_real_chat_protocol(self):
        pool = paper_pool()
        requests = []

        def chat(request):
            self.assertEqual(request.method, "POST")
            self.assertEqual(request.url.path, "/v1/chat/completions")
            body = json.loads(request.content)
            self.assertEqual(body["model"], MODEL)
            self.assertNotIn("chat_template_kwargs", body)
            requests.append(body)
            return completion(body, valid_output(body))

        with tempfile.TemporaryDirectory() as tmp, chat_boundary(chat) as clients:
            snapshot = refresh_pool(pool, tmp)["snapshot"]
            before = Path(snapshot).read_bytes()
            sidecar = str(Path(tmp) / "triage")
            from pdf_support import empty_notes_model, make_any_loader
            result = run(PipelineRequest(
                mode="analyze", from_snapshot=snapshot, max_candidates=8,
                base_url=BASE, model=MODEL, triage_output=sidecar,
                document_loader=make_any_loader(),
                document_model_override=empty_notes_model()))
            self.assertEqual(result.exit_code, 0, result.stderr_notes)
            self.assertEqual(Path(snapshot).read_bytes(), before)
            metadata = json.loads((Path(sidecar) / "triage.json").read_text())
            self.assertEqual(metadata["rubric_version"], "clef-importance-v1")
            self.assertEqual(len(metadata["rubric_hash"]), 64)
            self.assertEqual(metadata["fallback_reason"], "missing_endpoint")
            self.assertEqual(len(clients), 2)  # Routing and research own separate sessions.
            self.assertTrue(all(client.is_closed for client in clients))

        routing = [body for body in requests if is_triage(body)]
        analysis = [body for body in requests if not is_triage(body)]
        self.assertEqual(len(routing), 51)
        self.assertEqual(len(analysis), 4)
        submitted = [paper for body in routing
                     for paper in json.loads(user_prompt(body))["papers"]]
        expected = {work.openalex_id: build_input(work, default_profile(), MODEL)
                    for work in pool if work.abstract}
        self.assertEqual(len(submitted), 102)
        self.assertEqual({paper["work_id"]: paper["input"] for paper in submitted}, expected)
        for token in ("pool=106", "scored=102", "unknown=4", "selected=8", "analyzed=8"):
            self.assertIn(token, " ".join(result.stderr_notes))

        cohorts = [re.findall(r"^\[\d+\] (Paper \d+)\b", user_prompt(body), re.MULTILINE)
                   for body in analysis]
        self.assertEqual(len(cohorts[0]), 8)
        self.assertTrue(all(cohort == cohorts[0] for cohort in cohorts))
        specialist_instructions = {specialist_prompt(role).instructions for role in SPECIALIST_ROLES}
        for body in analysis[:3]:
            instructions = next(message["content"] for message in body["messages"]
                                if message["role"] in ("system", "developer"))
            self.assertIn(instructions, specialist_instructions)
            token_limit = body.get("max_completion_tokens", body.get("max_tokens"))
            self.assertIsNotNone(token_limit)
            self.assertEqual(token_limit, 2000)
        synthesis = user_prompt(analysis[-1])
        for role in SPECIALIST_ROLES:
            self.assertIn(role, synthesis)
        self.assertIn("untrusted model-generated data", synthesis)
        self.assertIn("**The Wow:**", result.stdout)
        self.assertIn("**Reproduce:**", result.stdout)
        first_id = next(work.openalex_id for work in pool if work.title == cohorts[0][0])
        self.assertIn(first_id, result.stdout)

    def test_invalid_specialist_evidence_is_retried_over_chat_then_recovers(self):
        seen = []

        def chat(request):
            body = json.loads(request.content)
            stage = "triage" if is_triage(body) else research_role(body)
            seen.append(stage)
            output = valid_output(body)
            if stage == "ml_methods" and seen.count(stage) == 1:
                output["opportunities"][0]["evidence"] = [199]
            return completion(body, output)

        result, metadata, client_count = self.run_cached(paper_pool()[1:3], chat)
        self.assertEqual(result.exit_code, 0, result.stderr_notes)
        self.assertEqual(seen.count("ml_methods"), 2)
        self.assertEqual(seen.count("triage"), 1)
        self.assertEqual(seen.count("synthesis"), 1)
        self.assertEqual(len(seen), 6)
        self.assertEqual(client_count, 2)
        self.assertEqual(len(metadata["results"]), 2)
        self.assertNotIn("W199", result.stdout)

    def test_persistently_invalid_specialist_stops_before_synthesis(self):
        seen = []

        def chat(request):
            body = json.loads(request.content)
            stage = "triage" if is_triage(body) else research_role(body)
            seen.append(stage)
            output = valid_output(body)
            if stage == "ml_methods":
                output["opportunities"][0]["evidence"] = [199]
            return completion(body, output)

        result, metadata, client_count = self.run_cached(paper_pool()[1:3], chat)
        self.assertEqual(result.exit_code, 3)
        self.assertEqual(result.stdout, "")
        self.assertEqual(seen.count("ml_methods"), 2)
        self.assertNotIn("synthesis", seen)
        self.assertEqual(client_count, 2)
        self.assertTrue(all(row["status"] == "scored" for row in metadata["results"]))
        self.assertIn("no partial report produced", " ".join(result.stderr_notes))

    def test_upstream_chat_error_body_is_not_exposed_in_pipeline_output(self):
        secret_marker = "UPSTREAM_PRIVATE_BODY_SENTINEL"
        seen = []

        def chat(request):
            body = json.loads(request.content)
            stage = "triage" if is_triage(body) else research_role(body)
            seen.append(stage)
            if stage == "ml_methods":
                return httpx2.Response(400, json={"error": {
                    "message": secret_marker, "type": "invalid_request_error"}})
            return completion(body, valid_output(body))

        result, _, client_count = self.run_cached(paper_pool()[1:3], chat)
        self.assertEqual(result.exit_code, 3)
        self.assertEqual(result.stdout, "")
        self.assertNotIn("synthesis", seen)
        self.assertNotIn(secret_marker, " ".join(result.stderr_notes))
        self.assertEqual(client_count, 2)

    def test_malformed_wire_answers_fail_routing_without_research(self):
        for invalidity in ("boolean_probability", "wrong_model", "foreign_id", "missing_answer"):
            with self.subTest(invalidity=invalidity):
                seen = []

                def chat(request):
                    body = json.loads(request.content)
                    self.assertTrue(is_triage(body), "Research must not run after failed routing")
                    seen.append(body)
                    output = valid_output(body)
                    row = output["responses"][0]
                    if invalidity == "boolean_probability":
                        row["answers"]["research_importance"]["noul"] = True
                    elif invalidity == "wrong_model":
                        row["model"] = "different-model"
                    elif invalidity == "foreign_id":
                        row["work_id"] = "https://openalex.org/W999"
                    else:
                        del row["answers"]["cross_domain_potential"]
                    return completion(body, output)

                result, metadata, client_count = self.run_cached(paper_pool()[1:2], chat)
                self.assertEqual(result.exit_code, 3)
                self.assertEqual(result.stdout, "")
                self.assertEqual(len(seen), 2)
                self.assertEqual(client_count, 1)
                self.assertEqual(metadata["results"][0]["status"], "failed")
                self.assertIsNone(metadata["results"][0]["research_importance"])

    def test_cached_collect_mode_performs_no_chat_requests(self):
        def forbidden_chat(request):
            raise AssertionError("Metadata-only runs must not invoke models")

        result, metadata, client_count = self.run_cached(
            paper_pool()[1:3], forbidden_chat, mode="collect")
        self.assertEqual(result.exit_code, 0)
        self.assertEqual(client_count, 0)
        self.assertIsNone(metadata)

    def test_thinking_disable_is_opt_in_and_serialized_for_every_stage(self):
        seen = []

        def chat(request):
            body = json.loads(request.content)
            seen.append(body)
            return completion(body, valid_output(body))

        result, _, _ = self.run_cached(paper_pool()[1:3], chat, disable_thinking=True)
        self.assertEqual(result.exit_code, 0, result.stderr_notes)
        self.assertEqual(len(seen), 5)
        self.assertTrue(all(body["chat_template_kwargs"] == {"enable_thinking": False}
                            for body in seen))

    def test_native_clef_and_failed_clef_use_the_same_real_chat_research_path(self):
        pool = paper_pool()
        expected = {work.title: build_input(work, default_profile(), MODEL)
                    for work in pool if work.abstract}
        for available in (True, False):
            with self.subTest(native_clef_available=available):
                native_inputs = []
                chat_requests = []

                def clef(request):
                    self.assertEqual(request.url.path, "/v1/systemone")
                    body = json.loads(request.content)
                    native_inputs.append(body)
                    self.assertEqual(body, expected[body["state"]["title"]])
                    if not available:
                        return httpx.Response(500, json={"error": "offline server unavailable"})
                    return httpx.Response(200, json={"model": MODEL, "answers": {
                        "research_importance": {"type": "noul", "noul": 0.8},
                        "cross_domain_potential": {"type": "noul", "noul": 0.6},
                    }})

                def chat(request):
                    body = json.loads(request.content)
                    chat_requests.append(body)
                    return completion(body, valid_output(body))

                result, metadata, client_count = self.run_cached(
                    pool, chat, max_candidates=8,
                    clef_base_url="http://127.0.0.1:9/v1", clef_model=MODEL,
                    clef_transport=httpx.MockTransport(clef))
                self.assertEqual(result.exit_code, 0, result.stderr_notes)
                if available:
                    self.assertEqual(len(native_inputs), 102)
                else:
                    # An unavailable native endpoint can fail fast; fallback must
                    # still screen every abstract, not only unattempted papers.
                    self.assertGreater(len(native_inputs), 0)
                    self.assertLessEqual(len(native_inputs), 102)
                self.assertEqual(sum(is_triage(body) for body in chat_requests), 0 if available else 51)
                self.assertEqual(sum(not is_triage(body) for body in chat_requests), 4)
                self.assertEqual(client_count, 1 if available else 2)
                self.assertEqual(metadata["backend"], "clef" if available else "qwen")
                self.assertEqual(metadata["probability_kind"], "native_noul" if available else "prompted_estimate")
                self.assertEqual(metadata["fallback_reason"], None if available else "screening_failed")
                self.assertEqual(len(metadata["results"]), 106)
                rerouted = [paper for body in chat_requests if is_triage(body)
                            for paper in json.loads(user_prompt(body))["papers"]]
                self.assertEqual(len(rerouted), 0 if available else 102)

    def test_shared_deadline_cancels_outstanding_http_and_closes_clients(self):
        started = []
        settled = []

        async def chat(request):
            body = json.loads(request.content)
            if is_triage(body):
                return completion(body, valid_output(body))
            role = research_role(body)
            started.append(role)
            try:
                await asyncio.sleep(10)  # Substitute an unresponsive remote HTTP request.
                return completion(body, valid_output(body))
            finally:
                settled.append(role)

        result, metadata, client_count = self.run_cached(
            paper_pool()[1:3], chat, analysis_timeout_s=0.5)
        self.assertEqual(result.exit_code, 3)
        self.assertEqual(result.stdout, "")
        self.assertGreater(len(started), 0)
        self.assertLessEqual(len(started), 2)
        self.assertCountEqual(started, settled)
        self.assertNotIn("synthesis", started)
        self.assertIn("overall 0.5s deadline", " ".join(result.stderr_notes))
        self.assertEqual(client_count, 2)
        self.assertTrue(all(row["status"] == "scored" for row in metadata["results"]))

    def test_unstructured_text_does_not_bypass_the_answer_contract(self):
        seen = []

        def chat(request):
            body = json.loads(request.content)
            self.assertTrue(is_triage(body))
            seen.append(body)
            return httpx2.Response(200, json={
                "id": "offline-text", "object": "chat.completion", "created": 1,
                "model": MODEL, "choices": [{"index": 0, "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "This paper looks relevant."}}],
            })

        result, metadata, client_count = self.run_cached(paper_pool()[1:2], chat)
        self.assertEqual(result.exit_code, 3)
        self.assertEqual(result.stdout, "")
        self.assertEqual(len(seen), 2)
        self.assertEqual(client_count, 1)
        self.assertEqual(metadata["results"][0]["status"], "failed")

    def test_fresh_collection_snapshot_routing_and_report_use_only_http_boundaries(self):
        source_requests = []
        chat_requests = []

        def source_response(request):
            self.assertEqual(request.method, "GET")
            self.assertEqual(request.url.host, "api.openalex.org")
            self.assertEqual(request.url.path, "/works")
            if "filter" in request.url.params:
                self.assertIn("from_publication_date:", request.url.params["filter"])
                self.assertEqual(request.url.params["sort"], "publication_date:desc")
            self.assertLessEqual(int(request.url.params["per-page"]), 50)
            source_requests.append(request)
            i = len(source_requests)
            return httpx.Response(200, json={"results": [
                {"id": "https://openalex.org/W1000", "title": "Shared discovery paper",
                 "abstract_inverted_index": {"Learning": [0], "incentives": [1]},
                 "publication_year": 2026, "publication_date": "2026-10-04", "cited_by_count": 5},
                {"id": f"https://openalex.org/W{1000+i}", "title": f"Branch paper {i}",
                 "abstract_inverted_index": None if i == 6 else {"Learning": [0], "mechanisms": [1]},
                 "publication_year": 2026, "publication_date": "2026-10-04", "cited_by_count": i},
            ]})

        def chat(request):
            body = json.loads(request.content)
            chat_requests.append(body)
            return completion(body, valid_output(body))

        source_client = httpx.Client(transport=httpx.MockTransport(source_response),
                                     trust_env=False, follow_redirects=False)
        with tempfile.TemporaryDirectory() as tmp, chat_boundary(chat) as clients, mock.patch.object(
                openalex, "default_client", return_value=source_client):
            from pdf_support import empty_notes_model, make_any_loader
            result = run(PipelineRequest(
                mode="analyze", refresh_dir=tmp, max_candidates=3,
                base_url=BASE, model=MODEL, triage_output=tmp,
                document_loader=make_any_loader(),
                document_model_override=empty_notes_model()))
            self.assertEqual(result.exit_code, 0, result.stderr_notes)
            snapshot = json.loads((Path(tmp) / "snapshot.json").read_text())
            self.assertEqual(len(snapshot["works"]), 13)
            self.assertEqual(snapshot["coverage"]["collected"], 13)
            self.assertEqual(snapshot["coverage"]["with_abstracts"], 12)
            self.assertEqual(snapshot["coverage"]["missing_abstracts"], 1)
            self.assertEqual(snapshot["coverage"]["llm_selected"], 3)
            self.assertEqual(snapshot["coverage"]["llm_analyzed"], 3)
            sidecar = json.loads((Path(tmp) / "triage.json").read_text())
            self.assertEqual(len(sidecar["results"]), 13)
            self.assertEqual(sum(row["status"] == "scored" for row in sidecar["results"]), 12)
            self.assertEqual(sum(row["status"] == "missing_abstract" for row in sidecar["results"]), 1)
            self.assertFalse((Path(tmp) / "snapshot.lock").exists())
            self.assertTrue(all(client.is_closed for client in clients))
        self.assertTrue(source_client.is_closed)
        self.assertEqual(len(source_requests), 12)
        self.assertEqual(sum(is_triage(body) for body in chat_requests), 6)
        self.assertEqual(sum(is_planning(body) for body in chat_requests), 2)
        self.assertEqual(sum(not is_triage(body) and not is_planning(body)
                             for body in chat_requests), 4)
        self.assertIn("**Evidence:**", result.stdout)
        self.assertIn("https://openalex.org/W", result.stdout)

    def test_source_http_failure_preserves_previous_snapshot_without_investigation(self):
        marker = "SOURCE_PRIVATE_BODY_SENTINEL"
        requests = []

        def source_response(request):
            requests.append(request)
            return httpx.Response(500, text=marker)

        planning_requests = []

        def planning_only(request):
            body = json.loads(request.content)
            self.assertTrue(is_planning(body), "Failed collection must stop before investigation")
            planning_requests.append(body)
            return completion(body, valid_output(body))

        source_client = httpx.Client(transport=httpx.MockTransport(source_response), trust_env=False)
        with tempfile.TemporaryDirectory() as tmp, chat_boundary(planning_only) as clients, mock.patch.object(
                openalex, "default_client", return_value=source_client):
            snapshot = refresh_pool(paper_pool()[1:3], tmp)["snapshot"]
            before = Path(snapshot).read_bytes()
            result = run(PipelineRequest(mode="analyze", refresh_dir=tmp,
                                         base_url=BASE, model=MODEL))
            self.assertEqual(result.exit_code, 2)
            self.assertEqual(result.stdout, "")
            self.assertEqual(Path(snapshot).read_bytes(), before)
            self.assertEqual(len(clients), 1)
            self.assertTrue(all(client.is_closed for client in clients))
            self.assertNotIn(marker, " ".join(result.stderr_notes))
        self.assertTrue(source_client.is_closed)
        self.assertEqual(len(planning_requests), 1)
        self.assertEqual(len(requests), 1)

    def test_maximum_200_paper_pool_is_completely_routed_over_chat(self):
        seen = []

        def chat(request):
            body = json.loads(request.content)
            seen.append(body)
            return completion(body, valid_output(body))

        result, metadata, client_count = self.run_cached(paper_pool(200), chat)
        self.assertEqual(result.exit_code, 0, result.stderr_notes)
        self.assertEqual(client_count, 2)
        self.assertEqual(len(metadata["results"]), 200)
        self.assertEqual(sum(row["status"] == "scored" for row in metadata["results"]), 192)
        self.assertEqual(sum(row["status"] == "missing_abstract" for row in metadata["results"]), 8)
        submitted_ids = [paper["work_id"] for body in seen if is_triage(body)
                         for paper in json.loads(user_prompt(body))["papers"]]
        self.assertEqual(len(submitted_ids), 192)
        self.assertEqual(len(set(submitted_ids)), 192)
        self.assertEqual(sum(is_triage(body) for body in seen), 96)
        self.assertEqual(sum(not is_triage(body) for body in seen), 4)
        for token in ("pool=200", "scored=192", "unknown=8", "selected=2", "analyzed=2"):
            self.assertIn(token, " ".join(result.stderr_notes))

    def test_coerced_specialist_citations_are_rejected_and_retried_on_the_wire(self):
        for index in (True, "1", 1.0):
            with self.subTest(index_type=type(index).__name__):
                seen = []

                def chat(request):
                    body = json.loads(request.content)
                    stage = "triage" if is_triage(body) else research_role(body)
                    seen.append(stage)
                    output = valid_output(body)
                    if stage == "ml_methods" and seen.count(stage) == 1:
                        output["opportunities"][0]["evidence"] = [index]
                    return completion(body, output)

                result, _, client_count = self.run_cached(paper_pool()[1:3], chat)
                self.assertEqual(result.exit_code, 0, result.stderr_notes)
                self.assertEqual(seen.count("ml_methods"), 2)
                self.assertEqual(seen.count("synthesis"), 1)
                self.assertEqual(client_count, 2)

    def test_coerced_final_citations_fail_closed_after_bounded_retries(self):
        seen = []

        def chat(request):
            body = json.loads(request.content)
            stage = "triage" if is_triage(body) else research_role(body)
            seen.append(stage)
            output = valid_output(body)
            if stage == "synthesis":
                output["opportunities"][0]["evidence"] = [True]
            return completion(body, output)

        result, _, client_count = self.run_cached(paper_pool()[1:3], chat)
        self.assertEqual(result.exit_code, 3)
        self.assertEqual(result.stdout, "")
        self.assertEqual(seen.count("synthesis"), 2)
        self.assertEqual(client_count, 2)

    def test_malformed_publisher_url_no_longer_aborts_a_valid_cached_report(self):
        pool = paper_pool()[1:3]
        for work in pool:
            work.primary_url = "http://[malformed"

        def chat(request):
            body = json.loads(request.content)
            return completion(body, valid_output(body))

        result, _, client_count = self.run_cached(pool, chat)
        self.assertEqual(result.exit_code, 0, result.stderr_notes)
        self.assertEqual(client_count, 2)
        self.assertIn("**Evidence:**", result.stdout)
        self.assertIn("https://openalex.org/W1", result.stdout)
        self.assertNotIn("http://[malformed", result.stdout)


if __name__ == "__main__":
    unittest.main()
