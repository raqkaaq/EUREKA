"""Full-pool offline acceptance: 106 synthetic works (102 abstracts, 4
missing) through the REAL CLEF screen_works (MockTransport), REAL
select_candidates, REAL pipeline, and FunctionModel synthesis.

Only external boundaries are doubled (HTTP transport, model). No OpenAlex
calls, no server launches, no .env reads. Synthetic decisions throughout:
nothing here is a real CLEF model judgment.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import httpx
from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models.function import FunctionModel

from radar.pipeline import PipelineRequest, run as _run_pipeline
from radar.schema.opportunities import RadarDraft
from radar.schema.papers import CollectedWork
from radar.storage import snapshots as _snapshots

MODEL = "test-clef-fullpool"


def _pool106() -> list[CollectedWork]:
    works = []
    for i in range(106):
        abstract = "" if i % 27 == 0 else (
            f"Full-pool paper {i} on diffusion models and behavioral "
            f"incentives with calibration error analysis. " * 6)[:1800]
        works.append(CollectedWork(
            openalex_id=f"https://openalex.org/W{i}", title=f"Fullpool paper {i}",
            abstract=abstract, publication_year=2025, doi="",
            primary_url=f"https://openalex.org/W{i}", locations=[],
            cited_by_count=i, matched_queries=["diffusion models"],
            query_kinds=["semantic"], score=0.0))
    return works


def _seed(tmp: str, works: list[CollectedWork]) -> str:
    summary = _snapshots.refresh_pool(works, tmp)
    return summary["snapshot"]


def _scored_handler(calls: list):
    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        body = json.loads(request.content.decode("utf-8"))
        title = body["state"]["title"]
        digest = int(hashlib.sha1(title.encode()).hexdigest()[:8], 16)
        rel = round(0.05 + (digest % 90) / 100.0, 4)
        payload = {"model": body["model"], "answers": {
            "research_importance": {"type": "noul", "noul": rel},
            "cross_domain_potential": {"type": "noul", "noul": 0.1}}}
        return httpx.Response(200, content=json.dumps(payload).encode(),
                              request=request)

    return handler


def _fixed_model(evidence=(0, 1)) -> FunctionModel:
    from radar.schema.opportunities import OpportunityDraft

    fixed = RadarDraft(
        opportunities=[OpportunityDraft(
            title="T", wow="W", investigate="I", reproduce="R",
            evidence=list(evidence))],
        ignore=[], next_move="done")

    def _impl(messages, info):
        return ModelResponse(parts=[ToolCallPart(
            tool_name="final_result", args=fixed.model_dump())])

    return FunctionModel(_impl)


class _NoOpenAlex:
    def get_json(self, url, params, headers, timeout):
        raise AssertionError("zero OpenAlex calls expected")


def _failed_routing_model() -> FunctionModel:
    def unavailable(messages, info):
        raise RuntimeError("simulated provider outage")
    return FunctionModel(unavailable)


class TestFullPoolOfflineAcceptance(unittest.TestCase):
    def test_106_pool_end_to_end_offline(self):
        works = _pool106()
        self.assertEqual(len(works), 106)
        self.assertEqual(sum(1 for w in works if not w.abstract.strip()), 4)
        calls: list = []
        transport = httpx.MockTransport(_scored_handler(calls))
        with tempfile.TemporaryDirectory() as tmp:
            snap = _seed(tmp, works)
            before = Path(snap).read_bytes()
            from pdf_support import empty_notes_model, make_loader, make_pdf_document
            docs = {w.openalex_id: make_pdf_document(w.openalex_id) for w in works}
            result = _run_pipeline(PipelineRequest(
                mode="analyze", max_candidates=8, from_snapshot=snap,
                clef_base_url="http://127.0.0.1:9", clef_model=MODEL,
                clef_transport=transport,
                model_override=_fixed_model(),
                source_override=_NoOpenAlex(),
                document_loader=make_loader(docs),
                document_model_override=empty_notes_model()))
            self.assertEqual(result.exit_code, 0)
            # 102 SystemOne calls (4 missing abstracts need none).
            self.assertEqual(len(calls), 102)
            line = " ".join(result.stderr_notes)
            for token in ("pool=106", "considered=106", "scored=102",
                          "unknown=4", "failed=0", "selected=8",
                          "analyzed=8", f"model={MODEL}"):
                self.assertIn(token, line)
            # Fixture PDFs are tiny so the full 8-paper shortlist fits the
            # PDF prompt budget; screening still covers the entire pool.
            # Every evidence URL resolves inside the pool.
            pool_ids = {w.openalex_id for w in works}
            import re

            used = set(re.findall(r"https://openalex\.org/W[0-9]+",
                                  result.stdout))
            self.assertTrue(used)
            self.assertLessEqual(used, pool_ids)
            # Cached source byte-identical, zero OpenAlex.
            self.assertEqual(Path(snap).read_bytes(), before)

    def test_both_services_down_stop_with_zero_synthesis(self):
        seen = []

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, content=b"down", request=request)

        def _model(messages, info):
            seen.append(1)
            raise AssertionError("synthesis must not run")

        with tempfile.TemporaryDirectory() as tmp:
            snap = _seed(tmp, _pool106()[:5])
            before = Path(snap).read_bytes()
            result = _run_pipeline(PipelineRequest(
                mode="analyze", max_candidates=2, from_snapshot=snap,
                clef_base_url="http://127.0.0.1:9", clef_model=MODEL,
                clef_transport=httpx.MockTransport(handler),
                triage_model_override=_failed_routing_model(),
                model_override=FunctionModel(_model),
                source_override=_NoOpenAlex()))
            self.assertEqual(result.exit_code, 3)
            self.assertEqual(seen, [])
            self.assertEqual(Path(snap).read_bytes(), before)

    def test_both_backends_malformed_stop_with_zero_synthesis(self):
        seen = []

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200, content=b'{"model": "wrong-model", "answers": {}}',
                request=request)

        def _model(messages, info):
            seen.append(1)
            raise AssertionError("synthesis must not run")

        with tempfile.TemporaryDirectory() as tmp:
            snap = _seed(tmp, _pool106()[:5])
            before = Path(snap).read_bytes()
            result = _run_pipeline(PipelineRequest(
                mode="analyze", max_candidates=2, from_snapshot=snap,
                clef_base_url="http://127.0.0.1:9", clef_model=MODEL,
                clef_transport=httpx.MockTransport(handler),
                triage_model_override=FunctionModel(lambda messages, info: ModelResponse(parts=[ToolCallPart(
                    info.output_tools[0].name, {"responses": []})])),
                model_override=FunctionModel(_model),
                source_override=_NoOpenAlex()))
            self.assertEqual(result.exit_code, 3)
            self.assertEqual(seen, [])
            self.assertEqual(Path(snap).read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
