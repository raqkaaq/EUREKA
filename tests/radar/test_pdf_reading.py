"""Full-PDF LLM reading through real PydanticAI FunctionModel seams."""

from __future__ import annotations

import asyncio
import unittest
from types import SimpleNamespace
from unittest import mock

from pydantic_ai.messages import ModelResponse, ToolCallPart, UserPromptPart
from pydantic_ai.models.function import FunctionModel

from radar.schema.documents import (
    ChunkReading,
    DocumentNotes,
    PDFDocument,
    PDFPage,
    PDFReading,
    PageEvidence,
)
from radar.schema.opportunities import OpportunityDraft, RadarDraft
from radar.schema.papers import CollectedWork


SHA = "a" * 64


def make_doc(work_id="https://openalex.org/W0", texts=("hello world paper text",)):
    pages = [PDFPage(number=i + 1, text=text) for i, text in enumerate(texts)]
    return PDFDocument(
        work_id=work_id, source_url="https://example.org/paper.pdf",
        sha256=SHA, relative_path=f"pdf/{work_id.rsplit('/', 1)[-1]}-{SHA}.pdf", pages=pages)


def make_work(openalex_id="https://openalex.org/W0"):
    return CollectedWork(openalex_id=openalex_id, title="Paper 0",
                         abstract="Learning mechanisms and human incentives.")


def valid_notes(quote="hello", page=1):
    return DocumentNotes(
        summary="summary of findings", methods="methods used",
        results="results observed", limitations="limits noted",
        evidence=[PageEvidence(page=page, quote=quote, finding="supports claim")])


def user_prompt(messages):
    return next(part.content for message in messages for part in message.parts
                if isinstance(part, UserPromptPart))


class TestChunking(unittest.TestCase):
    def test_chunk_covers_every_character_and_page(self):
        from radar.agent.pdf_reading import build_full_text, chunk_document

        doc = make_doc(texts=("A" * 100, "B" * 100))
        full, segments = build_full_text(doc)
        self.assertIn("[page 1]", full)
        self.assertIn("[page 2]", full)
        full2, spans = chunk_document(doc)
        self.assertEqual(full, full2)
        self.assertEqual(spans[0][0], 0)
        self.assertEqual(spans[-1][1], len(full))
        for i in range(len(spans) - 1):
            self.assertEqual(spans[i][1], spans[i + 1][0])
        covered = set()
        for _, _, pages in spans:
            covered.update(pages)
        self.assertEqual(covered, {1, 2})

    def test_pdf_reading_yaml_preflight_typed(self):
        from radar.prompts.catalog import (
            pdf_reading_prompt, pdf_reduction_prompt, validate_pdf_prompts)
        from radar.schema.configuration import AgentPrompt

        self.assertIsInstance(pdf_reading_prompt(), AgentPrompt)
        self.assertIsInstance(pdf_reduction_prompt(), AgentPrompt)
        validate_pdf_prompts()

    def test_pdf_prompts_state_schema_character_limits(self):
        # The live model overran these fields even after a schema retry.
        # Both chunk reading and reduction must spell out the output contract.
        from radar.prompts.catalog import pdf_reading_prompt, pdf_reduction_prompt

        for loader in (pdf_reading_prompt, pdf_reduction_prompt):
            with self.subTest(prompt=loader.__name__):
                instructions = loader().instructions
                for contract in (
                    "summary: at most 600 characters",
                    "methods: at most 300 characters",
                    "results: at most 300 characters",
                    "limitations: at most 300 characters",
                    "quote: at most 180 characters",
                    "finding: at most 180 characters",
                ):
                    self.assertIn(contract, instructions)
                self.assertIn("characters, not words", instructions)


class TestReaderValidation(unittest.TestCase):
    def test_schema_repair_can_be_followed_by_quote_repair(self):
        from radar.agent.pdf_reading import read_pdf

        replies = []

        def respond(messages, info):
            replies.append(messages)
            payload = valid_notes().model_dump()
            if len(replies) == 1:
                payload["methods"] = "m" * 301
            elif len(replies) == 2:
                payload["evidence"][0]["quote"] = "invented quotation"
            return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, payload)])

        reading = read_pdf(make_doc(), FunctionModel(respond))
        self.assertEqual(reading.notes.evidence[0].quote, "hello")
        self.assertEqual(len(replies), 3)

    def test_external_cancellation_is_not_swallowed(self):
        from radar.agent.pdf_reading import read_documents_async

        async def stall(messages, info):
            await asyncio.sleep(10)

        async def cancel_reader():
            task = asyncio.create_task(read_documents_async(
                [make_doc()], FunctionModel(stall), timeout_s=5))
            await asyncio.sleep(0.01)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task

        asyncio.run(cancel_reader())

    def test_boolean_document_deadline_is_rejected(self):
        from radar.agent.pdf_reading import validate_document_timeout

        with self.assertRaises(ValueError):
            validate_document_timeout(True)

    def test_late_page_evidence_reaches_reader(self):
        from radar.agent.pdf_reading import read_pdf

        late = "latepage marker ZETA-42 unique tail content here"
        doc = make_doc(texts=("first page body text here", f"second page {late}"))
        seen = []

        def respond(messages, info):
            prompt = user_prompt(messages)
            seen.append(prompt)
            from radar.prompts.catalog import pdf_reading_prompt
            if info.instructions == pdf_reading_prompt().instructions:
                # quote must be an exact substring of the supplied chunk
                quote = late if late in prompt else "first page body"
                return ModelResponse(parts=[ToolCallPart(
                    info.output_tools[0].name,
                    valid_notes(quote=quote, page=2 if late in prompt else 1).model_dump())])
            # reduction (single chunk -> no reduction expected, but handle)
            child = valid_notes(quote="first page body", page=1)
            return ModelResponse(parts=[ToolCallPart(
                info.output_tools[0].name, child.model_dump())])

        reading = read_pdf(doc, FunctionModel(respond))
        self.assertIsInstance(reading, PDFReading)
        combined = " ".join(seen)
        self.assertIn("ZETA-42", combined)

    def test_lossless_multiple_chunks_every_char_submitted(self):
        from radar.agent.pdf_reading import chunk_document, read_pdf

        texts = ("x-A " * 1500, "y-B " * 1500, "z-C " * 500)  # >6000 chars -> 3 chunks
        doc = make_doc(texts=texts)
        _, spans = chunk_document(doc)
        self.assertGreater(len(spans), 1)
        prompts = []

        def respond(messages, info):
            from radar.prompts.catalog import pdf_reading_prompt, pdf_reduction_prompt
            prompt = user_prompt(messages)
            if info.instructions == pdf_reading_prompt().instructions:
                prompts.append(prompt)
                # echo a real substring of this chunk as the quote
                chunk_body = prompt.split("--- begin chunk")[1]
                # find a stable token present in this chunk
                token = "x-A" if "x-A" in prompt else ("y-B" if "y-B" in prompt else "z-C")
                page = 1 if "x-A" in prompt else (2 if "y-B" in prompt else 3)
                # locate pages listed in prompt header for safety
                return ModelResponse(parts=[ToolCallPart(
                    info.output_tools[0].name,
                    valid_notes(quote=token, page=page).model_dump())])
            if info.instructions == pdf_reduction_prompt().instructions:
                # reducer: reuse first child evidence verbatim
                import json as _json
                payload = prompt.split("--- begin child notes ---")[1].split("--- end child notes ---")[0]
                children = _json.loads(payload)
                first = children[0]["evidence"][0]
                merged = valid_notes(quote=first["quote"], page=first["page"])
                return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, merged.model_dump())])
            raise AssertionError("unexpected agent")

        reading = read_pdf(doc, FunctionModel(respond))
        self.assertEqual(len(reading.chunks), len(spans))
        self.assertEqual(len(prompts), len(spans))
        # contiguous exact offsets, full coverage
        self.assertEqual(reading.chunks[0].start, 0)
        self.assertEqual(reading.chunks[-1].end, reading.text_chars)
        for i in range(len(reading.chunks) - 1):
            self.assertEqual(reading.chunks[i].end, reading.chunks[i + 1].start)
        covered = set()
        for chunk in reading.chunks:
            covered.update(chunk.pages)
        self.assertEqual(covered, {1, 2, 3})

    def test_all_notes_consumed_in_reduction(self):
        from radar.agent.pdf_reading import chunk_document, read_pdf

        # force small chunks by using long text; ensure >=5 chunks
        big = " ".join(f"segment-{i} " * 800 for i in range(6))
        doc = make_doc(texts=(big,))
        _, spans = chunk_document(doc)
        self.assertGreaterEqual(len(spans), 2)
        reduction_payloads = []
        leaf_markers = []

        def respond(messages, info):
            from radar.prompts.catalog import pdf_reading_prompt, pdf_reduction_prompt
            prompt = user_prompt(messages)
            if info.instructions == pdf_reading_prompt().instructions:
                index = int(prompt.split("CHUNK ", 1)[1].split(" ", 1)[0])
                marker = f"leaf-{index}"
                leaf_markers.append(marker)
                token = "segment-0" if "segment-0" in prompt else "segment"
                # page is always 1 here (single page doc)
                return ModelResponse(parts=[ToolCallPart(
                    info.output_tools[0].name,
                    valid_notes(quote=token, page=1).model_copy(update={"summary": marker}).model_dump())])
            reduction_payloads.append(prompt)
            import json as _json
            payload = prompt.split("--- begin child notes ---")[1].split("--- end child notes ---")[0]
            children = _json.loads(payload)
            first = children[0]["evidence"][0]
            merged = valid_notes(quote=first["quote"], page=first["page"])
            merged = merged.model_copy(update={"summary": "|".join(child["summary"] for child in children)})
            return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, merged.model_dump())])

        reading = read_pdf(doc, FunctionModel(respond))
        self.assertIsInstance(reading.notes, DocumentNotes)
        if len(spans) > 1:
            self.assertTrue(reduction_payloads)
            # every chunk note (including last) must appear in some reduction payload
            self.assertEqual(set(reading.notes.summary.split("|")), set(leaf_markers))
            self.assertEqual(len(reading.notes.summary.split("|")), len(spans))
            self.assertEqual(len(reading.chunks), len(spans))

    def test_invented_page_retries_then_fails(self):
        from radar.agent.pdf_reading import read_pdf
        from radar.provider.strata import StrataError

        doc = make_doc()
        calls = []

        def respond(messages, info):
            calls.append(1)
            bad = {"summary": "summary of findings", "methods": "methods used",
                   "results": "results observed", "limitations": "limits noted",
                   "evidence": [{"page": 99, "quote": "hello world paper text",
                                 "finding": "supports claim"}]}
            return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, bad)])

        with self.assertRaises(StrataError):
            read_pdf(doc, FunctionModel(respond))
        self.assertEqual(len(calls), 3)  # two bounded validation retries

    def test_invented_quote_retries_then_fails(self):
        from radar.agent.pdf_reading import read_pdf
        from radar.provider.strata import StrataError

        doc = make_doc()
        calls = []

        def respond(messages, info):
            calls.append(1)
            bad = valid_notes(quote="totally invented phrase not in source", page=1)
            return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, bad.model_dump())])

        with self.assertRaises(StrataError):
            read_pdf(doc, FunctionModel(respond))
        self.assertEqual(len(calls), 3)

    def test_incomplete_coverage_rejected(self):
        with self.assertRaises(Exception):
            PDFReading(
                work_id="w", source_url="https://example.org/p.pdf", sha256=SHA,
                relative_path=f"pdf/W0-{SHA}.pdf", page_count=2, text_chars=10,
                chunks=[ChunkReading(index=0, start=0, end=5, pages=[1],
                                     notes=valid_notes(quote="hello", page=1))],
                notes=valid_notes(quote="hello", page=1),
                extraction_warning="warn")


class TestDocumentsBatch(unittest.TestCase):
    def test_failures_recorded_without_partial_reading(self):
        from radar.agent.pdf_reading import read_documents
        from radar.prompts.catalog import pdf_reading_prompt

        good = make_doc(work_id="https://openalex.org/W0", texts=("good paper body text",))
        bad_texts = ("bad body",)
        bad = make_doc(work_id="https://openalex.org/W1", texts=bad_texts)

        def respond(messages, info):
            prompt = user_prompt(messages)
            if "bad body" in prompt:
                evil = valid_notes(quote="invented nowhere", page=1)
                return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, evil.model_dump())])
            return ModelResponse(parts=[ToolCallPart(
                info.output_tools[0].name,
                valid_notes(quote="good paper body text", page=1).model_dump())])

        readings, failures = read_documents([good, bad], FunctionModel(respond))
        self.assertEqual(len(readings), 1)
        self.assertEqual(readings[0].work_id, "https://openalex.org/W0")
        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0].work_id, "https://openalex.org/W1")

    def test_global_deadline_records_remaining_failed(self):
        from radar.agent.pdf_reading import read_documents_async

        docs = [make_doc(work_id=f"https://openalex.org/W{i}") for i in range(3)]

        async def stall(messages, info):
            await asyncio.Event().wait()

        readings, failures = asyncio.run(
            read_documents_async(docs, FunctionModel(stall), timeout_s=0.1))
        self.assertEqual(readings, [])
        self.assertEqual(len(failures), 3)
        self.assertTrue(all(f.category for f in failures))


class TestResearchTeamPdf(unittest.TestCase):
    def test_final_agent_retries_unverified_page_then_accepts_verified_page(self):
        from radar.agent.research_team import research_candidates
        from radar.prompts.catalog import opportunity_analysis_prompt
        from test_learning_dossier import _dossier_dict

        base_respond, _ = self._pdf_double()
        synthesis_calls = []
        def respond(messages, info):
            if info.instructions == opportunity_analysis_prompt().instructions:
                synthesis_calls.append(1)
                page = 2 if len(synthesis_calls) == 1 else 1
                return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, {
                    "learning_dossiers": [_dossier_dict(supporting_pages=[page])],
                    "opportunities": [], "ignore": [], "next_move": "Study the argument."})])
            return base_respond(messages, info)

        result = research_candidates([make_work()], model=FunctionModel(respond),
                                     documents=[make_doc(texts=("good paper body text", "Uncited page two."))])
        self.assertEqual(len(synthesis_calls), 2)
        self.assertEqual(result.draft.learning_dossiers[0].supporting_pages, [1])

    def _pdf_double(self, quote="good paper body text", page=1):
        from radar.prompts.catalog import (
            SPECIALIST_ROLES, opportunity_analysis_prompt, pdf_reading_prompt,
            pdf_reduction_prompt, specialist_prompt)

        roles = {specialist_prompt(r).instructions: r for r in SPECIALIST_ROLES}
        reader = pdf_reading_prompt().instructions
        reducer = pdf_reduction_prompt().instructions
        synth = opportunity_analysis_prompt().instructions
        calls = {"reading": 0, "roles": []}

        def respond(messages, info):
            if info.instructions == reader:
                calls["reading"] += 1
                return ModelResponse(parts=[ToolCallPart(
                    info.output_tools[0].name, valid_notes(quote=quote, page=page).model_dump())])
            if info.instructions == reducer:
                calls["reading"] += 1
                return ModelResponse(parts=[ToolCallPart(
                    info.output_tools[0].name, valid_notes(quote=quote, page=page).model_dump())])
            if info.instructions in roles:
                calls["roles"].append(roles[info.instructions])
                draft = RadarDraft(opportunities=[OpportunityDraft(
                    title="H", evidence=[0])], next_move="check")
                return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, draft.model_dump())])
            if info.instructions == synth:
                calls["roles"].append("synthesis")
                return ModelResponse(parts=[ToolCallPart(
                    info.output_tools[0].name, RadarDraft(next_move="done").model_dump())])
            raise AssertionError("unexpected instructions")

        return respond, calls

    def test_reading_calls_precede_four_role_calls_with_pdf_blocks(self):
        from radar.agent.research_team import research_candidates

        doc = make_doc(texts=("good paper body text",))
        respond, calls = self._pdf_double()
        result = research_candidates([make_work()], model=FunctionModel(respond), documents=[doc])
        self.assertGreaterEqual(calls["reading"], 1)
        self.assertEqual(len(calls["roles"]), 4)
        self.assertIn("full extracted PDF text read via", result.prompt)
        self.assertIn("good paper body text", result.prompt)
        self.assertEqual(len(result.included), 1)
        self.assertEqual(len(result.document_readings), 1)

    def test_failures_produce_zero_included_no_abstract_fallthrough(self):
        from radar.agent.research_team import research_candidates

        doc = make_doc(texts=("good paper body text",))
        synth_calls = []

        def respond(messages, info):
            from radar.prompts.catalog import pdf_reading_prompt
            if info.instructions == pdf_reading_prompt().instructions:
                bad = valid_notes(quote="invented nowhere", page=1)
                return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, bad.model_dump())])
            synth_calls.append(info.instructions)
            raise AssertionError("synthesis must not run when all readings fail")

        result = research_candidates([make_work()], model=FunctionModel(respond), documents=[doc])
        self.assertEqual(result.included, ())
        self.assertEqual(result.specialist_reports, ())
        self.assertEqual(len(result.document_failures), 1)
        self.assertEqual(synth_calls, [])
        self.assertNotIn("Learning mechanisms", result.prompt)

    def test_duplicate_and_unrelated_docs_rejected(self):
        from radar.agent.research_team import research_candidates

        good = make_doc(work_id="https://openalex.org/W0", texts=("good paper body text",))
        dup = make_doc(work_id="https://openalex.org/W0", texts=("good paper body text",))
        strange = make_doc(work_id="https://openalex.org/W99", texts=("good paper body text",))
        respond, _ = self._pdf_double()
        result = research_candidates([make_work()], model=FunctionModel(respond),
                                     documents=[good, dup, strange])
        self.assertEqual(len(result.included), 1)
        categories = [f.category for f in result.document_failures]
        self.assertIn("duplicate", categories)
        self.assertIn("unrelated", categories)

    def test_owned_session_closes_on_pdf_path(self):
        from radar.agent.research_team import research_candidates
        from radar.provider.strata import StrataSession

        doc = make_doc(texts=("good paper body text",))
        respond, _ = self._pdf_double()
        client = SimpleNamespace(aclose=mock.AsyncMock())
        result = research_candidates(
            [make_work()], session=StrataSession(FunctionModel(respond), client),
            documents=[doc])
        self.assertEqual(len(result.included), 1)
        client.aclose.assert_awaited_once()

    def test_final_dossier_pages_validated(self):
        from radar.agent.research_team import validate_pdf_dossier_pages

        doc = make_doc(texts=("good paper body text",))
        reading = PDFReading(
            work_id=doc.work_id, source_url=doc.source_url, sha256=doc.sha256,
            relative_path=doc.relative_path, page_count=2, text_chars=10,
            chunks=[ChunkReading(index=0, start=0, end=5, pages=[1],
                                 notes=valid_notes(quote="good", page=1)),
                    ChunkReading(index=1, start=5, end=10, pages=[2],
                                 notes=valid_notes(quote="good", page=1))],
            notes=DocumentNotes(summary="s" * 10, methods="m" * 10,
                                results="r" * 10, limitations="l" * 10,
                                evidence=[PageEvidence(page=2, quote="good", finding="f")]),
            extraction_warning="warn")
        included = [make_work()]
        readings = {"https://openalex.org/W0": reading}
        ok_dossier = SimpleNamespace(paper_index=0, supporting_pages=[2])
        ok_draft = SimpleNamespace(learning_dossiers=[ok_dossier])
        self.assertIs(validate_pdf_dossier_pages(ok_draft, included, readings), ok_draft)
        empty = SimpleNamespace(learning_dossiers=[SimpleNamespace(paper_index=0, supporting_pages=[])])
        with self.assertRaises(ValueError):
            validate_pdf_dossier_pages(empty, included, readings)
        bad = SimpleNamespace(learning_dossiers=[SimpleNamespace(paper_index=0, supporting_pages=[9])])
        with self.assertRaises(ValueError) as caught:
            validate_pdf_dossier_pages(bad, included, readings)
        self.assertIn("2", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
