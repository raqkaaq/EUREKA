"""Immutable full-text provenance in the authoritative database."""

import tempfile
import unittest
from unittest import mock

from radar.schema.documents import (
    PDFDocument, PDFPage, PDFReading, ChunkReading, DocumentNotes, PageEvidence, DocumentFailure,
)
from radar.schema.papers import CollectedWork
from radar.storage.sqlite import SQLiteStore, StorageError


def document(work_id="https://openalex.org/W1"):
    return PDFDocument(work_id=work_id, source_url="https://papers.example/paper.pdf",
                       sha256="1" * 64,
                       relative_path=f"pdf/{work_id.rsplit('/', 1)[-1]}-{'1' * 64}.pdf",
                       pages=[PDFPage(number=1, text="The controlled comparison found no improvement.")])


def reading(doc):
    from radar.processing.document_chunks import chunk_document
    text, spans = chunk_document(doc)
    notes = DocumentNotes(summary="A negative finding.", methods="Controlled comparison.",
                          results="No improvement.", limitations="Sample size not verified.",
                          evidence=[PageEvidence(page=1, quote="found no improvement", finding="Negative result.")])
    return PDFReading(**doc.model_dump(include={"work_id", "source_url", "sha256", "relative_path"}),
                      page_count=1, text_chars=len(text),
                      chunks=[ChunkReading(index=i, start=start, end=end, pages=pages, notes=notes)
                              for i, (start, end, pages) in enumerate(spans)],
                      notes=notes, extraction_warning=doc.extraction_warning)


def passage_report(doc, passage):
    from radar.schema.documents import PDFSource
    from radar.schema.learning import LearningDossierDraft
    from radar.schema.opportunities import EvidenceLink, RadarReport, ResolvedLearningDossier
    from test_learning_dossier import _dossier_dict

    return RadarReport(
        learning_question="Which comparison supports the negative result?",
        document_readings=[reading(doc)],
        learning_dossiers=[ResolvedLearningDossier(
            dossier=LearningDossierDraft.model_validate(_dossier_dict(
                supporting_pages=[passage.page], source_passage_ids=[passage.passage_id])),
            source=EvidenceLink(index=0, openalex_id=doc.work_id),
            evidence_level="pdf_text",
            source_pdf=PDFSource.model_validate(doc.model_dump(include=set(PDFSource.model_fields))),
            source_pages=[passage.page], source_passages=[passage])])


class TestDocumentStorage(unittest.TestCase):
    def test_pipeline_delivers_learning_question_and_retains_consulted_passage(self):
        from pydantic_ai.messages import ModelResponse, ToolCallPart, ToolReturnPart, UserPromptPart
        from pydantic_ai.models.function import FunctionModel
        from radar.pipeline import PipelineRequest, run
        from pdf_support import make_loader, empty_notes_model
        from test_analyze import _all_scored_scorer
        from test_database_storage import GraphBoundary
        from test_learning_dossier import _dossier_dict

        doc = document()
        question = "  Which premise supports the comparison? " + "Check the boundary conditions. " * 30
        final_prompts = []

        def respond(messages, info):
            if not info.function_tools:
                return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, {"next_move": "Study it."})])
            final_prompts.extend(part.content for message in messages for part in message.parts
                                 if isinstance(part, UserPromptPart))
            returns = [part.content for message in messages for part in message.parts
                       if isinstance(part, ToolReturnPart) and part.tool_name == "read_pdf_passage"]
            if not returns:
                return ModelResponse(parts=[ToolCallPart("read_pdf_passage",
                    {"paper_index": 0, "page": 1, "offset": 0})])
            return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name,
                {"learning_dossiers": [_dossier_dict(supporting_pages=[1],
                    source_passage_ids=[returns[-1]["passage_id"]])], "next_move": "Inspect this premise."})])

        with tempfile.TemporaryDirectory() as directory, mock.patch(
                "radar.storage.graph_projection.FalkorGraph", GraphBoundary):
            with SQLiteStore(directory) as store:
                self.seed(store)
            result = run(PipelineRequest(mode="analyze", from_database=True, storage_dir=directory,
                learning_question=question, triage_scorer=_all_scored_scorer,
                document_loader=make_loader({doc.work_id: doc}), document_model_override=empty_notes_model(),
                model_override=FunctionModel(respond)))
            self.assertEqual(result.exit_code, 0, result.stderr_notes)
            self.assertTrue(final_prompts)
            self.assertTrue(all(question in prompt for prompt in final_prompts))
            self.assertIn(question, result.stdout)
            self.assertIn(doc.pages[0].text, result.stdout)
            with SQLiteStore(directory) as store:
                saved = store.projection()[1][0][1]
                self.assertEqual(saved.learning_question, question)
                passage = saved.learning_dossiers[0].source_passages[0]
                self.assertEqual(passage.text, doc.pages[0].text)
                self.assertEqual(passage.work_id, doc.work_id)
                self.assertEqual(saved.learning_dossiers[0].dossier.source_passage_ids, [passage.passage_id])

    def test_report_rejects_forged_passage_text_even_with_verified_source_identity(self):
        from radar.processing.pdf_passages import PDFPassages

        doc = document()
        passage = PDFPassages([doc]).read(0, 1)
        forged = passage.model_copy(update={"text": "X" * len(passage.text)})
        report = passage_report(doc, forged)
        with tempfile.TemporaryDirectory() as directory, SQLiteStore(directory) as store:
            run_id = self.seed(store)
            store.save_document(run_id, doc)
            store.save_document_reading(run_id, reading(doc))
            with self.assertRaises(StorageError):
                store.save_report(run_id, report, 1)
            self.assertEqual(store.projection()[1], [])

    def test_passages_and_question_survive_reopening_using_this_runs_pdf_hash(self):
        from radar.processing.pdf_passages import PDFPassages

        doc = document().model_copy(update={"pages": [PDFPage(
            number=1, text="  αβ controlled comparison found no improvement.\nFinal line.")]})
        passage = PDFPassages([doc]).read(0, 1, 2)
        report = passage_report(doc, passage).model_copy(update={
            "learning_question": "Which premise is necessary? " * 40})
        newer = doc.model_copy(update={"sha256": "2" * 64,
            "relative_path": "pdf/W1-" + "2" * 64 + ".pdf",
            "pages": [PDFPage(number=1, text="A later extraction, from different PDF bytes.")]})
        with tempfile.TemporaryDirectory() as directory:
            with SQLiteStore(directory) as store:
                run_id = self.seed(store)
                store.save_document(run_id, doc)
                store.save_document_reading(run_id, reading(doc))
                newer_run = self.seed(store)
                store.save_document(newer_run, newer)
                self.assertEqual(store.cached_document(doc.work_id), newer)
                store.save_report(run_id, report, 1)
            with SQLiteStore(directory) as store:
                saved = store.projection()[1][0][1]
                self.assertEqual(saved, report)
                self.assertEqual(saved.learning_question, report.learning_question)
                self.assertEqual(saved.learning_dossiers[0].source_passages[0].text,
                                 "αβ controlled comparison found no improvement.\nFinal line.")

    def test_every_cited_passage_must_be_inside_the_stored_page(self):
        from radar.processing.pdf_passages import PDFPassages

        doc = document()
        passage = PDFPassages([doc]).read(0, 1)
        start = len(doc.pages[0].text) + 1
        end = start + len(passage.text)
        outside = passage.model_copy(update={"start": start, "end": end,
            "passage_id": f"W1:{doc.sha256}:p1:{start}-{end}"})
        report = passage_report(doc, passage)
        resolved = report.learning_dossiers[0]
        dossier = resolved.dossier.model_copy(update={
            "source_passage_ids": [passage.passage_id, outside.passage_id]})
        report = report.model_copy(update={"learning_dossiers": [resolved.model_copy(update={
            "dossier": dossier, "source_passages": [passage, outside]})]})
        with tempfile.TemporaryDirectory() as directory, SQLiteStore(directory) as store:
            run_id = self.seed(store)
            store.save_document(run_id, doc)
            store.save_document_reading(run_id, reading(doc))
            with self.assertRaises(StorageError):
                store.save_report(run_id, report, 1)
            self.assertEqual(store.projection()[1], [])

    def test_detailed_reading_survives_storage_and_discovery_memory_without_truncation(self):
        doc = document()
        original = reading(doc)
        summary = "Mechanism, assumptions, results and counterevidence. " * 25
        notes = original.notes.model_copy(update={"summary": summary})
        detailed = original.model_copy(update={
            "notes": notes,
            "chunks": [chunk.model_copy(update={"notes": notes}) for chunk in original.chunks]})
        with tempfile.TemporaryDirectory() as directory:
            with SQLiteStore(directory) as store:
                run_id = self.seed(store)
                store.save_document(run_id, doc)
                store.save_document_reading(run_id, detailed)
            with SQLiteStore(directory) as store:
                self.assertEqual(store.document_records(run_id)[0].reading, detailed)
                self.assertEqual(store.discovery_memory().findings[0].contribution, summary.strip())

    def test_verified_reextraction_repairs_same_hash_cache_without_rewriting_runs(self):
        with tempfile.TemporaryDirectory() as directory, SQLiteStore(directory) as store:
            old_run = self.seed(store)
            doc = document()
            stale = doc.model_copy(update={"pages": [PDFPage(number=1, text="Corrupted cached text.")]})
            store.save_document(old_run, stale)
            old_records = store.document_records(old_run)
            new_run = self.seed(store)
            store.save_document(new_run, doc)
            store.save_document_reading(new_run, reading(doc))
            self.assertEqual(store.cached_document(doc.work_id), doc)
            self.assertEqual(store.document_records(old_run), old_records)
            self.assertEqual(store.document_records(new_run)[0].status, "read")

    def test_reading_survives_a_later_specialist_failure(self):
        from pydantic_ai.models.function import FunctionModel
        from radar.pipeline import PipelineRequest, run
        from pdf_support import make_loader, empty_notes_model
        from test_analyze import _all_scored_scorer

        def failed_role(messages, info):
            raise ConnectionRefusedError("fixture specialist failure")

        with tempfile.TemporaryDirectory() as directory:
            with SQLiteStore(directory) as store:
                self.seed(store)
            result = run(PipelineRequest(
                mode="analyze", from_database=True, storage_dir=directory,
                triage_scorer=_all_scored_scorer,
                document_loader=make_loader({document().work_id: document()}),
                document_model_override=empty_notes_model(),
                model_override=FunctionModel(failed_role)))
            self.assertEqual(result.exit_code, 3)
            self.assertEqual(result.stdout, "")
            with SQLiteStore(directory) as store:
                _, metadata = store.latest_pool()
                records = store.document_records(metadata["run_id"])
                self.assertEqual(records[0].status, "read")
                self.assertIsNotNone(records[0].reading)
                self.assertEqual(store.projection()[1], [])

    def test_self_consistent_but_incomplete_reading_never_becomes_read(self):
        with tempfile.TemporaryDirectory() as directory, SQLiteStore(directory) as store:
            run_id = self.seed(store)
            doc = document()
            store.save_document(run_id, doc)
            good = reading(doc)
            forged = good.model_copy(update={
                "text_chars": 1,
                "chunks": [good.chunks[0].model_copy(update={"end": 1})]})
            with self.assertRaises(StorageError):
                store.save_document_reading(run_id, forged)
            self.assertEqual(store.document_records(run_id)[0].status, "extracted")

    def seed(self, store):
        run_id = store.begin_run("analyze", "fixture")
        store.save_pool(run_id, [CollectedWork(openalex_id="https://openalex.org/W1", abstract="Screening only.")])
        return run_id

    def test_extraction_and_reading_are_distinct_and_survive_reopening(self):
        doc = document()
        with tempfile.TemporaryDirectory() as directory:
            with SQLiteStore(directory) as store:
                run_id = self.seed(store)
                store.save_document(run_id, doc)
                self.assertEqual(store.document_records(run_id)[0].status, "extracted")
                store.save_document_reading(run_id, reading(doc))
                self.assertEqual(store.document_records(run_id)[0].status, "read")
            with SQLiteStore(directory) as store:
                self.assertEqual(store.cached_document(doc.work_id), doc)
                self.assertEqual(store.document_records(run_id)[0].reading, reading(doc))

    def test_unrelated_source_and_invented_quote_never_become_read(self):
        with tempfile.TemporaryDirectory() as directory, SQLiteStore(directory) as store:
            run_id = self.seed(store)
            with self.assertRaises(StorageError):
                store.save_document(run_id, document("https://openalex.org/W2"))
            doc = document()
            store.save_document(run_id, doc)
            good = reading(doc)
            bad_notes = good.notes.model_copy(update={"evidence": [PageEvidence(page=1, quote="invented finding", finding="Bad.")]})
            with self.assertRaises(StorageError):
                store.save_document_reading(run_id, good.model_copy(update={"notes": bad_notes}))
            self.assertEqual(store.document_records(run_id)[0].status, "extracted")

    def test_inaccessible_pdf_is_recorded_not_classified_as_unimportant(self):
        with tempfile.TemporaryDirectory() as directory, SQLiteStore(directory) as store:
            run_id = self.seed(store)
            failure = DocumentFailure(work_id="https://openalex.org/W1", category="missing_pdf_url")
            store.save_document_failure(run_id, failure)
            self.assertEqual(store.document_records(run_id)[0].failure, failure)
            self.assertEqual(store.document_records(run_id)[0].status, "failed")

    def test_report_gets_pdf_basis_only_with_this_runs_verified_reading(self):
        from radar.processing.evidence import attach_evidence
        from radar.schema.opportunities import LearningRadarDraft
        from test_learning_dossier import _dossier_dict
        from radar.output.markdown import render_markdown

        doc = document()
        result = reading(doc)
        work = CollectedWork(openalex_id=doc.work_id)
        draft = LearningRadarDraft.model_validate({"learning_dossiers": [_dossier_dict(supporting_pages=[1])]})
        report = attach_evidence(draft, [work], document_readings=[result])
        self.assertEqual(report.learning_dossiers[0].evidence_level, "pdf_text")
        self.assertEqual(report.learning_dossiers[0].source_pdf.sha256, doc.sha256)
        self.assertIn("#page=1", render_markdown(report))
        no_quotes = result.model_copy(update={"notes": result.notes.model_copy(update={"evidence": []})})
        with self.assertRaises(ValueError):
            attach_evidence(draft, [work], document_readings=[no_quotes])
        with tempfile.TemporaryDirectory() as directory, SQLiteStore(directory) as store:
            run_id = self.seed(store)
            with self.assertRaises(StorageError):
                store.save_report(run_id, report, 1)
            store.save_document(run_id, doc)
            store.save_document_reading(run_id, result)
            store.save_report(run_id, report, 1)
            self.assertEqual(store.projection()[1][0][1], report)
