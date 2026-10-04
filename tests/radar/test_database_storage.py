"""Database persistence through the public storage seam; no network or LLM."""

from pathlib import Path
from contextlib import closing
import sqlite3
import tempfile
import unittest
from unittest import mock

from radar.schema.papers import CollectedWork
from radar.storage.sqlite import SQLiteStore, StorageError


class GraphBoundary:
    """Deterministic projection double; native process behavior has its own suite."""

    def __init__(self, directory, **kwargs):
        self.directory = Path(directory)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        (self.directory / 'graph.rdb').write_bytes(b'graph-boundary-double')

    def rebuild(self, papers, reports):
        pass


class TestSQLiteStorage(unittest.TestCase):
    def test_full_pool_survives_reopen_without_json_artifacts(self):
        works = [CollectedWork(openalex_id=f"https://openalex.org/W{i}", title=f"Paper {i}")
                 for i in range(4)]
        with tempfile.TemporaryDirectory() as directory:
            with SQLiteStore(directory) as store:
                run_id = store.begin_run("collect", "OpenAlex")
                store.save_pool(run_id, works)
                store.finish_run(run_id, 0)
            with SQLiteStore(directory) as reopened:
                saved, metadata = reopened.latest_pool()
                self.assertEqual([w.openalex_id for w in saved], [w.openalex_id for w in works])
                self.assertEqual(metadata["coverage"]["collected"], 4)
            self.assertEqual([p.name for p in Path(directory).iterdir()], ["radar.sqlite3"])

    def test_duplicate_pool_is_rejected_without_losing_previous_pool(self):
        paper = CollectedWork(openalex_id="https://openalex.org/W1", title="Original")
        with tempfile.TemporaryDirectory() as directory, SQLiteStore(directory) as store:
            first = store.begin_run("collect", "OpenAlex")
            store.save_pool(first, [paper])
            failed = store.begin_run("collect", "OpenAlex")
            with self.assertRaises(StorageError):
                store.save_pool(failed, [paper, paper])
            store.finish_run(failed, 3, "StorageError")
            self.assertEqual(store.latest_pool()[0], [paper])

    def test_immutable_pool_history_and_score_only_delta(self):
        first_paper = CollectedWork(openalex_id="https://openalex.org/W1", title="Original", score=0.1)
        with tempfile.TemporaryDirectory() as directory, SQLiteStore(directory) as store:
            first = store.begin_run("collect", "OpenAlex")
            store.save_pool(first, [first_paper])
            second = store.begin_run("collect", "OpenAlex")
            same = store.save_pool(second, [first_paper.model_copy(update={"score": 0.9})])
            self.assertEqual(same["delta"]["unchanged_count"], 1)
            third = store.begin_run("collect", "OpenAlex")
            changed = store.save_pool(third, [first_paper.model_copy(update={"title": "Changed"})])
            self.assertEqual(changed["delta"]["changed_count"], 1)
            original = store.connection.execute("SELECT payload FROM pool_papers WHERE run_id=?", (first,)).fetchone()[0]
            self.assertEqual(CollectedWork.model_validate_json(original).title, "Original")

    def test_writer_lock_is_released_and_creates_no_lock_file(self):
        with tempfile.TemporaryDirectory() as directory:
            with SQLiteStore(directory):
                with self.assertRaises(StorageError):
                    with SQLiteStore(directory):
                        pass
            with SQLiteStore(directory):
                pass
            self.assertEqual({p.name for p in Path(directory).iterdir()}, {"radar.sqlite3"})

    def test_delta_compares_latest_pool_not_all_historical_papers(self):
        first = CollectedWork(openalex_id='https://openalex.org/W1', title='First')
        other = CollectedWork(openalex_id='https://openalex.org/W2', title='Other')
        with tempfile.TemporaryDirectory() as directory, SQLiteStore(directory) as store:
            for paper in (first, other):
                store.save_pool(store.begin_run('collect', 'OpenAlex'), [paper])
            returned = store.save_pool(store.begin_run('collect', 'OpenAlex'), [first])
            self.assertEqual(returned['delta']['new_count'], 1)
            self.assertEqual(len(store.projection()[0]), 2)

    def test_unknown_schema_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "radar.sqlite3"
            with closing(sqlite3.connect(path)) as connection:
                connection.execute("PRAGMA user_version=99")
            before = path.read_bytes()
            with self.assertRaises(StorageError):
                with SQLiteStore(directory):
                    pass
            self.assertEqual(path.read_bytes(), before)

    def test_corrupt_stored_payload_is_redacted_and_preserved(self):
        paper = CollectedWork(openalex_id="https://openalex.org/W1", title="Original")
        with tempfile.TemporaryDirectory() as directory, SQLiteStore(directory) as store:
            run_id = store.begin_run("collect", "OpenAlex")
            store.save_pool(run_id, [paper])
            with store.connection:
                store.connection.execute("UPDATE pool_papers SET payload='SECRET-CORRUPT-CONTENT'")
            with self.assertRaises(StorageError) as caught:
                store.latest_pool()
            self.assertNotIn('SECRET', str(caught.exception))
            self.assertEqual(store.connection.execute("SELECT payload FROM pool_papers").fetchone()[0],
                             'SECRET-CORRUPT-CONTENT')

    def test_triage_report_and_failure_are_stored_with_provenance(self):
        from radar.processing.evidence import attach_evidence
        from radar.schema.opportunities import OpportunityDraft, RadarDraft
        from radar.schema.triage import TriageBatch, TriageResult

        paper = CollectedWork(openalex_id="https://openalex.org/W1", title="Paper")
        batch = TriageBatch(model_id="test-model", rubric_version="test-rubric", results=[
            TriageResult(work_id=paper.openalex_id, status="missing_abstract")])
        report = attach_evidence(RadarDraft(opportunities=[OpportunityDraft(title="Hypothesis", evidence=[0])]), [paper])
        with tempfile.TemporaryDirectory() as directory:
            with SQLiteStore(directory) as store:
                run_id = store.begin_run("analyze", "OpenAlex")
                store.save_pool(run_id, [paper])
                store.save_triage(run_id, batch)
                store.save_report(run_id, report, 1)
                store.finish_run(run_id, 3, "FalkorError")
            with SQLiteStore(directory) as store:
                self.assertEqual(store.projection(), ([paper], [(run_id, report)]))
                saved = store.connection.execute("SELECT payload FROM screening").fetchone()[0]
                self.assertEqual(TriageBatch.model_validate_json(saved), batch)
                self.assertEqual(store.connection.execute("SELECT exit_code,failure_category FROM runs").fetchone(),
                                 (3, "FalkorError"))
                self.assertFalse(store.graph_state()["ready"])


class TestDatabasePipeline(unittest.TestCase):
    def test_independent_screening_success_and_safe_failure_are_saved_before_refusal(self):
        from radar.pipeline import PipelineRequest, run
        from radar.schema.triage import TriageBatch
        from test_qwen_triage import works, answers_model, synth_model

        pool, routing, synthesis = works(4), [], []

        def first_batch_fails(rows):
            if rows[0]["work_id"] == pool[0].openalex_id:
                raise RuntimeError("PRIVATE-FAILURE")
            return rows

        with tempfile.TemporaryDirectory() as directory:
            with SQLiteStore(directory) as store:
                initial = store.begin_run("collect", "fixture")
                store.save_pool(initial, pool)
                store.finish_run(initial, 0)
            result = run(PipelineRequest(mode="analyze", from_database=True,
                storage_dir=directory, triage_model_override=answers_model(routing, first_batch_fails),
                model_override=synth_model(synthesis)))
            self.assertEqual(result.exit_code, 3)
            self.assertEqual(synthesis, [])
            self.assertIn("unexpected_error", str(result.stderr_notes))
            self.assertNotIn("PRIVATE", str(result.stderr_notes))
            with SQLiteStore(directory) as store:
                payload = store.connection.execute("SELECT payload FROM screening").fetchone()[0]
                batch = TriageBatch.model_validate_json(payload)
                self.assertEqual([r.status for r in batch.results], ["failed", "failed", "scored", "scored"])
                self.assertEqual(len(store.latest_pool()[0]), 4)
                self.assertNotIn("PRIVATE", payload)
                self.assertEqual(store.connection.execute("SELECT COUNT(*) FROM reports").fetchone()[0], 0)
            self.assertEqual({p.name for p in Path(directory).iterdir()}, {"radar.sqlite3"})

    def test_collection_persists_full_pool_despite_output_bound(self):
        import json
        from radar.pipeline import PipelineRequest, run
        from test_cli_collect import FakeSource

        with tempfile.TemporaryDirectory() as directory, mock.patch(
                'radar.storage.graph_projection.FalkorGraph', GraphBoundary):
            result = run(PipelineRequest(mode='collect', storage_dir=directory,
                                         max_candidates=1, source_override=FakeSource()))
            self.assertEqual(result.exit_code, 0, result.stderr_notes)
            self.assertEqual(len(json.loads(result.stdout)['works']), 1)
            with SQLiteStore(directory) as store:
                self.assertEqual(len(store.latest_pool()[0]), 12)
            self.assertEqual({p.name for p in Path(directory).iterdir()}, {'radar.sqlite3', 'graph.rdb'})

    def test_graph_failure_preserves_pool_and_explicit_rebuild_repairs(self):
        from radar.pipeline import PipelineRequest, run
        from radar.storage.falkor import FalkorError
        from test_cli_collect import FakeSource

        with tempfile.TemporaryDirectory() as directory:
            with mock.patch('radar.storage.graph_projection.FalkorGraph', side_effect=FalkorError('SECRET')):
                result = run(PipelineRequest(mode='collect', storage_dir=directory, source_override=FakeSource()))
            self.assertEqual(result.exit_code, 3)
            self.assertIn('--rebuild-graph', result.stderr_notes[0])
            self.assertNotIn('SECRET', result.stderr_notes[0])
            with SQLiteStore(directory) as store:
                self.assertEqual(len(store.latest_pool()[0]), 12)
                self.assertFalse(store.graph_state()['ready'])
                self.assertEqual(store.connection.execute('SELECT exit_code FROM runs').fetchone()[0], 3)
            with mock.patch('radar.storage.graph_projection.FalkorGraph', GraphBoundary), mock.patch(
                    'radar.pipeline._collect_live', side_effect=AssertionError('no discovery')):
                repaired = run(PipelineRequest(storage_dir=directory, rebuild_graph=True))
            self.assertEqual(repaired.exit_code, 0, repaired.stderr_notes)
            with SQLiteStore(directory) as store:
                self.assertTrue(store.graph_state()['ready'])
                self.assertEqual(store.connection.execute('SELECT COUNT(*) FROM pools').fetchone()[0], 1)

    def test_cached_pool_uses_no_discovery_or_legacy_json_writers(self):
        from radar.pipeline import PipelineRequest, run
        from test_cli_collect import FakeSource

        with tempfile.TemporaryDirectory() as directory, mock.patch(
                'radar.storage.graph_projection.FalkorGraph', GraphBoundary), mock.patch(
                'radar.storage.snapshots.refresh_pool', side_effect=AssertionError('no JSON')), mock.patch(
                'radar.storage.triage.write_sidecar', side_effect=AssertionError('no JSON')):
            initial = run(PipelineRequest(mode='collect', storage_dir=directory, source_override=FakeSource()))
            self.assertEqual(initial.exit_code, 0)
            with mock.patch('radar.pipeline._collect_live', side_effect=AssertionError('no discovery')):
                cached = run(PipelineRequest(mode='collect', storage_dir=directory, from_database=True))
            self.assertEqual(cached.exit_code, 0, cached.stderr_notes)
            self.assertEqual({p.name for p in Path(directory).iterdir()}, {'radar.sqlite3', 'graph.rdb'})

    def test_legacy_import_preserves_original_bytes_and_all_records(self):
        from radar.pipeline import PipelineRequest, run
        from radar.storage.snapshots import refresh_pool

        paper = CollectedWork(openalex_id='https://openalex.org/W1', title='Legacy')
        with tempfile.TemporaryDirectory() as directory:
            legacy = Path(directory) / 'legacy'
            refresh_pool([paper], str(legacy))
            snapshot = legacy / 'snapshot.json'
            original = snapshot.read_bytes()
            with mock.patch('radar.storage.graph_projection.FalkorGraph', GraphBoundary), mock.patch(
                    'radar.pipeline._collect_live', side_effect=AssertionError('no discovery')):
                result = run(PipelineRequest(mode='collect', storage_dir=str(Path(directory) / 'db'),
                                             from_snapshot=str(snapshot)))
            self.assertEqual(result.exit_code, 0, result.stderr_notes)
            self.assertEqual(snapshot.read_bytes(), original)
            with SQLiteStore(Path(directory) / 'db') as store:
                self.assertEqual(store.latest_pool()[0], [paper])

    def test_invalid_mode_combination_does_not_create_storage(self):
        from radar.pipeline import PipelineRequest, run

        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'not-created'
            result = run(PipelineRequest(storage_dir=str(target), from_database=True, rebuild_graph=True))
            self.assertEqual(result.exit_code, 4)
            self.assertFalse(target.exists())

    def test_analysis_persists_real_typed_agent_report_and_failed_run_screening(self):
        from pydantic_ai.messages import ModelResponse, ToolCallPart
        from pydantic_ai.models.function import FunctionModel
        from radar.pipeline import PipelineRequest, run
        from radar.schema.opportunities import OpportunityDraft, RadarDraft
        from radar.schema.triage import TriageBatch, TriageResult
        from test_cli_collect import FakeSource

        def score(pool, profile):
            return TriageBatch(model_id='offline-test', rubric_version='test', results=[
                TriageResult(work_id=w.openalex_id, status='scored', ai_ml_relevance=0.8,
                             cross_domain_potential=0.7) for w in pool])

        def respond(messages, info):
            draft = RadarDraft(opportunities=[OpportunityDraft(title='Hypothesis', evidence=[0])])
            return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, draft.model_dump())])

        def fail(messages, info):
            raise RuntimeError('SECRET-UPSTREAM')

        with tempfile.TemporaryDirectory() as directory, mock.patch(
                'radar.storage.graph_projection.FalkorGraph', GraphBoundary):
            from pdf_support import empty_notes_model, make_loader, make_pdf_document
            from radar.source.openalex import DictTransport as _DT
            from radar.config.searches import build_query_plan as _plan
            from radar.config.interests import default_profile as _prof
            # FakeSource yields W1..W12; pre-build matching PDF doubles.
            _docs = {f"https://openalex.org/W{i}": make_pdf_document(
                f"https://openalex.org/W{i}") for i in range(1, 13)}
            successful = run(PipelineRequest(mode='analyze', storage_dir=directory, max_candidates=1,
                                             source_override=FakeSource(), triage_scorer=score,
                                             model_override=FunctionModel(respond),
                                             document_loader=make_loader(_docs),
                                             document_model_override=empty_notes_model()))
            self.assertEqual(successful.exit_code, 0, successful.stderr_notes)
            failed = run(PipelineRequest(mode='analyze', storage_dir=directory, from_database=True,
                                         triage_scorer=score, model_override=FunctionModel(fail),
                                         document_loader=make_loader(_docs),
                                         document_model_override=empty_notes_model()))
            self.assertEqual(failed.exit_code, 3)
            self.assertNotIn('SECRET', str(failed.stderr_notes))
            with SQLiteStore(directory) as store:
                self.assertEqual(len(store.projection()[0]), 12)
                reports = store.projection()[1]
                self.assertEqual(len(reports), 1)
                self.assertEqual(reports[0][1].opportunities[0].draft.title, 'Hypothesis')
                self.assertEqual(store.connection.execute('SELECT COUNT(*) FROM screening').fetchone()[0], 2)
                self.assertEqual(store.connection.execute('SELECT exit_code FROM runs ORDER BY rowid').fetchall(),
                                 [(0,), (3,)])
                self.assertFalse(store.graph_state()['ready'])


if __name__ == "__main__":
    unittest.main()
