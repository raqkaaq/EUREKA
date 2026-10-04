"""Reopen scientific reports without models, graph processes or storage writes."""

from contextlib import redirect_stdout, redirect_stderr
from contextlib import closing
import io
from pathlib import Path
import tempfile
import sqlite3
import unittest
from unittest import mock

from radar.schema.opportunities import RadarReport
from radar.schema.papers import CollectedWork
from radar.storage.sqlite import SQLiteStore


class TestReportLibrary(unittest.TestCase):
    def seed(self, directory):
        with SQLiteStore(directory) as store:
            first = store.begin_run("analyze", "fixture")
            store.save_pool(first, [CollectedWork(openalex_id="https://openalex.org/W1")])
            store.save_report(first, RadarReport(next_move="Reconstruct the limiting case."), 1)
            store.finish_run(first, 0)
            # A later collection is not a later report.
            last = store.begin_run("collect", "fixture")
            store.save_pool(last, [])
            store.finish_run(last, 0)
        return first

    def test_report_history_and_retrieval_leave_database_bytes_unchanged(self):
        from radar.pipeline import PipelineRequest, run

        with tempfile.TemporaryDirectory() as directory:
            report_id = self.seed(directory)
            path = Path(directory) / "radar.sqlite3"
            before = path.read_bytes()
            with mock.patch("radar.pipeline._run_database", side_effect=AssertionError("writer used")), \
                 mock.patch("radar.pipeline._validate_request", side_effect=AssertionError("model preflight used")):
                history = run(PipelineRequest(storage_dir=directory, list_reports=True))
                latest = run(PipelineRequest(storage_dir=directory, report_id="latest"))
                explicit = run(PipelineRequest(storage_dir=directory, report_id=report_id))
            self.assertEqual(history.exit_code, 0)
            self.assertIn(report_id, history.stdout)
            self.assertIn("legacy", history.stdout.lower())
            self.assertEqual(latest.exit_code, 0)
            self.assertEqual(latest.stdout, explicit.stdout)
            self.assertIn("Reconstruct the limiting case.", latest.stdout)
            self.assertIn(report_id, latest.stdout)
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual({p.name for p in Path(directory).iterdir()}, {"radar.sqlite3"})

    def test_missing_library_does_not_create_directory(self):
        from radar.pipeline import PipelineRequest, run

        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "not-created"
            result = run(PipelineRequest(storage_dir=str(target), list_reports=True))
            self.assertEqual(result.exit_code, 3)
            self.assertFalse(target.exists())

    def test_cli_reopens_report_and_rejects_mutating_mode_combinations(self):
        from radar.cli import main

        with tempfile.TemporaryDirectory() as directory:
            report_id = self.seed(directory)
            output, errors = io.StringIO(), io.StringIO()
            with redirect_stdout(output), redirect_stderr(errors):
                self.assertEqual(main(["--storage-dir", directory, "--report", report_id]), 0)
                self.assertEqual(main(["--storage-dir", directory, "--reports", "--collect-only"]), 4)
            self.assertIn("Reconstruct the limiting case.", output.getvalue())
            self.assertIn("cannot", errors.getvalue())

    def test_corrupt_report_is_redacted_and_preserved(self):
        from radar.pipeline import PipelineRequest, run

        with tempfile.TemporaryDirectory() as directory:
            self.seed(directory)
            path = Path(directory) / "radar.sqlite3"
            with closing(sqlite3.connect(path)) as connection:
                connection.execute("UPDATE reports SET payload=?", ("PRIVATE-CORRUPT-CONTENT",))
                connection.commit()
            before = path.read_bytes()
            result = run(PipelineRequest(storage_dir=directory, report_id="latest"))
            self.assertEqual(result.exit_code, 3)
            self.assertNotIn("PRIVATE", str(result))
            self.assertEqual(path.read_bytes(), before)

    def test_unknown_schema_and_unknown_id_preserve_database(self):
        from radar.pipeline import PipelineRequest, run

        with tempfile.TemporaryDirectory() as directory:
            self.seed(directory)
            path = Path(directory) / "radar.sqlite3"
            before = path.read_bytes()
            result = run(PipelineRequest(storage_dir=directory, report_id="' OR 1=1 --"))
            self.assertEqual(result.exit_code, 3)
            self.assertEqual(path.read_bytes(), before)
            with closing(sqlite3.connect(path)) as connection:
                connection.execute("PRAGMA user_version=99")
            before = path.read_bytes()
            result = run(PipelineRequest(storage_dir=directory, list_reports=True))
            self.assertEqual(result.exit_code, 3)
            self.assertEqual(path.read_bytes(), before)

    def test_history_is_bounded_and_latest_means_latest_report(self):
        from radar.storage.library import ReportLibrary
        from radar.storage.sqlite import StorageError

        with tempfile.TemporaryDirectory() as directory:
            self.seed(directory)
            with SQLiteStore(directory) as store:
                for index in range(22):
                    run_id = store.begin_run("analyze", "fixture")
                    store.save_pool(run_id, [])
                    store.save_report(run_id, RadarReport(next_move=f"Task {index}"), 0)
                    store.finish_run(run_id, 0)
            with ReportLibrary(directory) as library:
                self.assertEqual(len(library.recent()), 20)
                self.assertEqual(library.get().run_id, run_id)
                self.assertEqual(library.recent(limit=1)[0].report.next_move, "Task 21")
                for limit in (True, 0, 101, 1.5):
                    with self.assertRaises(StorageError):
                        library.recent(limit=limit)
