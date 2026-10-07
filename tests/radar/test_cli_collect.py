"""Pipeline collection: the whole bounded plan executes; the output bound
applies only to the final ranked slice. Fake source, no network."""

from __future__ import annotations

from tests.radar.discovery_support import planner_model

import json
import datetime as dt
import contextlib
import io
import unittest
from unittest import mock

from radar.pipeline import PipelineRequest, PipelineResult, run


def _raw(wid: str, title: str, year: int = 2026) -> dict:
    return {
        "id": wid,
        "title": title,
        "abstract_inverted_index": {"x": [0]},
        "doi": "",
        "publication_year": year,
        "publication_date": dt.date.today().isoformat(),
        "cited_by_count": 0,
    }


class FakeSource:
    """Serve one distinct work per planned query; record params."""

    def __init__(self):
        self.calls: list[dict] = []
        self.n = 0

    def get_json(self, url, params, headers, timeout):
        self.calls.append(dict(params))
        self.n += 1
        return {
            "results": [
                _raw(f"https://openalex.org/W{self.n}", f"Work {self.n}")
            ]
        }


class TestPipelineCollectionBounds(unittest.TestCase):
    def test_cli_passes_complete_learning_question_to_pipeline(self):
        from radar.cli import main

        question = "  Which assumptions support the mechanism? " + "Explain the boundary conditions. " * 30
        with mock.patch("radar.cli.run", return_value=PipelineResult(0)) as execute, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(["--learning-question", question]), 0)
        self.assertEqual(execute.call_args.args[0].learning_question, question)

    def test_blank_learning_question_is_rejected_before_source_or_storage_io(self):
        for question in ("", " \n\t "):
            for list_reports in (False, True):
                with self.subTest(question=question, list_reports=list_reports):
                    source = FakeSource()
                    with mock.patch("radar.storage.sqlite.SQLiteStore", side_effect=AssertionError("No storage I/O")), mock.patch(
                            "radar.storage.library.ReportLibrary", side_effect=AssertionError("No library I/O")):
                        result = run(PipelineRequest(learning_question=question,
                            source_override=source, storage_dir="unused", list_reports=list_reports))
                    self.assertEqual(result.exit_code, 4)
                    self.assertIn("learning-question", " ".join(result.stderr_notes))
                    self.assertEqual(source.calls, [])

    def test_small_max_still_executes_whole_plan(self):
        fake = FakeSource()
        result = run(PipelineRequest(planner_model_override=planner_model(),
            mode="collect", max_candidates=1, source_override=fake))
        self.assertEqual(result.exit_code, 0)
        self.assertEqual(len(fake.calls), 12)
        self.assertEqual(sum("filter" in p for p in fake.calls), 2)
        self.assertEqual(sum("search.semantic" in p for p in fake.calls), 4)
        self.assertEqual(sum(p.get("sort") == "publication_date:desc" for p in fake.calls), 2)
        # Output bound respected.
        shown = json.loads(result.stdout)
        self.assertEqual(len(shown), 1)


if __name__ == "__main__":
    unittest.main()
