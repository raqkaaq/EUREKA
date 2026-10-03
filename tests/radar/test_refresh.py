"""Refresh-dir snapshot mode: full-pool persistence, deltas, locks, failures.

Public seam: ``radar.refresh.refresh_pool`` (+ ``compute_delta``,
``update_llm_coverage``). Real temp filesystem; no network, no LLM.
"""

from __future__ import annotations

import json
import unittest

from radar import refresh as _refresh
from radar.cli import collect_candidates as _collect_candidates  # noqa: F401
from radar.models import CollectedWork, LocationInfo


def _work(
    wid: str,
    title: str = "Some title",
    abstract: str = "abstract text here",
    year: int = 2026,
    cited: int = 3,
    n_locations: int = 1,
) -> CollectedWork:
    locations = [
        LocationInfo(
            source_name=f"Source {i}",
            landing_url=f"https://example.org/{wid}/{i}",
            pdf_url="",
            is_oa=(i == 0),
        )
        for i in range(n_locations)
    ]
    return CollectedWork(
        openalex_id=wid,
        title=title,
        abstract=abstract,
        publication_year=year,
        doi="",
        primary_url=locations[0].landing_url if locations else wid,
        locations=locations,
        cited_by_count=cited,
        matched_queries=["machine learning"],
        query_kinds=["semantic"],
        score=0.1,
    )


def _pool(n: int, **kwargs) -> list[CollectedWork]:
    return [_work(f"https://openalex.org/W{i}", title=f"Work {i}", **kwargs) for i in range(n)]


def _read_snapshot(tmp: str) -> dict:
    with open(_refresh.snapshot_path(tmp), encoding="utf-8") as fh:
        return json.load(fh)


class TestRefreshSavesFullPool(unittest.TestCase):
    def test_pool_beyond_topn_is_saved(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            works = _pool(8)
            summary = _refresh.refresh_pool(works, tmp)
            snap = _read_snapshot(tmp)
            # All 8 pool works persisted, not just a topN slice.
            self.assertEqual(len(snap["works"]), 8)
            self.assertEqual(summary["coverage"]["collected"], 8)
            self.assertEqual(summary["coverage"]["llm_selected"], 0)
            self.assertEqual(summary["coverage"]["llm_analyzed"], 0)

    def test_locations_and_provenance_retained(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            works = _pool(2, n_locations=3)
            _refresh.refresh_pool(works, tmp)
            snap = _read_snapshot(tmp)
            for entry in snap["works"]:
                self.assertEqual(len(entry["locations"]), 3)
                self.assertTrue(entry["primary_url"])
                self.assertTrue(entry["matched_queries"])
                self.assertTrue(entry["query_kinds"])
                self.assertTrue(entry["openalex_id"])

    def test_missing_abstracts_retained_and_counted(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            works = [_work("https://openalex.org/W1", abstract="has abstract"),
                     _work("https://openalex.org/W2", abstract="")]
            summary = _refresh.refresh_pool(works, tmp)
            snap = _read_snapshot(tmp)
            self.assertEqual(len(snap["works"]), 2)
            ids = {w["openalex_id"] for w in snap["works"]}
            self.assertIn("https://openalex.org/W2", ids)
            self.assertEqual(summary["coverage"]["with_abstracts"], 1)
            self.assertEqual(summary["coverage"]["missing_abstracts"], 1)


class TestRefreshDelta(unittest.TestCase):
    def test_rerun_is_idempotent_regardless_of_order(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            works = _pool(4)
            first = _refresh.refresh_pool(works, tmp)
            self.assertEqual(first["delta"]["new_count"], 4)
            second = _refresh.refresh_pool(list(reversed(works)), tmp)
            self.assertEqual(second["delta"]["new_count"], 0)
            self.assertEqual(second["delta"]["changed_count"], 0)
            self.assertEqual(second["delta"]["unchanged_count"], 4)

    def test_added_and_changed_ids_reported(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            _refresh.refresh_pool(_pool(3), tmp)
            grown = _pool(3)
            grown.append(_work("https://openalex.org/W99", title="Brand new"))
            grown[0] = _work("https://openalex.org/W0", title="Retitled work")
            summary = _refresh.refresh_pool(grown, tmp)
            self.assertEqual(summary["delta"]["new_count"], 1)
            self.assertIn("https://openalex.org/W99", summary["delta"]["new_ids"])
            self.assertEqual(summary["delta"]["changed_count"], 1)
            self.assertIn("https://openalex.org/W0", summary["delta"]["changed_ids"])
            self.assertEqual(summary["delta"]["unchanged_count"], 2)

    def test_snapshot_carries_timestamp_and_disclosure(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            summary = _refresh.refresh_pool(_pool(1), tmp)
            snap = _read_snapshot(tmp)
            self.assertTrue(snap["collected_at_utc"])
            self.assertIn("OpenAlex", snap["discovery"]["source"])
            self.assertTrue(snap["discovery"]["bounded"])
            self.assertIn("not all of OpenAlex", summary["disclosure"])


class TestRefreshFailures(unittest.TestCase):
    def test_invalid_previous_snapshot_preserved_and_errors(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            _refresh.refresh_pool(_pool(2), tmp)
            path = _refresh.snapshot_path(tmp)
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("{not valid json")
            with self.assertRaises(_refresh.RefreshError):
                _refresh.refresh_pool(_pool(2), tmp)
            # Corrupt file left untouched for forensics, never deleted.
            with open(path, encoding="utf-8") as fh:
                self.assertEqual(fh.read(), "{not valid json")

    def test_persistence_failure_preserves_last_valid(self):
        import tempfile
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as tmp:
            _refresh.refresh_pool(_pool(2), tmp)
            before = _read_snapshot(tmp)
            with patch("os.replace", side_effect=OSError("disk gone")):
                with self.assertRaises(_refresh.RefreshError):
                    _refresh.refresh_pool(_pool(3), tmp)
            self.assertEqual(_read_snapshot(tmp), before)

    def test_lock_contention_errors_without_touching_snapshot(self):
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            _refresh.refresh_pool(_pool(2), tmp)
            before = _read_snapshot(tmp)
            # Simulate another active refresh holding the lock.
            lock = os.path.join(tmp, _refresh.LOCK_FILENAME)
            with open(lock, "w", encoding="utf-8") as fh:
                fh.write("other-pid")
            with self.assertRaises(_refresh.RefreshError):
                _refresh.refresh_pool(_pool(2), tmp)
            self.assertEqual(_read_snapshot(tmp), before)
            # Foreign lock is not ours: leave it in place.
            self.assertTrue(os.path.exists(lock))

    def test_lock_released_even_on_error(self):
        import os
        import tempfile
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as tmp:
            lock = os.path.join(tmp, _refresh.LOCK_FILENAME)
            with patch("os.replace", side_effect=OSError("disk gone")):
                with self.assertRaises(_refresh.RefreshError):
                    _refresh.refresh_pool(_pool(1), tmp)
            self.assertFalse(os.path.exists(lock))

    def test_collection_failure_leaves_snapshot_untouched(self):
        import tempfile
        from unittest.mock import patch

        from radar import cli as _cli

        with tempfile.TemporaryDirectory() as tmp:
            _refresh.refresh_pool(_pool(2), tmp)
            before = _read_snapshot(tmp)
            with patch.object(_cli, "collect_pool", side_effect=RuntimeError("net down")):
                code = _cli.main(["--collect-only", "--refresh-dir", tmp])
            self.assertNotEqual(code, 0)
            self.assertEqual(_read_snapshot(tmp), before)

    def test_update_llm_coverage_patches_counts(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            _refresh.refresh_pool(_pool(5), tmp)
            updated = _refresh.update_llm_coverage(tmp, llm_selected=5, llm_analyzed=2)
            self.assertEqual(updated["coverage"]["llm_selected"], 5)
            self.assertEqual(updated["coverage"]["llm_analyzed"], 2)
            self.assertEqual(updated["coverage"]["collected"], 5)


class TestStrictPreviousSnapshot(unittest.TestCase):
    """Only producer-v1 snapshots are a valid delta baseline.

    Unknown/missing schema, non-list works, non-dict entries, missing or
    invalid or duplicate OpenAlex IDs, and entries that fail CollectedWork
    validation must raise RefreshError, leave the original bytes untouched,
    and release the lock. No silent self-healing.
    """

    def _seed_raw(self, tmp: str, payload: object) -> str:
        import json as _json

        path = _refresh.snapshot_path(tmp)
        with open(path, "w", encoding="utf-8") as fh:
            _json.dump(payload, fh)
        with open(path, encoding="utf-8") as fh:
            return fh.read()

    def _assert_refused(self, tmp: str, before: str) -> None:
        import os as _os

        path = _refresh.snapshot_path(tmp)
        with self.assertRaises(_refresh.RefreshError):
            _refresh.refresh_pool(_pool(2), tmp)
        with open(path, encoding="utf-8") as fh:
            self.assertEqual(fh.read(), before)
        self.assertFalse(_os.path.exists(_os.path.join(tmp, _refresh.LOCK_FILENAME)))

    def test_unknown_schema_version_refused(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            before = self._seed_raw(tmp, {"schema_version": 999, "works": []})
            self._assert_refused(tmp, before)

    def test_missing_schema_version_refused(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            before = self._seed_raw(tmp, {"works": []})
            self._assert_refused(tmp, before)

    def test_non_list_works_refused(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            before = self._seed_raw(
                tmp, {"schema_version": 1, "works": "notalist"}
            )
            self._assert_refused(tmp, before)

    def test_non_dict_entry_refused(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            before = self._seed_raw(tmp, {"schema_version": 1, "works": [42]})
            self._assert_refused(tmp, before)

    def test_entry_missing_openalex_id_refused(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            before = self._seed_raw(
                tmp, {"schema_version": 1, "works": [{"title": "broken/no ID"}]}
            )
            self._assert_refused(tmp, before)

    def test_entry_invalid_collected_work_refused(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            before = self._seed_raw(
                tmp,
                {
                    "schema_version": 1,
                    "works": [
                        {
                            "openalex_id": "https://openalex.org/W1",
                            "title": "bad counts",
                            "cited_by_count": -5,
                        }
                    ],
                },
            )
            self._assert_refused(tmp, before)

    def test_duplicate_openalex_ids_refused(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            entry = _work("https://openalex.org/W1").model_dump()
            before = self._seed_raw(
                tmp, {"schema_version": 1, "works": [entry, entry]}
            )
            self._assert_refused(tmp, before)

    def test_writer_snapshot_with_missing_abstract_stays_readable(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            works = [
                _work("https://openalex.org/W1", abstract="has abstract"),
                _work("https://openalex.org/W2", abstract=""),
            ]
            _refresh.refresh_pool(works, tmp)
            summary = _refresh.refresh_pool(works, tmp)
            self.assertEqual(summary["delta"]["unchanged_count"], 2)
            self.assertEqual(summary["delta"]["changed_count"], 0)


class TestDeltaIgnoresDerivedScore(unittest.TestCase):
    """Score is a derived ranking signal, not work identity.

    A score-only change must not report a changed work; real metadata,
    abstract, or provenance changes still must.
    """

    def test_score_only_change_is_unchanged(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            _refresh.refresh_pool([_work("https://openalex.org/W1")], tmp)
            rescored = _work("https://openalex.org/W1")
            rescored.score = 0.99
            summary = _refresh.refresh_pool([rescored], tmp)
            self.assertEqual(summary["delta"]["changed_count"], 0)
            self.assertEqual(summary["delta"]["unchanged_count"], 1)

    def test_metadata_change_still_detected(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            _refresh.refresh_pool([_work("https://openalex.org/W1")], tmp)
            retitled = _work("https://openalex.org/W1", title="New title")
            retitled.score = 0.99
            summary = _refresh.refresh_pool([retitled], tmp)
            self.assertEqual(summary["delta"]["changed_count"], 1)
            self.assertIn(
                "https://openalex.org/W1", summary["delta"]["changed_ids"]
            )

    def test_snapshot_bounds_match_collector_constants(self):
        import tempfile

        from radar.models import MAX_QUERIES
        from radar.openalex import MAX_TOTAL_WORKS

        with tempfile.TemporaryDirectory() as tmp:
            _refresh.refresh_pool(_pool(1), tmp)
            snap = _read_snapshot(tmp)
            self.assertEqual(snap["discovery"]["max_requests"], MAX_QUERIES)
            self.assertEqual(snap["discovery"]["pool_cap"], MAX_TOTAL_WORKS)


if __name__ == "__main__":
    unittest.main()
