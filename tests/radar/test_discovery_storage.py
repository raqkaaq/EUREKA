"""Adaptive search memory and immutable plans persist only in SQLite."""

import hashlib
import tempfile
import unittest

from radar.schema.discovery import SearchIntent, SearchWavePlan, SearchWaveRecord, QueryFeedback
from radar.schema.papers import PlannedQuery
from radar.storage.sqlite import SQLiteStore, StorageError

HASH = hashlib.sha256(b"profile").hexdigest()
POLICY = hashlib.sha256(b"policy").hexdigest()


def plan():
    return SearchWavePlan(summary="Investigate a missing assumption", intents=[SearchIntent(
        learning_goal_id="generalization", question="Which assumption enables transfer?",
        rationale="Distinguish mechanisms from task similarity.",
        expected_learning_value="Reconstruct a generalization argument.", origin="agenda",
        query=PlannedQuery(kind="semantic", terms="assumptions for transfer generalization",
                           question_id="transfer", role="foundation"))])


class TestDiscoveryStorage(unittest.TestCase):
    def test_repeated_cached_runs_do_not_hide_the_generated_plan(self):
        with tempfile.TemporaryDirectory() as directory, SQLiteStore(directory) as store:
            original = store.begin_run("collect", "OpenAlex")
            store.save_search_wave(original, SearchWaveRecord(wave=1, profile_hash=HASH,
                policy_hash=POLICY, plan=plan()))
            store.finish_run(original, 0)
            for _ in range(25):
                run_id = store.begin_run("collect", "OpenAlex")
                store.save_search_wave(run_id, SearchWaveRecord(wave=1, profile_hash=HASH,
                    policy_hash=POLICY, origin="cached", plan=plan()))
                store.finish_run(run_id, 0)
            self.assertEqual(store.cached_search_plan(HASH, POLICY).run_id, original)

    def test_plan_and_feedback_survive_reopening_without_sidecars(self):
        with tempfile.TemporaryDirectory() as directory:
            with SQLiteStore(directory) as store:
                run_id = store.begin_run("collect", "OpenAlex")
                record = SearchWaveRecord(wave=1, profile_hash=HASH, policy_hash=POLICY, plan=plan())
                store.save_search_wave(run_id, record)
                feedback = QueryFeedback(query=plan().query_plan.queries[0], returned=2, accepted=1,
                                         new_to_run=1, new_to_library=1, with_abstract=1, with_pdf=0)
                store.save_search_feedback(run_id, 1, [feedback])
                store.finish_run(run_id, 0)
            with SQLiteStore(directory) as store:
                cached = store.cached_search_plan(HASH, POLICY)
                self.assertEqual(cached.record.plan, plan())
                self.assertEqual(cached.run_id, run_id)
                self.assertEqual(store.search_waves(run_id)[0].feedback, [feedback])
                self.assertEqual(store.discovery_memory().prior_feedback, [feedback])
                self.assertIsNone(store.cached_search_plan(HASH, "0" * 64))
                with self.assertRaises(StorageError):
                    store.save_search_feedback(run_id, 1, [])
                with self.assertRaises(StorageError):
                    store.save_search_wave(run_id, record)

    def test_feedback_cannot_rewrite_the_generated_queries(self):
        with tempfile.TemporaryDirectory() as directory, SQLiteStore(directory) as store:
            run_id = store.begin_run("collect", "OpenAlex")
            store.save_search_wave(run_id, SearchWaveRecord(wave=1, profile_hash=HASH, policy_hash=POLICY, plan=plan()))
            wrong = QueryFeedback(query=PlannedQuery(kind="keyword", terms="unrelated"), returned=0,
                                  accepted=0, new_to_run=0, new_to_library=0, with_abstract=0, with_pdf=0)
            with self.assertRaises(StorageError):
                store.save_search_feedback(run_id, 1, [wrong])
