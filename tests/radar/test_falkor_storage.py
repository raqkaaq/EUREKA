"""Falkor graph adapter exercised through its public seam; real engine when available."""

import fcntl
import os
import subprocess
import tempfile
import unittest
from unittest import mock

from radar.schema.opportunities import EvidenceLink, Opportunity, OpportunityDraft, RadarReport
from radar.schema.papers import CollectedWork
from radar.storage import falkor
from radar.storage.falkor import FalkorError, FalkorGraph

try:
    import redis
except ImportError:
    redis = None

try:
    import redislite
except ImportError:
    redislite = None

ENGINE = bool(
    redis is not None and redislite is not None and redislite.__redis_executable__
    and os.path.isfile(redislite.__redis_executable__)
)
requires_engine = unittest.skipUnless(ENGINE, "bundled Falkor engine is unavailable")

PAPER = CollectedWork(openalex_id="https://openalex.org/W1", title="Paper")
REPORT = RadarReport(opportunities=[Opportunity(
    draft=OpportunityDraft(title="Hypothesis"),
    evidence_links=[EvidenceLink(index=0, openalex_id=PAPER.openalex_id, title=PAPER.title)])])


def spawn_recorder():
    """Keep the real subprocess boundary while recording what was spawned."""
    spawned = []
    popen = subprocess.Popen

    def record(command, **kwargs):
        process = popen(command, **kwargs)
        spawned.append(process)
        return process

    return spawned, record


class TestGraphThroughTheEngine(unittest.TestCase):
    @requires_engine
    def test_rebuild_populates_the_graph_and_query_returns_rows(self):
        with tempfile.TemporaryDirectory() as directory, FalkorGraph(directory) as graph:
            graph.rebuild([PAPER], [("run-1", REPORT)])
            self.assertEqual(
                graph.query("MATCH (p:Paper) RETURN p.openalex_id, p.title"),
                [[PAPER.openalex_id, PAPER.title]],
            )
            self.assertEqual(
                graph.query(
                    "MATCH (r:Run)-[:HAS_OPPORTUNITY]->(o:Opportunity) "
                    "RETURN r.id, o.id, o.origin"
                ),
                [["run-1", "run-1/0", "hypothesis"]],
            )
            self.assertEqual(
                graph.query(
                    "MATCH (o:Opportunity)-[:SUPPORTED_BY]->(p:Paper) "
                    "RETURN o.id, p.openalex_id"
                ),
                [["run-1/0", PAPER.openalex_id]],
            )

    @requires_engine
    def test_rebuild_never_invents_a_link_that_names_no_stored_paper(self):
        unattached = RadarReport(opportunities=[Opportunity(
            draft=OpportunityDraft(title="Hypothesis"),
            evidence_links=[EvidenceLink(index=0, title="unmatched candidate")])])
        with tempfile.TemporaryDirectory() as directory, FalkorGraph(directory) as graph:
            graph.rebuild([PAPER], [("run-1", unattached)])
            self.assertEqual(graph.query("MATCH (o)-[r:SUPPORTED_BY]->(p) RETURN type(r)"), [])

    @requires_engine
    def test_reopen_reads_the_persisted_graph_rdb_and_writes_nothing_else(self):
        with tempfile.TemporaryDirectory() as directory:
            with FalkorGraph(directory) as graph:
                graph.rebuild([PAPER], [("run-1", REPORT)])
            self.assertEqual(sorted(os.listdir(directory)), ["graph.rdb"])
            with FalkorGraph(directory) as reopened:
                self.assertEqual(
                    reopened.query("MATCH (p:Paper) RETURN p.openalex_id"),
                    [[PAPER.openalex_id]],
                )
            self.assertEqual(sorted(os.listdir(directory)), ["graph.rdb"])

    @requires_engine
    def test_query_parameters_are_data_not_query_text(self):
        with tempfile.TemporaryDirectory() as directory, FalkorGraph(directory) as graph:
            graph.rebuild([PAPER], [("run-1", REPORT)])
            self.assertEqual(
                graph.query(
                    "MATCH (p:Paper) WHERE p.title = $title RETURN p.openalex_id",
                    {"title": "MATCH (p:Paper) DELETE p"},
                ),
                [],
            )
            self.assertEqual(
                graph.query("MATCH (p:Paper) WHERE p.title = $title RETURN p.title",
                            {"title": "no such title"}),
                [],
            )
            self.assertEqual(
                graph.query("MATCH (p:Paper) RETURN p.openalex_id"), [[PAPER.openalex_id]]
            )
            self.assertEqual(graph.query("MATCH (p:Paper) WHERE p.title = $title RETURN p.title",
                                         {'title': PAPER.title}), [[PAPER.title]])

    @requires_engine
    def test_unmatched_evidence_is_rejected_before_replacing_valid_graph(self):
        invalid = REPORT.model_copy(deep=True)
        invalid.opportunities[0].evidence_links[0].openalex_id = 'https://openalex.org/W999'
        with tempfile.TemporaryDirectory() as directory, FalkorGraph(directory) as graph:
            graph.rebuild([PAPER], [])
            with self.assertRaises(FalkorError):
                graph.rebuild([PAPER], [('invalid-run', invalid)])
            self.assertEqual(graph.query('MATCH (p:Paper) RETURN p.openalex_id'), [[PAPER.openalex_id]])

    @requires_engine
    def test_query_is_read_only(self):
        with tempfile.TemporaryDirectory() as directory, FalkorGraph(directory) as graph:
            graph.rebuild([PAPER], [])
            with self.assertRaises(FalkorError):
                graph.query("CREATE (p:Paper {openalex_id: 'invented'})")
            self.assertEqual(graph.query("MATCH (p:Paper) RETURN p.openalex_id"),
                             [[PAPER.openalex_id]])

    @requires_engine
    def test_empty_rebuild_clears_the_graph(self):
        with tempfile.TemporaryDirectory() as directory, FalkorGraph(directory) as graph:
            graph.rebuild([PAPER], [("run-1", REPORT)])
            graph.rebuild([], [])
            self.assertEqual(graph.query("MATCH (n) RETURN labels(n)"), [])

    @requires_engine
    def test_failure_inside_the_context_publishes_no_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                with FalkorGraph(directory) as graph:
                    graph.rebuild([PAPER], [])
                    raise ValueError("caller failure")
            self.assertEqual(sorted(os.listdir(directory)), [])

    @requires_engine
    def test_failed_save_raises_so_the_graph_is_not_marked_ready(self):
        with tempfile.TemporaryDirectory() as directory:
            os.chmod(directory, 0o500)
            try:
                with self.assertRaises(FalkorError):
                    with FalkorGraph(directory) as graph:
                        graph.rebuild([PAPER], [])
            finally:
                os.chmod(directory, 0o700)
            self.assertEqual(sorted(os.listdir(directory)), [])


class TestOwnershipAndLocking(unittest.TestCase):
    def test_tmp_storage_and_symlink_targets_are_rejected_without_writes(self):
        with self.assertRaises(FalkorError):
            with FalkorGraph('/tmp/radar-storage-forbidden'):
                pass
        with tempfile.TemporaryDirectory() as directory:
            from pathlib import Path
            target = Path(directory) / 'original.rdb'
            target.write_bytes(b'preserve-original')
            (Path(directory) / 'graph.rdb').symlink_to(target)
            with self.assertRaises(FalkorError):
                with FalkorGraph(directory):
                    pass
            self.assertEqual(target.read_bytes(), b'preserve-original')

    @requires_engine
    def test_queries_after_close_fail_with_safe_adapter_error(self):
        with tempfile.TemporaryDirectory() as directory:
            with FalkorGraph(directory) as graph:
                graph.rebuild([PAPER], [])
            with self.assertRaises(FalkorError):
                graph.query('MATCH (p:Paper) RETURN p')

    @requires_engine
    def test_directory_lock_contention_and_lock_false_for_the_holder(self):
        with tempfile.TemporaryDirectory() as directory:
            with FalkorGraph(directory):
                with self.assertRaises(FalkorError):
                    with FalkorGraph(directory):
                        pass
            fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with FalkorGraph(directory, lock=False) as nested:
                    self.assertEqual(nested.query("MATCH (p:Paper) RETURN p"), [])
            finally:
                os.close(fd)
            self.assertEqual(sorted(os.listdir(directory)), ["graph.rdb"])

    @requires_engine
    def test_context_reaps_the_process_it_started(self):
        spawned, record = spawn_recorder()
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(falkor.subprocess, "Popen", record):
                with FalkorGraph(directory) as graph:
                    graph.rebuild([PAPER], [])
                    self.assertEqual(len(spawned), 1)
                    self.assertIsNone(spawned[0].poll())
                self.assertEqual(len(spawned), 1)
                self.assertIsNotNone(spawned[0].poll())

    @requires_engine
    def test_unresponsive_engine_is_reaped_only_through_its_own_handle(self):
        spawned, record = spawn_recorder()
        original = redis.Redis.execute_command

        def unreachable(client, command, *args, **kwargs):
            if command == "SHUTDOWN":
                raise redis.RedisError("server unreachable")
            return original(client, command, *args, **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(falkor.subprocess, "Popen", record), \
                    mock.patch.object(redis.Redis, "execute_command", unreachable):
                with self.assertRaises(FalkorError):
                    with FalkorGraph(directory) as graph:
                        graph.rebuild([PAPER], [])
                self.assertEqual(len(spawned), 1)
                self.assertIsNotNone(spawned[0].poll())

    @requires_engine
    def test_startup_failure_releases_the_lock_and_reaps_the_process(self):
        fake = mock.Mock()
        fake.poll.return_value = 1
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(falkor.subprocess, "Popen", return_value=fake) as spawn:
                with self.assertRaises(FalkorError) as raised:
                    with FalkorGraph(directory):
                        pass
            command = spawn.call_args.args[0]
            self.assertIs(spawn.call_args.kwargs["stdout"], subprocess.DEVNULL)
            self.assertIs(spawn.call_args.kwargs["stderr"], subprocess.DEVNULL)
            password = command[command.index("--requirepass") + 1]
            self.assertNotIn(password, str(raised.exception))
            self.assertNotIn("--requirepass", str(raised.exception))
            self.assertEqual(sorted(os.listdir(directory)), [])
            with FalkorGraph(directory) as graph:
                self.assertEqual(graph.query("MATCH (p:Paper) RETURN p"), [])

    @requires_engine
    def test_missing_native_components_raise_a_safe_falkor_error(self):
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(redislite, "__redis_executable__", ""):
                with self.assertRaises(FalkorError):
                    with FalkorGraph(directory):
                        pass
            self.assertEqual(sorted(os.listdir(directory)), [])
            with FalkorGraph(directory) as graph:
                self.assertEqual(graph.query("MATCH (p:Paper) RETURN p"), [])


if __name__ == "__main__":
    unittest.main()
