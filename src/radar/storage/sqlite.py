"""Authoritative paper pools, screening, reports and run history in SQLite.

Schema creation and migrations are package-owned. An advisory directory lock
serializes CLI writers without a separate lock file; SQLite handles transactions.
"""

from __future__ import annotations

import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import sqlite3
from uuid import uuid4

from radar.config.runtime import MAX_TOTAL_WORKS
from radar.schema.opportunities import RadarReport
from radar.schema.papers import CollectedWork
from radar.schema.triage import TriageBatch
from radar.storage.snapshots import DISCLOSURE, compute_delta, coverage_of


class StorageError(RuntimeError):
    """Safe storage error; never includes stored paper or model contents."""


_SCHEMA = """
BEGIN IMMEDIATE;
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY, mode TEXT NOT NULL, source TEXT NOT NULL,
    started_at TEXT NOT NULL, finished_at TEXT, exit_code INTEGER,
    failure_category TEXT
);
CREATE TABLE IF NOT EXISTS papers (
    openalex_id TEXT PRIMARY KEY, title TEXT NOT NULL, abstract TEXT NOT NULL,
    payload TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS pools (
    run_id TEXT PRIMARY KEY REFERENCES runs(id), metadata TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS pool_papers (
    run_id TEXT NOT NULL REFERENCES pools(run_id), position INTEGER NOT NULL,
    work_id TEXT NOT NULL REFERENCES papers(openalex_id), payload TEXT NOT NULL,
    PRIMARY KEY (run_id, work_id), UNIQUE (run_id, position)
);
CREATE TABLE IF NOT EXISTS screening (
    run_id TEXT PRIMARY KEY REFERENCES runs(id), payload TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS reports (
    run_id TEXT PRIMARY KEY REFERENCES runs(id), payload TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS state (key TEXT PRIMARY KEY, value INTEGER NOT NULL);
INSERT OR IGNORE INTO state VALUES ('revision', 0), ('graph_revision', -1);
PRAGMA user_version = 1;
COMMIT;
"""


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


class SQLiteStore:
    """One bounded CLI writer; methods retain typed inputs and immutable history."""

    def __init__(self, directory: str | Path):
        self.directory = Path(directory).expanduser().resolve()
        self.path = self.directory / "radar.sqlite3"
        self.connection: sqlite3.Connection | None = None
        self._lock_fd: int | None = None

    def __enter__(self) -> SQLiteStore:
        try:
            if self.directory == Path('/tmp') or Path('/tmp') in self.directory.parents:
                raise StorageError("Radar storage must not be located in /tmp.")
            self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
            if self.path.is_symlink() or (self.directory / 'graph.rdb').is_symlink():
                raise StorageError("Database paths must not be symbolic links.")
            self._lock_fd = os.open(self.directory, os.O_RDONLY | os.O_DIRECTORY)
            try:
                fcntl.flock(self._lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise StorageError("Another Radar run is using this storage directory.") from None
            self.connection = sqlite3.connect(self.path, timeout=2.0)
            version = self.connection.execute("PRAGMA user_version").fetchone()[0]
            if version not in (0, 1):
                raise StorageError("Unsupported Radar database schema; database preserved.")
            if version == 0 and self.connection.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' LIMIT 1"
            ).fetchone() is not None:
                raise StorageError("Unrecognized database schema; database preserved.")
            self.connection.execute("PRAGMA foreign_keys=ON")
            self.connection.execute("PRAGMA trusted_schema=OFF")
            self.connection.execute("PRAGMA temp_store=MEMORY")
            self.connection.execute("PRAGMA journal_mode=DELETE")
            self.connection.execute("PRAGMA synchronous=FULL")
            self.connection.executescript(_SCHEMA)
            self.path.chmod(0o600)
            return self
        except (OSError, sqlite3.Error, StorageError) as exc:
            self.__exit__(None, None, None)
            if isinstance(exc, StorageError):
                raise
            raise StorageError(f"Cannot open Radar database ({type(exc).__name__}).") from exc

    def __exit__(self, *_args) -> None:
        try:
            if self.connection is not None:
                self.connection.close()
        finally:
            self.connection = None
            if self._lock_fd is not None:
                os.close(self._lock_fd)
                self._lock_fd = None

    @property
    def _db(self) -> sqlite3.Connection:
        if self.connection is None:
            raise StorageError("Radar database is not open.")
        return self.connection

    def begin_run(self, mode: str, source: str) -> str:
        run_id = uuid4().hex
        with self._db:
            self._db.execute("INSERT INTO runs(id,mode,source,started_at) VALUES (?,?,?,?)",
                             (run_id, mode, source, _now()))
        return run_id

    def finish_run(self, run_id: str, exit_code: int, failure_category: str | None = None) -> None:
        with self._db:
            self._db.execute("UPDATE runs SET finished_at=?,exit_code=?,failure_category=? WHERE id=?",
                             (_now(), exit_code, failure_category, run_id))

    def latest_pool(self) -> tuple[list[CollectedWork], dict]:
        row = self._db.execute("SELECT run_id,metadata FROM pools ORDER BY rowid DESC LIMIT 1").fetchone()
        if row is None:
            raise StorageError("No stored paper pool; collect or import a legacy snapshot first.")
        rows = self._db.execute("SELECT payload FROM pool_papers WHERE run_id=? ORDER BY position", (row[0],))
        try:
            papers = [CollectedWork.model_validate_json(item[0]) for item in rows]
            metadata = json.loads(row[1])
            if not isinstance(metadata, dict) or not isinstance(metadata.get('collected_at_utc'), str):
                raise ValueError('Invalid pool metadata')
            return papers, metadata
        except ValueError as exc:
            raise StorageError("Stored pool is invalid; database preserved.") from exc

    def save_pool(self, run_id: str, works: list[CollectedWork], *, collected_at: str | None = None) -> dict:
        works = [CollectedWork.model_validate(w.model_dump()) for w in works]
        if len(works) > MAX_TOTAL_WORKS or len({w.openalex_id for w in works}) != len(works):
            raise StorageError("Paper pool exceeds the bound or contains duplicate work IDs.")
        has_previous = self._db.execute("SELECT 1 FROM pools LIMIT 1").fetchone() is not None
        previous = self.latest_pool()[0] if has_previous else []
        summary = {
            "database": str(self.path), "run_id": run_id,
            "collected_at_utc": collected_at or _now(),
            "coverage": coverage_of(works),
            "delta": compute_delta([w.model_dump() for w in previous], [w.model_dump() for w in works]),
            "disclosure": DISCLOSURE,
            "discovery": {"source": "OpenAlex", "bounded": True},
        }
        with self._db:
            self._db.executemany(
                "INSERT INTO papers VALUES (?,?,?,?) ON CONFLICT(openalex_id) DO UPDATE "
                "SET title=excluded.title,abstract=excluded.abstract,payload=excluded.payload",
                [(w.openalex_id, w.title, w.abstract, w.model_dump_json()) for w in works])
            self._db.execute("INSERT INTO pools VALUES (?,?)", (run_id, json.dumps(summary)))
            self._db.executemany("INSERT INTO pool_papers VALUES (?,?,?,?)",
                                 [(run_id, i, w.openalex_id, w.model_dump_json()) for i, w in enumerate(works)])
            self._db.execute("UPDATE state SET value=value+1 WHERE key='revision'")
        return summary

    def save_triage(self, run_id: str, batch: TriageBatch) -> None:
        batch = TriageBatch.model_validate(batch.model_dump())
        expected = {r[0] for r in self._db.execute("SELECT work_id FROM pool_papers WHERE run_id=?", (run_id,))}
        actual = [r.work_id for r in batch.results]
        if len(actual) != len(expected) or set(actual) != expected:
            raise StorageError("Screening does not cover this stored pool exactly.")
        with self._db:
            self._db.execute("INSERT INTO screening VALUES (?,?)", (run_id, batch.model_dump_json()))

    def save_report(self, run_id: str, report: RadarReport, selected: int) -> None:
        report = RadarReport.model_validate(report.model_dump())
        expected = {r[0] for r in self._db.execute("SELECT work_id FROM pool_papers WHERE run_id=?", (run_id,))}
        if any(link.openalex_id not in expected for opportunity in report.opportunities for link in opportunity.evidence_links):
            raise StorageError("Report evidence must belong to this run's stored pool.")
        if any(resolved.source.openalex_id not in expected for resolved in report.learning_dossiers):
            raise StorageError("Learning dossier source must belong to this run's stored pool.")
        row = self._db.execute("SELECT metadata FROM pools WHERE run_id=?", (run_id,)).fetchone()
        if row is None or not 0 <= selected <= len(expected):
            raise StorageError("Invalid stored report coverage.")
        for resolved in report.learning_dossiers:
            if not 0 <= resolved.source.index < selected:
                raise StorageError("Learning dossier source must be in the analyzed index range.")
            source = self._db.execute(
                "SELECT payload FROM pool_papers WHERE run_id=? AND work_id=?",
                (run_id, resolved.source.openalex_id),
            ).fetchone()
            try:
                if source is None or not CollectedWork.model_validate_json(source[0]).abstract.strip():
                    raise ValueError("Missing abstract")
            except ValueError as exc:
                raise StorageError("Learning dossier requires a valid abstract in this run's pool.") from exc
        metadata = json.loads(row[0])
        metadata['coverage'].update(llm_selected=selected, llm_analyzed=selected)
        with self._db:
            self._db.execute("INSERT INTO reports VALUES (?,?)", (run_id, report.model_dump_json()))
            self._db.execute("UPDATE pools SET metadata=? WHERE run_id=?", (json.dumps(metadata), run_id))
            self._db.execute("UPDATE state SET value=value+1 WHERE key='revision'")

    def projection(self) -> tuple[list[CollectedWork], list[tuple[str, RadarReport]]]:
        try:
            papers = [CollectedWork.model_validate_json(row[0])
                      for row in self._db.execute("SELECT payload FROM papers ORDER BY openalex_id")]
            reports = [(row[0], RadarReport.model_validate_json(row[1]))
                       for row in self._db.execute("SELECT run_id,payload FROM reports ORDER BY rowid")]
            return papers, reports
        except ValueError as exc:
            raise StorageError("Stored graph inputs are invalid; database preserved.") from exc

    def graph_state(self) -> dict[str, int | bool]:
        state = dict(self._db.execute("SELECT key,value FROM state"))
        return {**state, "ready": state['revision'] == state['graph_revision']
                and (self.directory / 'graph.rdb').is_file()}

    def graph_pending(self) -> None:
        with self._db:
            self._db.execute("UPDATE state SET value=-1 WHERE key='graph_revision'")

    def graph_ready(self) -> None:
        with self._db:
            self._db.execute("UPDATE state SET value=(SELECT value FROM state WHERE key='revision') "
                             "WHERE key='graph_revision'")
