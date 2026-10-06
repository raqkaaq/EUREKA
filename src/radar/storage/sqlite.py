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
from radar.schema.documents import PDFDocument, PDFSource, PDFReading, DocumentFailure, DocumentRecord
from radar.schema.discovery import (
    CachedSearchPlan, DiscoveryMemory, QueryFeedback, SavedFinding, SearchWaveRecord,
)
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
CREATE TABLE IF NOT EXISTS paper_documents (
    work_id TEXT NOT NULL REFERENCES papers(openalex_id), sha256 TEXT NOT NULL,
    payload TEXT NOT NULL, PRIMARY KEY(work_id,sha256)
);
CREATE TABLE IF NOT EXISTS run_documents (
    run_id TEXT NOT NULL REFERENCES runs(id), work_id TEXT NOT NULL REFERENCES papers(openalex_id),
    payload TEXT NOT NULL, PRIMARY KEY(run_id,work_id)
);
CREATE TABLE IF NOT EXISTS run_searches (
    run_id TEXT NOT NULL REFERENCES runs(id), wave INTEGER NOT NULL CHECK(wave IN (1,2)),
    profile_hash TEXT NOT NULL, policy_hash TEXT NOT NULL, payload TEXT NOT NULL,
    PRIMARY KEY(run_id,wave)
);
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

    def _unfinished_run(self, run_id: str) -> None:
        row = self._db.execute("SELECT finished_at FROM runs WHERE id=?", (run_id,)).fetchone()
        if row is None or row[0] is not None:
            raise StorageError("Search history requires an existing unfinished run.")

    def save_search_wave(self, run_id: str, record: SearchWaveRecord) -> None:
        record = SearchWaveRecord.model_validate(record.model_dump())
        self._unfinished_run(run_id)
        if self._db.execute("SELECT 1 FROM run_searches WHERE run_id=? AND wave=?", (run_id, record.wave)).fetchone():
            raise StorageError("Generated search plans cannot be rewritten.")
        with self._db:
            self._db.execute("INSERT INTO run_searches VALUES (?,?,?,?,?)",
                             (run_id, record.wave, record.profile_hash, record.policy_hash, record.model_dump_json()))

    def search_waves(self, run_id: str) -> list[SearchWaveRecord]:
        try:
            return [SearchWaveRecord.model_validate_json(row[0]) for row in self._db.execute(
                "SELECT payload FROM run_searches WHERE run_id=? ORDER BY wave", (run_id,))]
        except ValueError as exc:
            raise StorageError("Stored search history is invalid; database preserved.") from exc

    def save_search_feedback(self, run_id: str, wave: int, feedback: list[QueryFeedback]) -> None:
        self._unfinished_run(run_id)
        records = {r.wave: r for r in self.search_waves(run_id)}
        record = records.get(wave)
        if record is None:
            raise StorageError("Retrieval feedback requires a generated search plan.")
        feedback = [QueryFeedback.model_validate(f.model_dump()) for f in feedback]
        if [f.query for f in feedback] != record.plan.query_plan.queries:
            raise StorageError("Retrieval feedback must match the planned queries in order.")
        if record.feedback or record.retrieval_failure:
            raise StorageError("Observed retrieval history cannot be rewritten.")
        updated = SearchWaveRecord.model_validate(record.model_copy(update={"feedback": feedback}).model_dump())
        with self._db:
            self._db.execute("UPDATE run_searches SET payload=? WHERE run_id=? AND wave=?",
                             (updated.model_dump_json(), run_id, wave))

    def save_search_failure(self, run_id: str, wave: int) -> None:
        """Record a failed wave without inventing per-query observations."""
        self._unfinished_run(run_id)
        record = next((r for r in self.search_waves(run_id) if r.wave == wave), None)
        if record is None or record.feedback or record.retrieval_failure:
            raise StorageError("Failed retrieval requires an unobserved generated wave.")
        updated = record.model_copy(update={"retrieval_failure": "source_error"})
        with self._db:
            self._db.execute("UPDATE run_searches SET payload=? WHERE run_id=? AND wave=?",
                             (updated.model_dump_json(), run_id, wave))

    def cached_search_plan(self, profile_hash: str, policy_hash: str) -> CachedSearchPlan | None:
        for payload, run_id, started_at in self._db.execute(
            "SELECT s.payload,s.run_id,r.started_at FROM run_searches s JOIN runs r ON r.id=s.run_id "
            "WHERE s.wave=1 AND s.profile_hash=? AND s.policy_hash=? ORDER BY s.rowid DESC",
            (profile_hash, policy_hash)):
            try:
                record = SearchWaveRecord.model_validate_json(payload)
            except ValueError as exc:
                raise StorageError("Cached search plan is invalid; database preserved.") from exc
            if record.profile_hash != profile_hash or record.policy_hash != policy_hash:
                raise StorageError("Cached search fingerprints are inconsistent; database preserved.")
            if record.origin == "generated":
                return CachedSearchPlan(run_id=run_id, started_at=started_at, record=record)
        return None

    def known_work_ids(self) -> set[str]:
        """Catalogue membership for exact novelty counts; never all sent to an LLM."""
        return {row[0] for row in self._db.execute("SELECT openalex_id FROM papers")}

    def discovery_memory(self) -> DiscoveryMemory:
        """Bounded intact findings/questions/history, with no mastery inference."""
        findings: list[SavedFinding] = []
        seen: set[str] = set()
        for payload, in self._db.execute("SELECT payload FROM reports ORDER BY rowid DESC LIMIT 6"):
            try:
                report = RadarReport.model_validate_json(payload)
            except ValueError as exc:
                raise StorageError("Stored findings are invalid; database preserved.") from exc
            for resolved in report.learning_dossiers:
                work_id = resolved.source.openalex_id
                if work_id in seen:
                    continue
                dossier = resolved.dossier
                findings.append(SavedFinding(
                    work_id=work_id, title=resolved.source.title[:2000],
                    contribution=dossier.reported_contribution, limits=dossier.assumptions_limits,
                    open_questions=dossier.open_questions, evidence_level=resolved.evidence_level))
                seen.add(work_id)
        # Completed readings survive synthesis failure and remain useful memory.
        for payload, title in self._db.execute(
            "SELECT d.payload,p.title FROM run_documents d JOIN papers p ON p.openalex_id=d.work_id "
            "ORDER BY d.rowid DESC LIMIT 12"):
            try:
                record = DocumentRecord.model_validate_json(payload)
            except ValueError as exc:
                raise StorageError("Stored reading memory is invalid; database preserved.") from exc
            if record.status != "read" or record.work_id in seen:
                continue
            notes = record.reading.notes
            findings.append(SavedFinding(work_id=record.work_id, title=title[:2000],
                                        contribution=notes.summary, limits=[notes.limitations], evidence_level="pdf_text"))
            seen.add(record.work_id)
        # Prefer source IDs for findings, then recently inserted catalogue IDs.
        known = list(dict.fromkeys([f.work_id for f in findings] + [r[0] for r in self._db.execute(
            "SELECT openalex_id FROM papers ORDER BY rowid DESC LIMIT 100")]))[:100]
        memory = DiscoveryMemory(known_work_ids=known)
        for finding in findings:
            if len(memory.findings) >= 6:
                break
            proposed = memory.model_copy(update={"findings": memory.findings + [finding]})
            if len(proposed.model_dump_json()) <= 10000:
                memory = proposed
        for payload, in self._db.execute("SELECT payload FROM run_searches ORDER BY rowid DESC LIMIT 4"):
            try:
                record = SearchWaveRecord.model_validate_json(payload)
            except ValueError as exc:
                raise StorageError("Stored retrieval memory is invalid; database preserved.") from exc
            for feedback in record.feedback:
                if len(memory.prior_feedback) >= 24:
                    break
                proposed = memory.model_copy(update={"prior_feedback": memory.prior_feedback + [feedback]})
                if len(proposed.model_dump_json()) <= 14000:
                    memory = proposed
        return DiscoveryMemory.model_validate(memory.model_dump())

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

    def _document_member(self, run_id: str, work_id: str) -> None:
        if self._db.execute("SELECT 1 FROM pool_papers WHERE run_id=? AND work_id=?",
                            (run_id, work_id)).fetchone() is None:
            raise StorageError("Document must belong to this run's paper pool.")

    def save_document(self, run_id: str, document: PDFDocument) -> None:
        document = PDFDocument.model_validate(document.model_dump())
        self._document_member(run_id, document.work_id)
        source = PDFSource.model_validate(document.model_dump(include=set(PDFSource.model_fields)))
        record = DocumentRecord(work_id=document.work_id, status="extracted", source=source)
        with self._db:
            # This is the verified extraction cache, not immutable run history.
            # Re-extraction of the same PDF can repair a corrupt cached payload.
            self._db.execute("INSERT INTO paper_documents VALUES (?,?,?) "
                             "ON CONFLICT(work_id,sha256) DO UPDATE SET payload=excluded.payload",
                             (document.work_id, document.sha256, document.model_dump_json()))
            self._db.execute("INSERT INTO run_documents VALUES (?,?,?) ON CONFLICT(run_id,work_id) "
                             "DO UPDATE SET payload=excluded.payload",
                             (run_id, document.work_id, record.model_dump_json()))

    def cached_document(self, work_id: str) -> PDFDocument | None:
        row = self._db.execute("SELECT payload FROM paper_documents WHERE work_id=? ORDER BY rowid DESC LIMIT 1",
                               (work_id,)).fetchone()
        try:
            return PDFDocument.model_validate_json(row[0]) if row else None
        except ValueError as exc:
            raise StorageError("Stored document is invalid; database preserved.") from exc

    def document_records(self, run_id: str) -> list[DocumentRecord]:
        try:
            return [DocumentRecord.model_validate_json(row[0]) for row in self._db.execute(
                "SELECT payload FROM run_documents WHERE run_id=? ORDER BY rowid", (run_id,))]
        except ValueError as exc:
            raise StorageError("Stored document outcome is invalid; database preserved.") from exc

    def save_document_failure(self, run_id: str, failure: DocumentFailure) -> None:
        failure = DocumentFailure.model_validate(failure.model_dump())
        self._document_member(run_id, failure.work_id)
        previous = next((r for r in self.document_records(run_id) if r.work_id == failure.work_id), None)
        record = DocumentRecord(work_id=failure.work_id, status="failed", failure=failure,
                                source=previous.source if previous else None)
        with self._db:
            self._db.execute("INSERT INTO run_documents VALUES (?,?,?) ON CONFLICT(run_id,work_id) "
                             "DO UPDATE SET payload=excluded.payload",
                             (run_id, failure.work_id, record.model_dump_json()))

    def save_document_reading(self, run_id: str, reading: PDFReading) -> None:
        reading = PDFReading.model_validate(reading.model_dump())
        self._document_member(run_id, reading.work_id)
        previous = next((r for r in self.document_records(run_id) if r.work_id == reading.work_id), None)
        if previous is None or previous.source is None or previous.source.sha256 != reading.sha256:
            raise StorageError("A reading requires this run's extracted document.")
        row = self._db.execute("SELECT payload FROM paper_documents WHERE work_id=? AND sha256=?",
                               (reading.work_id, reading.sha256)).fetchone()
        try:
            document = PDFDocument.model_validate_json(row[0]) if row else None
            if document is None or len(document.pages) != reading.page_count:
                raise ValueError("Source page mismatch")
            from radar.processing.document_chunks import chunk_document
            text, spans = chunk_document(document)
            actual_spans = [(c.start, c.end, c.pages) for c in reading.chunks]
            if len(text) != reading.text_chars or spans != actual_spans:
                raise ValueError("Source text coverage mismatch")
            pages = {p.number: " ".join(p.text.split()) for p in document.pages}
            child_quotes = set()
            for chunk in reading.chunks:
                for evidence in chunk.notes.evidence:
                    quote = " ".join(evidence.quote.split())
                    if (not quote or evidence.page not in chunk.pages
                            or quote not in pages.get(evidence.page, "")
                            or quote not in " ".join(text[chunk.start:chunk.end].split())):
                        raise ValueError("Unsupported source quote")
                    child_quotes.add((evidence.page, quote))
            if any((e.page, " ".join(e.quote.split())) not in child_quotes for e in reading.notes.evidence):
                raise ValueError("Final notes introduce unsupported source quotes")
            record = DocumentRecord(work_id=reading.work_id, status="read", source=previous.source, reading=reading)
        except ValueError as exc:
            raise StorageError("Document reading does not match its immutable PDF source.") from exc
        with self._db:
            self._db.execute("UPDATE run_documents SET payload=? WHERE run_id=? AND work_id=?",
                             (record.model_dump_json(), run_id, reading.work_id))

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
        outcomes = {r.work_id: r for r in self.document_records(run_id)}
        for reading in report.document_readings:
            record = outcomes.get(reading.work_id)
            if record is None or record.status != "read" or record.reading != reading:
                raise StorageError("Report requires this run's verified complete PDF reading.")
        for failure in report.document_failures:
            record = outcomes.get(failure.work_id)
            if record is None or record.status != "failed" or record.failure != failure:
                raise StorageError("Report PDF failures must match this run's outcomes.")
        for resolved in report.learning_dossiers:
            if not 0 <= resolved.source.index < selected:
                raise StorageError("Learning dossier source must be in the analyzed index range.")
            if resolved.evidence_level == "pdf_text":
                continue  # Validated above against the run's immutable document reading.
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
