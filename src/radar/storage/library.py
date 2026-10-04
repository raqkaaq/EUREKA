"""Read-only saved-report library; never creates databases or starts services."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sqlite3

from radar.schema.opportunities import RadarReport
from radar.storage.sqlite import StorageError


@dataclass(frozen=True)
class SavedReport:
    run_id: str
    started_at: str
    exit_code: int | None
    report: RadarReport


class ReportLibrary:
    """A read-only SQLite connection independent of the CLI writer lifecycle."""

    def __init__(self, directory: str | Path):
        self.path = Path(directory).expanduser().resolve() / "radar.sqlite3"
        self.connection: sqlite3.Connection | None = None

    def __enter__(self) -> ReportLibrary:
        try:
            if self.path.is_symlink() or not self.path.is_file():
                raise StorageError("No readable Radar database; collect papers first.")
            self.connection = sqlite3.connect(self.path.as_uri() + "?mode=ro", uri=True, timeout=2.0)
            self.connection.execute("PRAGMA query_only=ON")
            self.connection.execute("PRAGMA trusted_schema=OFF")
            self.connection.execute("PRAGMA temp_store=MEMORY")
            if self.connection.execute("PRAGMA user_version").fetchone()[0] != 1:
                raise StorageError("Unsupported Radar database schema; database preserved.")
            return self
        except (OSError, sqlite3.Error, StorageError) as exc:
            self.__exit__(None, None, None)
            if isinstance(exc, StorageError):
                raise
            raise StorageError("Cannot read Radar report library; database preserved.") from exc

    def __exit__(self, *_args) -> None:
        if self.connection is not None:
            self.connection.close()
            self.connection = None

    def _read(self, *, run_id: str | None = None, limit: int = 20) -> list[SavedReport]:
        if self.connection is None:
            raise StorageError("Report library is not open.")
        sql = "SELECT r.run_id,u.started_at,u.exit_code,r.payload FROM reports r JOIN runs u ON u.id=r.run_id"
        values: tuple = (limit,)
        if run_id is not None:
            sql += " WHERE r.run_id=?"
            values = (run_id, limit)
        sql += " ORDER BY r.rowid DESC LIMIT ?"
        try:
            return [SavedReport(row[0], row[1], row[2], RadarReport.model_validate_json(row[3]))
                    for row in self.connection.execute(sql, values)]
        except (sqlite3.Error, ValueError) as exc:
            raise StorageError("Stored report is invalid or unreadable; database preserved.") from exc

    def recent(self, limit: int = 20) -> list[SavedReport]:
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100:
            raise StorageError("Report history limit must be within 1..100.")
        return self._read(limit=limit)

    def get(self, run_id: str = "latest") -> SavedReport:
        rows = self._read(run_id=None if run_id == "latest" else run_id, limit=1)
        if not rows:
            raise StorageError("No saved report matches; use --reports to inspect report IDs.")
        return rows[0]
