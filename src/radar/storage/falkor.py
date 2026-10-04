"""Disposable FalkorDB graph projection over the SQLite-authoritative record.

SQLite stays authoritative: the graph is a rebuildable view, so a failed
rebuild leaves it pending for the next rebuild instead of half-published, and
nothing here claims an atomic swap between the two databases.

Only ``graph.rdb`` is written to the storage directory. The bundled engine is
started directly in the foreground on an ephemeral loopback port with a
per-instance password; no config, pid, log, socket or settings files are
created, upstream output is discarded, and error messages never carry the
password or server output.

    with FalkorGraph(directory) as graph:
        graph.rebuild(papers, reports)
        rows = graph.query("MATCH (p:Paper) RETURN p.title", {"t": "x"})

``lock=False`` is for a caller that already holds the directory advisory lock;
the default protects concurrent owned engines with flock on the directory FD.
"""

from __future__ import annotations

import fcntl
import os
import secrets
import socket
import subprocess
import time
from pathlib import Path
from typing import Any, Sequence

from radar.schema.opportunities import RadarReport
from radar.schema.papers import CollectedWork

GRAPH_NAME = "radar"
RDB_FILENAME = "graph.rdb"
HYPOTHESIS_ORIGIN = "hypothesis"
DOSSIER_ORIGIN = "model_interpretation"
DOSSIER_EVIDENCE_LEVEL = "abstract"
STARTUP_TIMEOUT_SECONDS = 15.0
SHUTDOWN_TIMEOUT_SECONDS = 5.0
SHUTDOWN_GRACE_SECONDS = 2.0
QUERY_TIMEOUT_MS = 5000
SOCKET_TIMEOUT_SECONDS = 5.0


class FalkorError(RuntimeError):
    """Safe Falkor failure; never carries the password or upstream output."""


class FalkorGraph:
    """One owned FalkorDB engine over a Radar storage directory."""

    def __init__(self, directory: str | Path, *, lock: bool = True) -> None:
        self.directory = Path(directory).expanduser().resolve()
        self.lock = lock
        self._lock_fd: int | None = None
        self._process: subprocess.Popen[bytes] | None = None
        self._client: Any | None = None
        self._graph: Any | None = None

    def __enter__(self) -> FalkorGraph:
        try:
            self._open_directory()
            self._acquire_lock()
            self._start()
        except FalkorError:
            self._cleanup()
            raise
        except Exception as exc:
            self._cleanup()
            raise FalkorError("Falkor engine could not be started") from exc
        return self

    def __exit__(self, exc_type: type | None, exc: BaseException | None, tb: Any) -> bool:
        try:
            if exc_type is None and self._client is not None:
                self._save()
            if self._client is not None:
                try:
                    self._client.execute_command("SHUTDOWN", "NOSAVE", "NOW")
                except Exception:
                    pass
            clean = self._reap()
            if exc_type is None and not clean:
                raise FalkorError("Falkor engine did not shut down cleanly")
        finally:
            self._cleanup()
        return False

    def rebuild(
        self,
        papers: Sequence[CollectedWork],
        reports: Sequence[tuple[str, RadarReport]],
    ) -> None:
        """Repopulate the graph from the authoritative SQLite projection.

        Opportunities are model-generated hypotheses; evidence edges come only
        from deterministically attached links that name a stored paper, so no
        citation or scientific relationship is invented here. Learning
        dossiers are model interpretations grounded at the abstract level:
        ``HAS_LEARNING_DOSSIER`` (run -> dossier) and ``ABOUT_PAPER``
        (dossier -> existing paper) use only the deterministically attached
        source, with no invented scientific ontology.
        """
        identities = {paper.openalex_id for paper in papers}
        if any(link.openalex_id and link.openalex_id not in identities
               for _, report in reports for opportunity in report.opportunities
               for link in opportunity.evidence_links):
            raise FalkorError("Report evidence names a paper absent from SQLite projection")
        if any(resolved.source.openalex_id not in identities
               for _, report in reports for resolved in report.learning_dossiers):
            raise FalkorError("Learning dossier source names a paper absent from SQLite projection")
        self._write("MATCH (n) DETACH DELETE n", {})
        for paper in papers:
            self._write(
                "MERGE (p:Paper {openalex_id: $openalex_id}) SET p.title = $title",
                {"openalex_id": paper.openalex_id, "title": paper.title},
            )
        for run_id, report in reports:
            self._write("MERGE (r:Run {id: $run_id})", {"run_id": run_id})
            for index, opportunity in enumerate(report.opportunities):
                opportunity_id = f"{run_id}/{index}"
                self._write(
                    "MERGE (o:Opportunity {id: $opportunity_id}) "
                    "SET o.title = $title, o.origin = $origin",
                    {"opportunity_id": opportunity_id, "title": opportunity.draft.title,
                     "origin": HYPOTHESIS_ORIGIN},
                )
                self._write(
                    "MATCH (r:Run {id: $run_id}) MATCH (o:Opportunity {id: $opportunity_id}) "
                    "MERGE (r)-[:HAS_OPPORTUNITY]->(o)",
                    {"run_id": run_id, "opportunity_id": opportunity_id},
                )
                for link in opportunity.evidence_links:
                    if not link.openalex_id:
                        continue
                    self._write(
                        "MATCH (o:Opportunity {id: $opportunity_id}) "
                        "MATCH (p:Paper {openalex_id: $openalex_id}) "
                        "MERGE (o)-[:SUPPORTED_BY]->(p)",
                        {"opportunity_id": opportunity_id, "openalex_id": link.openalex_id},
                    )
            for dossier_index, resolved in enumerate(report.learning_dossiers):
                dossier_id = f"{run_id}/learning/{dossier_index}"
                self._write(
                    "MERGE (d:LearningDossier {id: $dossier_id}) "
                    "SET d.paper_index = $paper_index, d.origin = $origin, "
                    "d.evidence_level = $evidence_level",
                    {"dossier_id": dossier_id,
                     "paper_index": resolved.dossier.paper_index,
                     "origin": DOSSIER_ORIGIN,
                     "evidence_level": DOSSIER_EVIDENCE_LEVEL},
                )
                self._write(
                    "MATCH (r:Run {id: $run_id}) MATCH (d:LearningDossier {id: $dossier_id}) "
                    "MERGE (r)-[:HAS_LEARNING_DOSSIER]->(d)",
                    {"run_id": run_id, "dossier_id": dossier_id},
                )
                if not resolved.source.openalex_id:
                    continue
                self._write(
                    "MATCH (d:LearningDossier {id: $dossier_id}) "
                    "MATCH (p:Paper {openalex_id: $openalex_id}) "
                    "MERGE (d)-[:ABOUT_PAPER]->(p)",
                    {"dossier_id": dossier_id, "openalex_id": resolved.source.openalex_id},
                )

    def query(self, cypher: str, params: dict | None = None) -> list[Any]:
        """Read-only result rows for CLI querying and testing."""
        if not isinstance(cypher, str) or not cypher.strip():
            raise FalkorError("Falkor queries must be a non-empty Cypher string")
        if params is not None and not isinstance(params, dict):
            raise FalkorError("Falkor query parameters must be a mapping")
        graph = self._graph
        if graph is None:
            raise FalkorError("Falkor graph is not open")
        try:
            if GRAPH_NAME not in self._client.execute_command("GRAPH.LIST"):
                return []
            result = graph.ro_query(cypher, params, timeout=QUERY_TIMEOUT_MS)
        except Exception as exc:
            raise FalkorError("Falkor read-only query failed") from exc
        return list(result.result_set)

    def _open_directory(self) -> None:
        if self.directory == Path('/tmp') or Path('/tmp') in self.directory.parents:
            raise FalkorError("Radar storage must not be located in /tmp")
        if (self.directory / RDB_FILENAME).is_symlink():
            raise FalkorError("Database paths must not be symbolic links")
        try:
            self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        except OSError as exc:
            raise FalkorError(f"Falkor graph directory {self.directory} is not usable") from exc

    def _acquire_lock(self) -> None:
        if not self.lock:
            return
        try:
            fd = os.open(self.directory, os.O_RDONLY | os.O_DIRECTORY)
        except OSError as exc:
            raise FalkorError(f"Falkor graph directory {self.directory} is not usable") from exc
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(fd)
            raise FalkorError(
                "another Radar run holds the storage directory lock; "
                "use lock=False when the caller already holds it"
            ) from None
        self._lock_fd = fd

    def _start(self) -> None:
        try:
            import redis
            import redislite
            from falkordb import Graph
            from redis.backoff import NoBackoff
            from redis.retry import Retry
        except ImportError as exc:
            raise FalkorError("Falkor engine native dependencies are not installed") from exc
        executable, module = redislite.__redis_executable__, redislite.__falkordb_module__
        if not executable or not module or not os.path.isfile(module):
            raise FalkorError("Falkor engine native components are unavailable")
        port = self._ephemeral_port()
        password = secrets.token_urlsafe(32)
        command = [
            str(executable), "--port", str(port), "--bind", "127.0.0.1",
            "--requirepass", password, "--loadmodule", str(module),
            "--dir", str(self.directory), "--dbfilename", RDB_FILENAME,
            "--save", "", "--appendonly", "no", "--logfile", "", "--pidfile", "",
            "--daemonize", "no",
        ]
        try:
            self._process = subprocess.Popen(
                command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                umask=0o077,
            )
        except OSError as exc:
            raise FalkorError("Falkor engine could not be started") from exc
        client_options = dict(host="127.0.0.1", port=port, password=password,
                              decode_responses=True, retry=Retry(NoBackoff(), 0))
        deadline = time.monotonic() + STARTUP_TIMEOUT_SECONDS
        while True:
            if self._process.poll() is not None:
                raise FalkorError("Falkor engine exited during startup")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise FalkorError("Falkor engine did not become reachable within the startup deadline")
            self._client = redis.Redis(**client_options, socket_timeout=min(0.25, remaining),
                                       socket_connect_timeout=min(0.25, remaining))
            try:
                self._client.ping()
                break
            except Exception:
                self._client.close()
                self._client = None
                time.sleep(min(0.1, max(0, deadline - time.monotonic())))
        self._client.close()
        self._client = redis.Redis(**client_options, socket_timeout=SOCKET_TIMEOUT_SECONDS,
                                   socket_connect_timeout=0.25)
        self._graph = Graph(self._client, GRAPH_NAME)

    def _ephemeral_port(self) -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind(("127.0.0.1", 0))
            return probe.getsockname()[1]

    def _save(self) -> None:
        try:
            self._client.execute_command("SAVE")
        except Exception as exc:
            raise FalkorError(
                "graph snapshot save failed; the graph is not ready"
            ) from exc

    def _reap(self) -> bool:
        process = self._process
        if process is None:
            return True
        if process.poll() is not None:
            return process.returncode == 0
        try:
            return process.wait(timeout=SHUTDOWN_TIMEOUT_SECONDS) == 0
        except subprocess.TimeoutExpired:
            pass
        try:
            process.terminate()
            process.wait(timeout=SHUTDOWN_GRACE_SECONDS)
            return False
        except (OSError, subprocess.TimeoutExpired):
            pass
        try:
            process.kill()
            process.wait(timeout=SHUTDOWN_GRACE_SECONDS)
        except (OSError, subprocess.TimeoutExpired):
            return False
        return False

    def _cleanup(self) -> None:
        if self._process is not None and self._process.poll() is None and self._client is not None:
            try:
                self._client.execute_command("SHUTDOWN", "NOSAVE", "NOW")
            except Exception:
                pass
        self._reap()
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass
            self._client = None
        self._graph = None
        self._process = None
        if self._lock_fd is not None:
            try:
                fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
            except OSError:
                pass
            try:
                os.close(self._lock_fd)
            except OSError:
                pass
            self._lock_fd = None

    def _write(self, cypher: str, params: dict) -> None:
        if self._graph is None:
            raise FalkorError("Falkor graph is not open")
        try:
            self._graph.query(cypher, params, timeout=QUERY_TIMEOUT_MS)
        except Exception as exc:
            raise FalkorError("Falkor graph write failed") from exc
