"""Rebuild the disposable graph from SQLite; publish readiness after disk save."""

from radar.storage.falkor import FalkorError, FalkorGraph
from radar.storage.sqlite import SQLiteStore, StorageError


def rebuild_graph(store: SQLiteStore) -> None:
    """Caller holds the directory lock throughout reading, rebuilding and save.

    The two databases are not a distributed transaction. Failures leave SQLite
    authoritative and the graph explicitly pending, repairable by this function.
    """
    store.graph_pending()
    try:
        papers, reports = store.projection()
        with FalkorGraph(store.directory, lock=False) as graph:
            graph.rebuild(papers, reports)
        if not (store.directory / 'graph.rdb').is_file():
            raise StorageError("Falkor did not persist its database; SQLite is preserved. Retry --rebuild-graph.")
        store.graph_ready()
    except FalkorError as exc:
        raise StorageError("Falkor projection failed; SQLite records are preserved. "
                           "Retry with --rebuild-graph.") from exc
