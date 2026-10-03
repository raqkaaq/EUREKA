"""Metadata-only refresh persistence: atomic full-pool snapshot + deltas.

The collector gathers a bounded OpenAlex pool (<=6 requests, <=200 works)
but the CLI otherwise only exposes the topN slice. This module persists the
*full* normalized pool (metadata/abstracts/locations/provenance) as a single
atomic JSON snapshot, without any LLM involvement.

Snapshot file layout (``snapshot.json`` inside ``refresh_dir``)::

    {
      "schema_version": 1,
      "collected_at_utc": "<UTC ISO-8601 timestamp>",
      "discovery": {"source": "OpenAlex", "bounded": true, ...},
      "coverage": {"collected": N, "with_abstracts": X,
                   "missing_abstracts": Y, "llm_selected": 0,
                   "llm_analyzed": 0},
      "delta": {"new_count": .., "changed_count": .., "unchanged_count": ..,
                "new_ids": [...], "changed_ids": [...]},
      "works": [ {<CollectedWork dict>}, ... ]  # sorted by openalex_id
    }

Delta identity is keyed on stable OpenAlex IDs and is insensitive to
snapshot timestamp, pool ranking, work ordering, and derived ranking
scores. Overlapping refreshes
are prevented by a bounded nonblocking lock file (fail-fast, always
released by its owner, even on error).
"""

from __future__ import annotations

import datetime as _dt
import json as _json
import os as _os
from typing import Any

from radar.models import MAX_QUERIES, CollectedWork
from radar.openalex import MAX_TOTAL_WORKS

SCHEMA_VERSION = 1
SNAPSHOT_FILENAME = "snapshot.json"
LOCK_FILENAME = "snapshot.lock"

DISCLOSURE = (
    "Bounded OpenAlex discovery sample (at most 6 requests, at most 200 "
    "pooled works); coverage counts describe this snapshot only, "
    "not all of OpenAlex."
)


class RefreshError(RuntimeError):
    """Refresh failure with the last valid snapshot left untouched."""


def snapshot_path(refresh_dir: str) -> str:
    """Return the snapshot file path inside *refresh_dir*."""
    return _os.path.join(str(refresh_dir), SNAPSHOT_FILENAME)


def _lock_path(refresh_dir: str) -> str:
    return _os.path.join(str(refresh_dir), LOCK_FILENAME)


def _utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def coverage_of(
    works: list[CollectedWork],
    llm_selected: int = 0,
    llm_analyzed: int = 0,
) -> dict[str, int]:
    """Explicit coverage counts for a pool (missing abstracts counted)."""
    with_abs = sum(1 for w in works if (w.abstract or "").strip())
    return {
        "collected": len(works),
        "with_abstracts": with_abs,
        "missing_abstracts": len(works) - with_abs,
        "llm_selected": int(llm_selected),
        "llm_analyzed": int(llm_analyzed),
    }


def _canonical(entry: Any) -> dict[str, Any]:
    """Order-insensitive canonical form of one persisted work entry.

    The derived ranking ``score`` is excluded: it is recomputed on every
    run and is not work identity, so a score-only change must not read
    as a changed work.
    """
    if isinstance(entry, dict):
        items = entry.items()
    else:
        items = dict(entry).items()
    return {k: v for k, v in items if k != "score"}


def compute_delta(
    old_works: list[dict[str, Any]], new_works: list[dict[str, Any]]
) -> dict[str, Any]:
    """Diff two snapshots' work lists by stable OpenAlex ID.

    Order-insensitive; ignores ranking and derived scores. Returns counts
    plus the sorted new/changed ID lists.
    """
    old_by_id = {str(w.get("openalex_id", "")): _canonical(w) for w in old_works if isinstance(w, dict)}
    new_by_id = {str(w.get("openalex_id", "")): _canonical(w) for w in new_works if isinstance(w, dict)}
    new_ids = sorted(i for i in new_by_id if i not in old_by_id)
    changed_ids = sorted(
        i for i in new_by_id if i in old_by_id and new_by_id[i] != old_by_id[i]
    )
    unchanged = sum(1 for i in new_by_id if i in old_by_id and new_by_id[i] == old_by_id[i])
    return {
        "new_count": len(new_ids),
        "changed_count": len(changed_ids),
        "unchanged_count": unchanged,
        "new_ids": new_ids,
        "changed_ids": changed_ids,
    }


def _acquire_lock(lock_path: str) -> int:
    """Create the lock file atomically; fail fast when one is held."""
    try:
        fd = _os.open(lock_path, _os.O_CREAT | _os.O_EXCL | _os.O_WRONLY, 0o644)
    except FileExistsError as exc:
        raise RefreshError(
            f"another refresh is already in progress (lock {lock_path}); "
            "not starting an overlapping refresh"
        ) from exc
    try:
        _os.write(fd, str(_os.getpid()).encode("utf-8", errors="replace"))
    except OSError:
        pass
    return fd


def _release_lock(fd: int, lock_path: str) -> None:
    try:
        _os.close(fd)
    except OSError:
        pass
    try:
        _os.unlink(lock_path)
    except OSError:
        pass


def _invalid_previous(path: str, reason: str) -> RefreshError:
    """Build a concise, redacted error for an unusable previous snapshot."""
    short = " ".join(reason.strip().split())[:200]
    return RefreshError(
        f"previous snapshot {path} is invalid ({short}); "
        "leaving it untouched, refusing to overwrite"
    )


def _load_previous(path: str) -> list[dict[str, Any]]:
    """Load and strictly validate the previous snapshot's works.

    Only producer schema v1 is supported: the payload must carry
    ``schema_version == 1`` and every entry of ``works`` must validate as
    a :class:`CollectedWork` with a unique OpenAlex ID. Anything else
    raises without touching the file -- malformed entries are never
    silently discarded or healed.
    """
    try:
        with open(path, encoding="utf-8") as fh:
            payload = _json.load(fh)
    except (OSError, ValueError) as exc:
        raise _invalid_previous(path, f"{type(exc).__name__}: {exc}") from exc
    if not isinstance(payload, dict):
        raise _invalid_previous(path, "top-level JSON value is not an object")
    version = payload.get("schema_version")
    if (
        isinstance(version, bool)
        or not isinstance(version, int)
        or version != SCHEMA_VERSION
    ):
        raise _invalid_previous(
            path,
            f"unsupported schema_version {version!r} "
            f"(supports v{SCHEMA_VERSION} only)",
        )
    raw_works = payload.get("works")
    if not isinstance(raw_works, list):
        raise _invalid_previous(path, "'works' is not a list")
    works: list[dict[str, Any]] = []
    seen: set[str] = set()
    for pos, entry in enumerate(raw_works):
        if not isinstance(entry, dict):
            raise _invalid_previous(path, f"works[{pos}] is not an object")
        try:
            work = CollectedWork.model_validate(entry)
        except ValueError as exc:
            raise _invalid_previous(path, f"works[{pos}] invalid: {exc}") from exc
        wid = work.openalex_id
        if not wid.strip():
            raise _invalid_previous(path, f"works[{pos}] has an empty openalex_id")
        if wid in seen:
            raise _invalid_previous(path, f"duplicate openalex_id {wid!r}")
        seen.add(wid)
        works.append(work.model_dump())
    return works


def _write_atomic(path: str, payload: dict[str, Any]) -> None:
    tmp_path = f"{path}.tmp-{_os.getpid()}"
    try:
        with open(tmp_path, "w", encoding="utf-8") as fh:
            _json.dump(payload, fh, indent=2, ensure_ascii=False)
            fh.write("\n")
    except OSError as exc:
        try:
            _os.unlink(tmp_path)
        except OSError:
            pass
        raise RefreshError(
            f"could not stage snapshot {path} ({exc}); previous snapshot preserved"
        ) from exc
    try:
        _os.replace(tmp_path, path)
    except OSError as exc:
        try:
            _os.unlink(tmp_path)
        except OSError:
            pass
        raise RefreshError(
            f"could not persist snapshot {path} ({exc}); previous snapshot preserved"
        ) from exc


def refresh_pool(
    works: list[CollectedWork],
    refresh_dir: str,
    *,
    collected_at: str | None = None,
    max_requests: int = MAX_QUERIES,
    pool_cap: int = MAX_TOTAL_WORKS,
) -> dict[str, Any]:
    """Persist the full pool as one atomic snapshot; return the summary.

    Raises :class:`RefreshError` on lock contention, an invalid previous
    snapshot, or any persistence failure -- always preserving the last
    valid snapshot and releasing the lock.
    """
    try:
        _os.makedirs(refresh_dir, exist_ok=True)
    except OSError as exc:
        raise RefreshError(
            f"cannot use refresh dir {refresh_dir} ({exc}); nothing was written"
        ) from exc
    lock_path = _lock_path(refresh_dir)
    try:
        fd = _acquire_lock(lock_path)
    except RefreshError:
        raise
    except OSError as exc:
        raise RefreshError(
            f"cannot create refresh lock ({exc}); nothing was written"
        ) from exc
    try:
        path = snapshot_path(refresh_dir)
        if _os.path.exists(path):
            old_works = _load_previous(path)
        else:
            old_works = []
        entries = [w.model_dump() for w in works]
        entries.sort(key=lambda e: str(e.get("openalex_id", "")))
        delta = compute_delta(old_works, entries)
        coverage = coverage_of(works)
        snapshot = {
            "schema_version": SCHEMA_VERSION,
            "collected_at_utc": collected_at or _utc_now(),
            "discovery": {
                "source": "OpenAlex",
                "bounded": True,
                "max_requests": int(max_requests),
                "pool_cap": int(pool_cap),
                "note": DISCLOSURE,
            },
            "coverage": coverage,
            "delta": delta,
            "works": entries,
        }
        _write_atomic(path, snapshot)
        return {
            "snapshot": path,
            "coverage": coverage,
            "delta": delta,
            "disclosure": DISCLOSURE,
        }
    finally:
        _release_lock(fd, lock_path)


def update_llm_coverage(
    refresh_dir: str, llm_selected: int, llm_analyzed: int
) -> dict[str, Any]:
    """Patch LLM counts into an existing snapshot's coverage (post-analysis)."""
    lock_path = _lock_path(refresh_dir)
    try:
        fd = _acquire_lock(lock_path)
    except RefreshError:
        raise
    except OSError as exc:
        raise RefreshError(f"cannot create refresh lock ({exc})") from exc
    try:
        path = snapshot_path(refresh_dir)
        try:
            with open(path, encoding="utf-8") as fh:
                payload = _json.load(fh)
        except (OSError, ValueError) as exc:
            raise RefreshError(
                f"cannot update coverage: snapshot {path} unreadable ({exc})"
            ) from exc
        if not isinstance(payload, dict) or not isinstance(payload.get("coverage"), dict):
            raise RefreshError(f"cannot update coverage: snapshot {path} has bad shape")
        payload["coverage"]["llm_selected"] = int(llm_selected)
        payload["coverage"]["llm_analyzed"] = int(llm_analyzed)
        _write_atomic(path, payload)
        return {
            "snapshot": path,
            "coverage": dict(payload["coverage"]),
            "disclosure": DISCLOSURE,
        }
    finally:
        _release_lock(fd, lock_path)
