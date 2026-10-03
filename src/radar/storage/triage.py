"""Triage sidecar: optional atomic JSON artifact beside metadata snapshots.

A sidecar records one CLEF screening batch (model/rubric provenance plus
per-work results) without ever mutating the strict v1 metadata snapshot.
Writes are atomic (temp file + rename in the same directory); on any
failure the previous sidecar -- or its absence -- is preserved.
"""

from __future__ import annotations

import datetime as _dt
import json as _json
import os as _os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from radar.schema.triage import TriageBatch

SIDECAR_FILENAME = "triage.json"
SIDECAR_SCHEMA = "triage-sidecar/v1"


class TriageSidecarError(RuntimeError):
    """Sidecar persistence failure; any previous artifact is preserved."""


def sidecar_path(directory: str) -> str:
    """Return the sidecar file path inside *directory*."""
    return _os.path.join(str(directory), SIDECAR_FILENAME)


def _utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def _result_to_json(result: Any) -> dict[str, Any]:
    dump = getattr(result, "model_dump", None)
    if callable(dump):
        payload = dump()
        if isinstance(payload, dict):
            return payload
    return {
        "work_id": getattr(result, "work_id", ""),
        "status": getattr(result, "status", "failed"),
        "ai_ml_relevance": getattr(result, "ai_ml_relevance", None),
        "cross_domain_potential": getattr(result, "cross_domain_potential", None),
    }


def write_sidecar(
    directory: str,
    batch: "TriageBatch",
    *,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Persist one screening batch atomically; return the summary.

    Raises :class:`TriageSidecarError` without touching any previous
    artifact when staging or renaming fails.
    """
    try:
        _os.makedirs(directory, exist_ok=True)
    except OSError as exc:
        raise TriageSidecarError(
            f"cannot use triage output dir {directory} ({exc}); nothing written"
        ) from exc
    path = sidecar_path(directory)
    tmp_path = f"{path}.tmp-{_os.getpid()}"
    results = getattr(batch, "results", []) or []
    payload = {
        "schema": SIDECAR_SCHEMA,
        "generated_at_utc": generated_at or _utc_now(),
        "model_id": getattr(batch, "model_id", ""),
        "rubric_version": getattr(batch, "rubric_version", ""),
        "results": [_result_to_json(r) for r in results],
    }
    try:
        with open(tmp_path, "w", encoding="utf-8") as fh:
            _json.dump(payload, fh, indent=2, ensure_ascii=False)
            fh.write("\n")
    except OSError as exc:
        try:
            _os.unlink(tmp_path)
        except OSError:
            pass
        raise TriageSidecarError(
            f"could not stage triage sidecar {path} ({exc}); "
            "previous artifact preserved"
        ) from exc
    try:
        _os.replace(tmp_path, path)
    except OSError as exc:
        try:
            _os.unlink(tmp_path)
        except OSError:
            pass
        raise TriageSidecarError(
            f"could not persist triage sidecar {path} ({exc}); "
            "previous artifact preserved"
        ) from exc
    return {"sidecar": path, "results": len(payload["results"])}


__all__ = [
    "SIDECAR_FILENAME",
    "SIDECAR_SCHEMA",
    "TriageSidecarError",
    "sidecar_path",
    "write_sidecar",
]
