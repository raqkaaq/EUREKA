"""Sole orchestration boundary: collect, refresh, cached, and analyze flows.

The pipeline coordinates config, source, provider, agent, processing,
storage, and output. It owns transport/session lifetimes, coverage
semantics, and the typed error boundary (usage/source/analysis/storage
errors map to CLI exit codes). No printing, no argparse, no prompts.
"""

from __future__ import annotations

import os as _os
import sqlite3 as _sqlite3
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from radar.config.interests import (
    clean_keyword_override,
    default_profile,
    validate_lookback_days,
)
from radar.config.runtime import (
    MAX_MAX_CANDIDATES,
    MAX_QUERIES,
    MAX_TOTAL_WORKS,
    validate_analysis_timeout,
    validate_max_tokens,
    validate_openalex_timeout,
)
from radar.processing import ranking as _ranking
from radar.processing.evidence import attach_evidence
from radar.provider import strata as _strata
from radar.schema.papers import CollectedWork

if TYPE_CHECKING:
    from pydantic_ai.models import Model as _Model
    from radar.config.interests import RadarProfile as _RadarProfile
    from radar.schema.triage import TriageBatch as _TriageBatch
    from radar.storage.sqlite import SQLiteStore as _SQLiteStore


class PipelineUsageError(ValueError):
    """Bad flags, paths, or combos (CLI exit 4)."""


class PipelineSourceError(RuntimeError):
    """Discovery failure (CLI exit 2)."""


class PipelineAnalysisError(RuntimeError):
    """Model/report failure (CLI exit 3)."""


class PipelineStorageError(RuntimeError):
    """Snapshot state failure (CLI exit 3)."""


@dataclass(frozen=True)
class PipelineRequest:
    """One radar run: collect-only metadata or full analysis."""

    mode: Literal["collect", "analyze"] = "collect"
    max_candidates: int = 12
    lookback_days: int = 90
    timeout_s: float = 10.0
    keywords: tuple[str, ...] | None = None
    refresh_dir: str | None = None
    from_snapshot: str | None = None
    analysis_timeout_s: float = 90.0
    max_tokens: int = 2000
    disable_thinking: bool = False
    base_url: str | None = None
    model: str | None = None
    # Full-pool routing: CLEF preferred, Qwen fallback; collect-only ignores it.
    clef_base_url: str | None = None
    clef_model: str | None = None
    clef_timeout_s: float = 10.0
    triage_timeout_s: float = 60.0
    triage_output: str | None = None
    storage_dir: str | None = None
    from_database: bool = False
    rebuild_graph: bool = False
    # Injected doubles (tests only; production leaves all None).
    source_override: Any | None = None
    model_override: Any | None = None
    triage_model_override: Any | None = None
    clef_transport: Any | None = None
    triage_scorer: (
        Callable[[Sequence[CollectedWork], "_RadarProfile"], "_TriageBatch"] | None
    ) = None


@dataclass(frozen=True)
class PipelineResult:
    """Typed run outcome: exit intent, stdout payload, stderr notes."""

    exit_code: int
    stdout: str = ""
    stderr_notes: tuple[str, ...] = ()


def run(request: PipelineRequest) -> PipelineResult:
    """Execute one pipeline run, mapping typed errors to exit codes."""
    try:
        return _execute(request)
    except PipelineUsageError as exc:
        return PipelineResult(4, "", (f"radar: error: {exc}",))
    except PipelineSourceError as exc:
        return PipelineResult(2, "", (f"radar: error: {exc}",))
    except (PipelineAnalysisError, PipelineStorageError) as exc:
        return PipelineResult(3, "", (f"radar: error: {exc}",))


def _execute(request: PipelineRequest) -> PipelineResult:
    _validate_request(request)
    if request.storage_dir is not None:
        return _run_database(request)
    if request.from_snapshot is not None:
        if request.refresh_dir is not None:
            raise PipelineUsageError(
                "--from-snapshot cannot be combined with --refresh-dir; "
                "cached analysis never overwrites snapshot state."
            )
        return _run_cached(request)
    return _run_live(request)


def _validate_request(request: PipelineRequest) -> None:
    if (request.from_database or request.rebuild_graph) and not request.storage_dir:
        raise PipelineUsageError("Database modes require a storage directory.")
    if request.from_snapshot is not None and (request.from_database or request.rebuild_graph):
        raise PipelineUsageError("Choose only one cached/import/rebuild source.")
    if request.from_database and request.rebuild_graph:
        raise PipelineUsageError("--from-db cannot be combined with --rebuild-graph.")
    if request.storage_dir is not None and (request.refresh_dir or request.triage_output):
        raise PipelineUsageError("Database persistence cannot also write legacy JSON sidecars.")
    if not (1 <= request.max_candidates <= MAX_MAX_CANDIDATES):
        raise PipelineUsageError(
            f"--max-candidates must be within 1..{MAX_MAX_CANDIDATES}"
        )
    try:
        validate_openalex_timeout(request.timeout_s)
        validate_analysis_timeout(request.analysis_timeout_s)
        validate_max_tokens(request.max_tokens)
        validate_lookback_days(request.lookback_days)
    except ValueError as exc:
        raise PipelineUsageError(str(exc)) from exc
    # Validate editable policy before any source, provider or storage I/O.
    from radar.config.searches import search_config
    from radar.prompts.catalog import (
        opportunity_analysis_prompt, paper_triage_prompt, screening_questions,
    )
    from radar.agent.research_team import validate_research_prompts

    try:
        if request.from_snapshot is None and not (request.from_database or request.rebuild_graph):
            search_config()
        if request.mode == "analyze" and not request.rebuild_graph:
            screening_questions()
            paper_triage_prompt()
            opportunity_analysis_prompt()
            validate_research_prompts()
    except ValueError as exc:
        raise PipelineUsageError(str(exc)) from exc


def _run_database(request: PipelineRequest) -> PipelineResult:
    """All normal CLI storage, with legacy JSON permitted only as read-only input."""
    from radar.output import json as _out_json
    from radar.output import markdown as _out_md
    from radar.storage.graph_projection import rebuild_graph
    from radar.storage.sqlite import SQLiteStore, StorageError
    from radar.storage.snapshots import RefreshError, load_snapshot

    # Reject bad imports before opening/initializing any database or calling a model.
    imported = None
    if request.from_snapshot is not None:
        if not _os.path.isfile(request.from_snapshot):
            raise PipelineUsageError(f"snapshot not found: {request.from_snapshot}")
        try:
            imported = load_snapshot(request.from_snapshot)
        except RefreshError as exc:
            raise PipelineStorageError("Invalid legacy snapshot; no database or model was touched.") from exc
    try:
        with SQLiteStore(str(request.storage_dir)) as store:
            source = "legacy_snapshot" if imported is not None else "SQLite" if (request.from_database or request.rebuild_graph) else "OpenAlex"
            run_id = store.begin_run("rebuild" if request.rebuild_graph else request.mode, source)
            try:
                if request.rebuild_graph:
                    rebuild_graph(store)
                    store.finish_run(run_id, 0)
                    return PipelineResult(0, _out_json.refresh_envelope({
                        "database": str(store.path), "run_id": run_id, "graph": store.graph_state()}))
                notes: tuple[str, ...] = ()
                collected_at = None
                if imported is not None or request.from_database:
                    pool, metadata = imported if imported is not None else store.latest_pool()
                    collected_at = str(metadata['collected_at_utc'])
                    notes = (_out_md.staleness_note(collected_at),)
                else:
                    pool = _ranking.rank_works(_wrap_collection(request), _keywords(request))
                summary = store.save_pool(run_id, pool, collected_at=collected_at)
                if request.mode == "collect":
                    selected = _ranking.select_topn(pool, request.max_candidates)
                    result = PipelineResult(0, "", notes)
                else:
                    result = _analyze_pool(request, pool_full=pool, refresh_collected_at=None,
                                           prefix_notes=notes, storage=store, stored_run=run_id)
                rebuild_graph(store)
                store.finish_run(run_id, result.exit_code)
                if request.mode == "collect":
                    summary.update(selected=len(selected), works=[_out_json.work_to_json(w) for w in selected],
                                   graph=store.graph_state())
                    return PipelineResult(0, _out_json.refresh_envelope(summary), result.stderr_notes)
                return PipelineResult(result.exit_code, result.stdout, result.stderr_notes + (
                    f"storage: SQLite run={run_id}; Falkor projection ready",))
            except (PipelineUsageError, PipelineSourceError, PipelineAnalysisError, StorageError, _sqlite3.Error) as exc:
                code = 4 if isinstance(exc, PipelineUsageError) else 2 if isinstance(exc, PipelineSourceError) else 3
                store.finish_run(run_id, code, type(exc).__name__)
                raise
    except (StorageError, _sqlite3.Error) as exc:
        message = str(exc) if isinstance(exc, StorageError) else f"Database operation failed ({type(exc).__name__})."
        raise PipelineStorageError(message) from exc


def _keywords(request: PipelineRequest) -> list[str]:
    return list(_active_profile(request).keywords)


def _collect_live(request: PipelineRequest) -> list[CollectedWork]:
    from radar.config.searches import build_query_plan
    from radar.source.openalex import (
        HttpxTransport,
        RetryingTransport,
        collect,
    )

    profile = _active_profile(request)
    plan = build_query_plan(profile, max_queries=MAX_QUERIES)
    if request.source_override is not None:
        return collect(
            plan, RetryingTransport(request.source_override),
            timeout=request.timeout_s, max_total=MAX_TOTAL_WORKS,
        )
    base = HttpxTransport()
    try:
        return collect(
            plan, RetryingTransport(base),
            timeout=request.timeout_s, max_total=MAX_TOTAL_WORKS,
        )
    finally:
        base.close()


def _wrap_collection(request: PipelineRequest) -> list[CollectedWork]:
    try:
        return _collect_live(request)
    except ValueError as exc:
        raise PipelineUsageError(str(exc)) from exc
    except Exception as exc:
        raise PipelineSourceError(
            f"OpenAlex collection failed ({type(exc).__name__}: {exc}). "
            "Check network access to https://api.openalex.org and retry; "
            "per-request timeouts are bounded by --timeout."
        ) from exc


def _refresh_full_pool(
    pool: list[CollectedWork], refresh_dir: str
) -> dict[str, Any]:
    from radar.storage import snapshots as _snapshots

    try:
        return _snapshots.refresh_pool(pool, refresh_dir)
    except _snapshots.RefreshError as exc:
        raise PipelineStorageError(
            f"refresh failed ({exc}). Previous snapshot preserved."
        ) from exc


def _run_live(request: PipelineRequest) -> PipelineResult:
    from radar.output import json as _out_json
    from radar.storage import snapshots as _snapshots

    pool = _ranking.rank_works(_wrap_collection(request), _keywords(request))
    refresh_collected_at: str | None = None
    if request.refresh_dir:
        summary = _refresh_full_pool(pool, request.refresh_dir)
        refresh_collected_at = str(summary["collected_at_utc"])
        if request.mode == "collect":
            return PipelineResult(0, _out_json.refresh_envelope(summary))
    if request.mode == "collect":
        top = pool[: request.max_candidates]
        return PipelineResult(0, _out_json.collect_envelope(top))
    return _analyze_pool(
        request, pool_full=pool, refresh_collected_at=refresh_collected_at,
    )


def _run_cached(request: PipelineRequest) -> PipelineResult:
    from radar.output import json as _out_json
    from radar.storage import snapshots as _snapshots
    from radar.output import markdown as _out_md

    path = str(request.from_snapshot)
    if not _os.path.exists(path):
        raise PipelineUsageError(f"snapshot not found: {path}")
    try:
        cached_works, meta = _snapshots.load_snapshot(path)
    except _snapshots.RefreshError as exc:
        raise PipelineStorageError(
            f"invalid snapshot source ({exc}). No model was called."
        ) from exc
    notes = [_out_md.staleness_note(str(meta["collected_at_utc"]))]
    top = _ranking.select_topn(cached_works, request.max_candidates)
    if request.mode == "collect":
        return PipelineResult(
            0,
            _out_json.snapshot_inspection_envelope(
                snapshot=path,
                collected_at_utc=str(meta["collected_at_utc"]),
                coverage=dict(meta["coverage"]),
                discovery=dict(meta["discovery"]),
                disclosure=_snapshots.DISCLOSURE,
                selected=len(top),
                works=top,
            ),
            tuple(notes),
        )
    return _analyze_pool(
        request, pool_full=cached_works, refresh_collected_at=None,
        prefix_notes=tuple(notes),
    )


def _check_batch_shape(batch: Any, pool: list[CollectedWork]) -> "_TriageBatch":
    """Validate strict judgments and exact full-pool coverage before selection."""
    from radar.schema.triage import TriageBatch, TriageResult

    try:
        if not isinstance(batch, TriageBatch):
            batch = TriageBatch(
                model_id=batch.model_id, rubric_version=batch.rubric_version,
                results=[TriageResult.model_validate(vars(r)) for r in batch.results])
        else:
            batch = TriageBatch.model_validate(batch.model_dump())
        ids = [r.work_id for r in batch.results]
        expected = {w.openalex_id for w in pool}
        if len(ids) != len(pool) or len(expected) != len(pool) or set(ids) != expected:
            raise ValueError("incomplete or duplicated identities")
    except Exception:
        raise PipelineAnalysisError(
            "Routing returned invalid judgments or incomplete identity coverage; "
            "stopping before synthesis with the last valid snapshot preserved."
        ) from None
    return batch


def _active_profile(request: PipelineRequest) -> "_RadarProfile":
    """Discovery profile with the active CLI keyword override applied."""
    from radar.config.interests import RadarProfile

    profile: RadarProfile = default_profile(lookback_days=request.lookback_days)
    override = clean_keyword_override(
        list(request.keywords) if request.keywords else None
    )
    if override is not None:
        profile = profile.model_copy(update={"keywords": override})
    return profile


def _screen_full_pool(request: PipelineRequest, pool: list[CollectedWork],
                      profile: "_RadarProfile") -> "_TriageBatch":
    """Prefer CLEF; on absence/runtime failure, Qwen reroutes the entire pool."""
    from radar.config.runtime import (
        validate_clef_overall_timeout,
        validate_clef_request_timeout,
    )

    try:
        clef_timeout = validate_clef_request_timeout(request.clef_timeout_s)
        triage_timeout = validate_clef_overall_timeout(request.triage_timeout_s)
    except ValueError as exc:
        raise PipelineUsageError(str(exc)) from exc

    def fallback(reason: str) -> "_TriageBatch":
        from radar.agent import paper_triage

        if request.triage_model_override is None:
            try:
                _strata.resolve_base_url(request.base_url)
            except _strata.StrataError:
                raise PipelineUsageError(
                    "Qwen fallback requires a plain private-LAN Strata chat "
                    "endpoint: set STRATA_BASE_URL (legacy FREETOKEN_BASE_URL) or pass --base-url, "
                    "without embedded credentials."
                ) from None
        try:
            batch = paper_triage.screen_works(
                pool, profile, model=request.triage_model_override,
                model_id="qwen-test-model" if request.triage_model_override else None,
                base_url=request.base_url, configured_model=request.model,
                overall_timeout_s=triage_timeout,
                disable_thinking=_strata.resolve_disable_thinking(request.disable_thinking),
                fallback_reason=reason)
            return _check_batch_shape(batch, pool)
        except PipelineAnalysisError:
            raise
        except Exception:
            raise PipelineAnalysisError(
                "Qwen fallback routing failed; check the configured Strata "
                "chat endpoint/model. No synthesis was attempted; snapshot preserved."
            ) from None

    if request.triage_scorer is None:
        if not (request.clef_base_url or _os.environ.get("CLEF_BASE_URL", "").strip()):
            return fallback("missing_endpoint")
        from radar.provider.clef import ClefConfig, ClefError, screen_works

        try:
            config = ClefConfig.resolve(
                base_url=request.clef_base_url, model=request.clef_model,
                request_timeout_s=clef_timeout, overall_timeout_s=triage_timeout)
        except (ValueError, ClefError) as exc:
            raise PipelineUsageError(str(exc)) from None
        except Exception:
            raise PipelineUsageError("CLEF configuration could not be validated.") from None

    try:
        if request.triage_scorer is not None:
            batch = request.triage_scorer(pool, profile)
        else:
            batch = screen_works(pool, profile, config=config,
                                 transport=request.clef_transport)
        batch = _check_batch_shape(batch, pool)
    except Exception:
        return fallback("screening_error")
    if _has_deadline(batch):
        return fallback("deadline")
    if any(r.status == "failed" for r in batch.results):
        return fallback("screening_failed")
    return batch


def _has_deadline(batch: Any) -> bool:
    return any(
        getattr(r, "status", None) == "deadline"
        for r in (getattr(batch, "results", []) or []))


def _deadline_count(batch: Any) -> int:
    return sum(
        1 for r in (getattr(batch, "results", []) or [])
        if getattr(r, "status", None) == "deadline")


def _analyze_pool(
    request: PipelineRequest,
    *,
    pool_full: list[CollectedWork],
    refresh_collected_at: str | None,
    prefix_notes: tuple[str, ...] = (),
    storage: "_SQLiteStore | None" = None,
    stored_run: str | None = None,
) -> PipelineResult:
    from radar.agent import research_team as _team
    from radar.output import markdown as _out_md
    from radar.output import triage as _out_triage
    from radar.processing.triage import select_candidates
    from radar.storage import snapshots as _snapshots
    from radar.storage import triage as _triage_store

    notes: list[str] = list(prefix_notes)
    if not pool_full:
        return PipelineResult(0, "No candidates collected; nothing to analyze.",
                              tuple(notes))
    profile = _active_profile(request)
    batch = _screen_full_pool(request, pool_full, profile)
    if storage is not None and stored_run is not None:
        storage.save_triage(stored_run, batch)
    summary = _out_triage.summarize_batch(batch)
    if request.triage_output:
        try:
            _triage_store.write_sidecar(request.triage_output, batch)
        except _triage_store.TriageSidecarError as exc:
            notes.append(f"radar: warning: {exc}")
    if summary["failed"] > 0 or _has_deadline(batch):
        raise PipelineAnalysisError(
            f"{summary['backend']} screening incomplete: {summary['failed']} failed, "
            f"{_deadline_count(batch)} deadline "
            f"(model={summary['model_id']} rubric={summary['rubric_version']}); "
            "stopping before Strata synthesis with the last valid "
            "snapshot preserved. Qwen routing uses Strata Chat Completions "
            "with Pydantic-validated SystemOne answer contracts."
        )
    shortlist = select_candidates(pool_full, batch, request.max_candidates)
    try:
        included = _team.select_for_prompt(shortlist, request.max_candidates)
    except ValueError as exc:
        raise PipelineUsageError(str(exc)) from exc
    if not included:
        notes.append(_out_triage.triage_coverage_line(
            pool_total=len(pool_full), summary=summary, selected=0, opportunities=0))
        return PipelineResult(0, "No candidates fit the research prompt budget; nothing analyzed.",
                              tuple(notes))
    disable_thinking = _strata.resolve_disable_thinking(request.disable_thinking)
    session = None
    if request.model_override is not None:
        model = request.model_override
    else:
        try:
            config = _strata.StrataConfig.resolve(
                base_url=request.base_url, model=request.model)
            session = _strata.build_session(config)
        except _strata.StrataError as exc:
            raise PipelineAnalysisError(str(exc)) from exc
        model = None
    try:
        research = _team.research_candidates(
            included,
            model=model,
            max_candidates=request.max_candidates,
            analysis_timeout_s=request.analysis_timeout_s,
            max_tokens=request.max_tokens,
            disable_thinking=disable_thinking,
            session=session,
        )
    except _strata.StrataError as exc:
        raise PipelineAnalysisError(str(exc)) from exc
    draft = research.draft
    included = list(research.included)
    try:
        report = attach_evidence(draft, included)
        rendered = _out_md.render_markdown(report)
    except Exception as exc:
        raise PipelineAnalysisError(
            f"Report generation failed ({type(exc).__name__}: {exc})."
        ) from exc
    if storage is not None and stored_run is not None:
        storage.save_report(stored_run, report, len(included))
    notes.append(_out_triage.triage_coverage_line(
        pool_total=len(pool_full), summary=summary,
        selected=len(included), opportunities=len(draft.opportunities)))
    notes.append("research: specialists=" + ",".join(
        report.role for report in research.specialist_reports) + "; synthesis=opportunity_analysis")
    if request.refresh_dir and refresh_collected_at is not None:
        try:
            _snapshots.update_llm_coverage(
                request.refresh_dir,
                llm_selected=len(included),
                llm_analyzed=len(included),
                expect_collected_at=refresh_collected_at,
            )
        except _snapshots.RefreshError as exc:
            notes.append(
                f"radar: warning: could not update refresh coverage ({exc})")
    return PipelineResult(0, rendered, tuple(notes))


__all__ = [
    "PipelineAnalysisError",
    "PipelineRequest",
    "PipelineResult",
    "PipelineSourceError",
    "PipelineStorageError",
    "PipelineUsageError",
    "run",
]
