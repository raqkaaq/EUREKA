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
from radar.schema.documents import PDFDocument, DocumentFailure
from radar.schema.discovery import DiscoveryMemory, SearchWaveRecord

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
    learning_question: str | None = None
    # Full-pool routing: CLEF preferred, Qwen fallback; collect-only ignores it.
    clef_base_url: str | None = None
    clef_model: str | None = None
    clef_timeout_s: float = 10.0
    triage_timeout_s: float = 60.0
    qwen_triage_timeout_s: float | None = None
    triage_output: str | None = None
    storage_dir: str | None = None
    from_database: bool = False
    rebuild_graph: bool = False
    list_reports: bool = False
    report_id: str | None = None
    # Injected doubles (tests only; production leaves all None).
    source_override: Any | None = None
    model_override: Any | None = None
    planner_model_override: Any | None = None
    triage_model_override: Any | None = None
    clef_transport: Any | None = None
    triage_scorer: (
        Callable[[Sequence[CollectedWork], "_RadarProfile"], "_TriageBatch"] | None
    ) = None
    document_timeout_s: float = 900.0
    # Explicit acquisition seam: (work, directory, cached) -> PDFDocument.
    document_loader: Callable[[CollectedWork, str, PDFDocument | None], PDFDocument] | None = None
    document_model_override: Any | None = None


@dataclass(frozen=True)
class PipelineResult:
    """Typed run outcome: exit intent, stdout payload, stderr notes."""

    exit_code: int
    stdout: str = ""
    stderr_notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class _CollectionOutcome:
    works: list[CollectedWork]
    notes: tuple[str, ...] = ()


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
    if request.learning_question is not None and (
        not isinstance(request.learning_question, str) or not request.learning_question.strip()
    ):
        raise PipelineUsageError("--learning-question must not be blank.")
    if request.list_reports or request.report_id is not None:
        return _run_library(request)
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


def _run_library(request: PipelineRequest) -> PipelineResult:
    """Browse saved scientific output without policy/model preflight or writers."""
    from radar.output.markdown import render_markdown
    from radar.storage.library import ReportLibrary
    from radar.storage.sqlite import StorageError

    if not request.storage_dir or not request.storage_dir.strip():
        raise PipelineUsageError("Report browsing requires a storage directory.")
    if request.list_reports and request.report_id is not None:
        raise PipelineUsageError("Choose --reports or --report, not both.")
    if request.from_database or request.rebuild_graph or request.from_snapshot is not None or request.refresh_dir or request.triage_output:
        raise PipelineUsageError("Report browsing cannot be combined with collection/import/analysis/rebuild modes.")
    if request.report_id is not None and (not request.report_id.strip() or len(request.report_id) > 128):
        raise PipelineUsageError("Choose a valid report ID or latest.")
    try:
        with ReportLibrary(request.storage_dir) as library:
            if request.list_reports:
                lines = ["# Saved radar reports", ""]
                for saved in library.recent():
                    dossiers = len(getattr(saved.report, "learning_dossiers", []))
                    label = f"{dossiers} learning dossier(s)" if dossiers else "legacy/opportunity-only report"
                    status = "finished" if saved.exit_code == 0 else "run unfinished or failed; saved report retained"
                    lines.append(f"- `{saved.run_id}` — {saved.started_at}; {label}; {status}")
                if len(lines) == 2:
                    lines.append("No reports saved yet.")
                lines.extend(["", "Open one with `radar --report RUN_ID` (or `--report latest`)."])
                return PipelineResult(0, "\n".join(lines))
            saved = library.get(request.report_id or "latest")
            header = f"Saved report `{saved.run_id}` — {saved.started_at}\n\n"
            if saved.exit_code != 0:
                header += "Note: the enclosing run did not finish successfully; this is its retained report.\n\n"
            return PipelineResult(0, header + render_markdown(saved.report))
    except StorageError as exc:
        raise PipelineStorageError(str(exc)) from exc


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
        from radar.config.documents import validate_document_timeout
        validate_document_timeout(request.document_timeout_s)
        from radar.config.runtime import validate_qwen_overall_timeout
        if request.qwen_triage_timeout_s is not None:
            validate_qwen_overall_timeout(request.qwen_triage_timeout_s)
    except ValueError as exc:
        raise PipelineUsageError(str(exc)) from exc
    # Validate editable policy before any source, provider or storage I/O.
    from radar.config.searches import search_config
    from radar.config.triage import qwen_screening_policy
    from radar.prompts.catalog import (
        opportunity_analysis_prompt, paper_triage_prompt, screening_questions,
    )
    from radar.agent.research_team import validate_research_prompts

    try:
        if request.from_snapshot is None and not (request.from_database or request.rebuild_graph):
            search_config()
            _active_profile(request)
            from radar.prompts.catalog import search_planning_prompt
            search_planning_prompt()
        if request.mode == "analyze" and not request.rebuild_graph:
            screening_questions()
            qwen_screening_policy()
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
                    collection = _wrap_collection(request, storage=store, stored_run=run_id)
                    pool = _ranking.rank_works(collection.works, _keywords(request))
                    notes += collection.notes
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
            except (PipelineUsageError, PipelineSourceError, PipelineAnalysisError, PipelineStorageError, StorageError, _sqlite3.Error) as exc:
                code = 4 if isinstance(exc, PipelineUsageError) else 2 if isinstance(exc, PipelineSourceError) else 3
                store.finish_run(run_id, code, type(exc).__name__)
                raise
    except (StorageError, _sqlite3.Error) as exc:
        message = str(exc) if isinstance(exc, StorageError) else f"Database operation failed ({type(exc).__name__})."
        raise PipelineStorageError(message) from exc


def _keywords(request: PipelineRequest) -> list[str]:
    return list(_active_profile(request).keywords)


def _plan_searches(request, profile, memory, policy, **kwargs):
    """One bounded planning call; create/close its Strata client in one loop."""
    import asyncio
    from radar.agent.discovery_planning import plan_searches_async

    async def plan():
        session = None
        try:
            async with asyncio.timeout(policy.planning_timeout_s):
                model = request.planner_model_override
                if model is None:
                    config = await _strata.StrataConfig.resolve_async(base_url=request.base_url, model=request.model)
                    session = _strata.build_session(config)
                    model = session.model
                return await plan_searches_async(
                    profile, memory, model=model, policy=policy,
                    disable_thinking=_strata.resolve_disable_thinking(request.disable_thinking), **kwargs)
        finally:
            if session is not None:
                await session.http_client.aclose()
    try:
        return asyncio.run(plan())
    except Exception:
        raise _strata.StrataError("Adaptive search planning failed; check the configured Strata endpoint/model.") from None


def _collect_live(request: PipelineRequest, *, storage=None, stored_run=None) -> _CollectionOutcome:
    from radar.config.searches import search_config
    from radar.agent.discovery_planning import planning_fingerprints, validate_search_plan
    from radar.source.openalex import (
        HttpxTransport,
        RetryingTransport,
        collect_with_feedback,
    )
    profile = _active_profile(request)
    policy = search_config()
    memory = storage.discovery_memory() if storage is not None else DiscoveryMemory()
    known_ids = storage.known_work_ids() if storage is not None else set()
    profile_hash, policy_hash = planning_fingerprints(profile, policy)
    notes: list[str] = []
    cached = False
    try:
        first = _plan_searches(request, profile, memory, policy)
    except _strata.StrataError:
        saved = storage.cached_search_plan(profile_hash, policy_hash) if storage is not None else None
        if saved is None:
            raise PipelineAnalysisError("Adaptive search planning failed and no compatible saved plan exists; no OpenAlex queries executed.") from None
        required = [wid for intent in saved.record.plan.intents for wid in intent.source_work_ids]
        required += [intent.query.seed_work_id for intent in saved.record.plan.intents if intent.query.seed_work_id]
        if any(wid not in known_ids for wid in required):
            raise PipelineAnalysisError("Saved search plan references unavailable source papers; no queries executed.") from None
        cache_memory = memory.model_copy(update={"known_work_ids": list(dict.fromkeys(required + memory.known_work_ids))[:100]})
        try:
            first = validate_search_plan(saved.record.plan, profile, cache_memory, policy=policy)
        except ValueError:
            raise PipelineAnalysisError("Saved search plan is no longer valid; no OpenAlex queries executed.") from None
        cached = True
        notes.append(f"discovery: planner unavailable; reusing cached initial plan from {saved.started_at} (run={saved.run_id}); stale plan, refinement skipped.")
    if storage is not None and stored_run is not None:
        storage.save_search_wave(stored_run, SearchWaveRecord(
            wave=1, profile_hash=profile_hash, policy_hash=policy_hash,
            origin="cached" if cached else "generated", plan=first))
    base = HttpxTransport() if request.source_override is None else None
    transport = RetryingTransport(request.source_override if base is None else base)
    try:
        try:
            initial = collect_with_feedback(first.query_plan, transport, known_work_ids=known_ids,
                                            timeout=request.timeout_s, max_total=MAX_TOTAL_WORKS)
        except Exception:
            if storage is not None and stored_run is not None:
                storage.save_search_failure(stored_run, 1)
            raise
        if storage is not None and stored_run is not None:
            storage.save_search_feedback(stored_run, 1, initial.feedback)
        combined = initial
        queries = len(first.intents)
        if not cached and len(initial.works) < MAX_TOTAL_WORKS:
            try:
                followup = _plan_searches(request, profile, memory, policy,
                                         feedback=initial.feedback, retrieved=initial.works, previous_plan=first)
            except _strata.StrataError:
                notes.append("discovery: refinement planning failed; retaining initial retrieval without static fallback.")
            else:
                if len(first.intents) + len(followup.intents) > MAX_QUERIES:
                    raise PipelineAnalysisError("Adaptive discovery exceeded its total query budget.")
                if storage is not None and stored_run is not None:
                    storage.save_search_wave(stored_run, SearchWaveRecord(
                        wave=2, profile_hash=profile_hash, policy_hash=policy_hash, plan=followup))
                try:
                    combined = collect_with_feedback(followup.query_plan, transport, existing=initial.works,
                                                     known_work_ids=known_ids, timeout=request.timeout_s,
                                                     max_total=MAX_TOTAL_WORKS)
                except Exception:
                    if storage is not None and stored_run is not None:
                        storage.save_search_failure(stored_run, 2)
                    notes.append("discovery: followup retrieval failed; retaining initial retrieval; partial followup observations unavailable.")
                else:
                    queries += len(followup.intents)
                    if storage is not None and stored_run is not None:
                        storage.save_search_feedback(stored_run, 2, combined.feedback)
        notes.append(f"discovery: completed-wave queries={queries}; unique papers={len(combined.works)}; retrieval intent is not scientific importance or mastery.")
        return _CollectionOutcome(combined.works, tuple(notes))
    finally:
        if base is not None:
            base.close()


def _wrap_collection(request: PipelineRequest, *, storage=None, stored_run=None) -> _CollectionOutcome:
    from radar.storage.sqlite import StorageError
    try:
        return _collect_live(request, storage=storage, stored_run=stored_run)
    except (PipelineAnalysisError, PipelineStorageError):
        raise
    except (StorageError, _sqlite3.Error) as exc:
        message = str(exc) if isinstance(exc, StorageError) else f"Database operation failed ({type(exc).__name__})."
        raise PipelineStorageError(message) from exc
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

    collection = _wrap_collection(request)
    pool = _ranking.rank_works(collection.works, _keywords(request))
    refresh_collected_at: str | None = None
    if request.refresh_dir:
        summary = _refresh_full_pool(pool, request.refresh_dir)
        refresh_collected_at = str(summary["collected_at_utc"])
        if request.mode == "collect":
            return PipelineResult(0, _out_json.refresh_envelope(summary), collection.notes)
    if request.mode == "collect":
        top = pool[: request.max_candidates]
        return PipelineResult(0, _out_json.collect_envelope(top), collection.notes)
    return _analyze_pool(
        request, pool_full=pool, refresh_collected_at=refresh_collected_at, prefix_notes=collection.notes,
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
                overall_timeout_s=request.qwen_triage_timeout_s,
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


def _default_document_loader(work: CollectedWork, directory: str,
                             cached: PDFDocument | None) -> PDFDocument:
    """Production acquisition through OpenAlex locations only."""
    from radar.source import pdf as _pdf

    return _pdf.acquire_pdf(work, directory, cached=cached)


def _acquire_shortlist_pdfs(
    request: PipelineRequest,
    shortlist: list[CollectedWork],
    *,
    storage: "_SQLiteStore | None" = None,
    stored_run: str | None = None,
) -> tuple[list[PDFDocument], list[DocumentFailure]]:
    """Attempt every shortlisted PDF; persist each outcome immediately."""
    from radar.source.pdf import PDFError as _PDFError

    loader = request.document_loader or _default_document_loader
    documents: list[PDFDocument] = []
    failures: list[DocumentFailure] = []
    # Even legacy API runs keep downloads in the normal data location, never
    # temporary directories or generated harness/sidecar files.
    pdf_root = str(storage.directory) if storage is not None else _os.environ.get("RADAR_STORAGE_DIR", "data/radar")
    for work in shortlist:
        cached = None
        if storage is not None:
            cached = storage.cached_document(work.openalex_id)
        try:
            try:
                document = loader(work, pdf_root, cached)
            except _PDFError as exc:
                if cached is None or exc.category != "cache_mismatch":
                    raise
                document = loader(work, pdf_root, None)
            document = PDFDocument.model_validate(document.model_dump())
            if document.work_id != work.openalex_id:
                raise ValueError("Acquisition returned an unrelated document")
        except Exception as exc:
            category = exc.category if isinstance(exc, _PDFError) else "acquisition_failed"
            failure = DocumentFailure(work_id=work.openalex_id, category=category)
            failures.append(failure)
            if storage is not None and stored_run is not None:
                storage.save_document_failure(stored_run, failure)
            continue
        documents.append(document)
        if storage is not None and stored_run is not None:
            storage.save_document(stored_run, document)
    return documents, failures


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
            f"failure_categories={summary['failure_categories']}; "
            "stopping before Strata synthesis with the stored paper pool preserved. "
            "Qwen routing uses Strata Chat Completions "
            "with Pydantic-validated SystemOne answer contracts."
        )
    shortlist = select_candidates(pool_full, batch, request.max_candidates)
    if not shortlist:
        notes.append(_out_triage.triage_coverage_line(
            pool_total=len(pool_full), summary=summary, selected=0, opportunities=0))
        return PipelineResult(
            0, "No scored papers met the importance threshold; no investigation performed.",
            tuple(notes))
    # Full-PDF integration: every shortlisted paper is attempted; no
    # abstract-budget pre-filter and no abstract fallback.
    from radar.config.documents import validate_document_timeout as _validate_doc_timeout

    try:
        document_timeout = _validate_doc_timeout(request.document_timeout_s)
    except ValueError as exc:
        raise PipelineUsageError(str(exc)) from exc
    documents, acquisition_failures = _acquire_shortlist_pdfs(
        request, shortlist, storage=storage, stored_run=stored_run)
    if not documents:
        raise PipelineAnalysisError(
            "No PDF readings completed; no synthesis produced. "
            f"Attempted {len(shortlist)} shortlisted paper(s); "
            "explicit acquisition failures are recorded, no abstract fallback.")
    def retain_readings(readings, failures):
        if storage is not None and stored_run is not None:
            for reading in readings:
                storage.save_document_reading(stored_run, reading)
            for failure in failures:
                storage.save_document_failure(stored_run, failure)

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
            shortlist,
            model=model,
            max_candidates=request.max_candidates,
            analysis_timeout_s=request.analysis_timeout_s,
            max_tokens=request.max_tokens,
            learning_question=request.learning_question,
            disable_thinking=disable_thinking,
            session=session,
            documents=documents,
            document_timeout_s=document_timeout,
            document_model=request.document_model_override,
            on_documents_read=retain_readings,
        )
    except _strata.StrataError as exc:
        raise PipelineAnalysisError(str(exc)) from exc
    readings = list(research.document_readings)
    prior_ids = {f.work_id for f in acquisition_failures}
    merged_failures = list(acquisition_failures)
    merged_failures.extend(f for f in research.document_failures if f.work_id not in prior_ids)
    if not readings or not research.included:
        raise PipelineAnalysisError(
            "No PDF readings completed; no synthesis produced. "
            "Explicit acquisition/reading failures are recorded, no abstract fallback.")
    draft = research.draft
    included = list(research.included)
    try:
        report = attach_evidence(
            draft, included,
            document_readings=readings, document_failures=merged_failures,
            learning_question=research.learning_question,
            source_passages=list(research.source_passages))
        rendered = _out_md.render_markdown(report)
    except Exception as exc:
        raise PipelineAnalysisError(
            f"Report generation failed ({type(exc).__name__}: {exc})."
        ) from exc
    if storage is not None and stored_run is not None:
        storage.save_report(stored_run, report, len(included))
    pages = sum(r.page_count for r in readings)
    chunks = sum(len(r.chunks) for r in readings)
    notes.append(_out_triage.triage_coverage_line(
        pool_total=len(pool_full), summary=summary,
        selected=len(included), opportunities=len(draft.opportunities)))
    notes.append(
        f"radar: pdf coverage attempted={len(shortlist)} read={len(readings)} "
        f"pages={pages} chunks={chunks} synthesized={len(included)} "
        f"failures={len(merged_failures)}")
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
