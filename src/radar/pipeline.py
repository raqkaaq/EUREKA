"""Sole orchestration boundary: collect, refresh, cached, and analyze flows.

The pipeline coordinates config, source, provider, agent, processing,
storage, and output. It owns transport/session lifetimes, coverage
semantics, and the typed error boundary (usage/source/analysis/storage
errors map to CLI exit codes). No printing, no argparse, no prompts.
"""

from __future__ import annotations

import os as _os
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
from radar.provider import freetoken as _freetoken
from radar.schema.papers import CollectedWork

if TYPE_CHECKING:
    from pydantic_ai.models import Model as _Model


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
    # Injected doubles (tests only; production leaves both None).
    source_override: Any | None = None
    model_override: Any | None = None


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
    if request.from_snapshot is not None:
        if request.refresh_dir is not None:
            raise PipelineUsageError(
                "--from-snapshot cannot be combined with --refresh-dir; "
                "cached analysis never overwrites snapshot state."
            )
        return _run_cached(request)
    return _run_live(request)


def _validate_request(request: PipelineRequest) -> None:
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


def _keywords(request: PipelineRequest) -> list[str]:
    profile = default_profile(lookback_days=request.lookback_days)
    override = clean_keyword_override(
        list(request.keywords) if request.keywords else None
    )
    if override is not None:
        profile = profile.model_copy(update={"keywords": override})
    return profile.keywords


def _collect_live(request: PipelineRequest) -> list[CollectedWork]:
    from radar.source.openalex import (
        HttpxTransport,
        RetryingTransport,
        build_query_plan,
        collect,
    )

    profile = default_profile(lookback_days=request.lookback_days)
    override = clean_keyword_override(
        list(request.keywords) if request.keywords else None
    )
    if override is not None:
        profile = profile.model_copy(update={"keywords": override})
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
        preselected=top, prefix_notes=tuple(notes),
    )


def _analyze_pool(
    request: PipelineRequest,
    *,
    pool_full: list[CollectedWork],
    refresh_collected_at: str | None,
    preselected: list[CollectedWork] | None = None,
    prefix_notes: tuple[str, ...] = (),
) -> PipelineResult:
    from radar.agent import opportunity_analysis as _agent
    from radar.output import markdown as _out_md
    from radar.storage import snapshots as _snapshots

    notes: list[str] = list(prefix_notes)
    ranked = (
        list(preselected)
        if preselected is not None
        else _ranking.select_topn(pool_full, request.max_candidates)
    )
    included = _agent.select_for_prompt(ranked, request.max_candidates)
    if not included:
        return PipelineResult(0, "No candidates collected; nothing to analyze.",
                              tuple(notes))
    disable_thinking = _freetoken.resolve_disable_thinking(request.disable_thinking)
    session = None
    if request.model_override is not None:
        model = request.model_override
    else:
        try:
            config = _freetoken.FreeTokenConfig.resolve(
                base_url=request.base_url, model=request.model)
            session = _freetoken.build_session(config)
        except _freetoken.FreeTokenError as exc:
            raise PipelineAnalysisError(str(exc)) from exc
        model = None
    try:
        draft, _prompt = _agent.analyze_candidates(
            included,
            model=model,
            max_candidates=request.max_candidates,
            analysis_timeout_s=request.analysis_timeout_s,
            max_tokens=request.max_tokens,
            disable_thinking=disable_thinking,
            session=session,
        )
    except _freetoken.FreeTokenError as exc:
        raise PipelineAnalysisError(str(exc)) from exc
    try:
        report = attach_evidence(draft, included)
        rendered = _out_md.render_markdown(report)
    except Exception as exc:
        raise PipelineAnalysisError(
            f"Report generation failed ({type(exc).__name__}: {exc})."
        ) from exc
    notes.append(_out_md.coverage_line(
        len(pool_full), len(included), len(draft.opportunities)))
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
