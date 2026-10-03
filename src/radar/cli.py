"""One-command AI/ML opportunity radar CLI.

- ``uv run python -m radar --collect-only --max-candidates 8`` prints
  bounded real OpenAlex candidates as JSON (no LLM calls).
- ``uv run python -m radar`` collects, analyzes via the private-network FreeToken
  endpoint (PydanticAI only), and prints a Markdown report.

Exit codes: 0 ok, 2 external-service (OpenAlex) failure, 3 analysis/report
failure (including FreeToken), 4 usage/config error.
"""

from __future__ import annotations

import argparse
import json
import math as _math
import sys

from radar import freetoken as _freetoken
from radar import refresh as _refresh
from radar.analyze import analyze_candidates
from radar.models import CollectedWork
from radar.openalex import (
    DEFAULT_RETRY_AFTER_S,
    MAX_RETRY_AFTER_S,
    MAX_TOTAL_WORKS,
    HttpxTransport,
    RetryingTransport,
    build_query_plan,
    collect,
)
from radar.profile import default_profile
from radar.prompt import MAX_CANDIDATES_IN_PROMPT, bound_candidates
from radar.report import attach_evidence, render_markdown

# Re-exported so existing callers keep one import site for retry policy.
__all__ = ["RetryingTransport"]

DEFAULT_MAX_CANDIDATES = 12
MAX_MAX_CANDIDATES = 25
DEFAULT_TIMEOUT_S = 10.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="radar",
        description="AI/ML opportunity radar (OpenAlex discovery + local FreeToken analysis).",
    )
    parser.add_argument(
        "--collect-only",
        action="store_true",
        help="Print bounded OpenAlex candidates as JSON and skip LLM analysis.",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=DEFAULT_MAX_CANDIDATES,
        help=f"Candidate bound 1..{MAX_MAX_CANDIDATES} (default {DEFAULT_MAX_CANDIDATES}).",
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=90,
        help="OpenAlex recency window in days, 1..3650 (default 90).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT_S,
        help="Per-request OpenAlex timeout in seconds, (0, 30] (default 10).",
    )
    parser.add_argument(
        "--keywords",
        nargs="+",
        default=None,
        help="Override default AI/ML keywords (default profile is AI/ML + behavioral/economic).",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="FreeToken base URL (loopback/private LAN only; or set FREETOKEN_BASE_URL).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="FreeToken model id override (default: FREETOKEN_MODEL env or local /models).",
    )
    parser.add_argument(
        "--refresh-dir",
        default=None,
        help="Persist the FULL normalized pool as one atomic metadata-only "
        "snapshot (snapshot.json) in PATH, independent of FreeToken. "
        "Combine with --collect-only for unattended refresh (no LLM).",
    )
    parser.add_argument(
        "--from-snapshot",
        default=None,
        metavar="PATH",
        help="Analyze cached snapshot metadata with zero OpenAlex calls. "
        "Rejects --refresh-dir (no silent overwrite).",
    )
    parser.add_argument(
        "--analysis-timeout",
        type=float,
        default=_freetoken.ANALYSIS_TIMEOUT_S,
        help=f"Overall analysis deadline in seconds, (0, {_freetoken.ANALYSIS_MAX_TIMEOUT_S}] "
        f"(default {_freetoken.ANALYSIS_TIMEOUT_S}).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=_freetoken.ANALYSIS_MAX_TOKENS,
        help=f"Per-run model output cap, 128..8000 (default {_freetoken.ANALYSIS_MAX_TOKENS}).",
    )
    parser.add_argument(
        "--disable-thinking",
        action="store_true",
        help="Opt-in: send the server-specific thinking-disable key "
        "(or set FREETOKEN_DISABLE_THINKING=1). Default omits it; not every "
        "backend supports it.",
    )
    return parser


def _fail(message: str, code: int) -> int:
    print(f"radar: error: {message}", file=sys.stderr)
    return code


def _analyze_timeout(value: float) -> float:
    """Validate the overall analysis deadline: finite seconds in (0, 300]."""
    try:
        timeout = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"--analysis-timeout must be within (0, {_freetoken.ANALYSIS_MAX_TIMEOUT_S}]s"
        ) from exc
    if not _math.isfinite(timeout) or not (
        0 < timeout <= _freetoken.ANALYSIS_MAX_TIMEOUT_S
    ):
        raise ValueError(
            f"--analysis-timeout must be within (0, {_freetoken.ANALYSIS_MAX_TIMEOUT_S}]s"
        )
    return timeout


def work_to_json(work: CollectedWork) -> dict:
    return {
        "openalex_id": work.openalex_id,
        "title": work.title,
        "abstract": work.abstract[:2000],
        "publication_year": work.publication_year,
        "doi": work.doi,
        "primary_url": work.primary_url,
        "cited_by_count": work.cited_by_count,
        "score": work.score,
        "matched_queries": work.matched_queries,
        "query_kinds": work.query_kinds,
    }


def collect_pool(
    lookback_days: int,
    timeout: float,
    keywords: list[str] | None,
) -> list[CollectedWork]:
    """Collect the full bounded pool via OpenAlex (sole discovery API).

    The whole bounded query plan is executed against a hard
    candidate-pool cap (200). Callers apply ``--max-candidates`` as a final
    slice; refresh mode persists this full pool.
    """
    if not (1 <= lookback_days <= 3650):
        raise ValueError("--lookback-days must be within 1..3650")
    if not (0 < timeout <= 30):
        raise ValueError("--timeout must be within (0, 30]")
    profile = default_profile(lookback_days=lookback_days)
    if keywords:
        cleaned = [k.strip() for k in keywords if k and k.strip()]
        if not cleaned:
            raise ValueError("--keywords must contain at least one non-empty term")
        profile = profile.model_copy(update={"keywords": cleaned[:20]})
    # Five default AI/ML themes plus the newest-first query fit the hard
    # six-request plan bound, so no default theme is silently omitted.
    plan = build_query_plan(profile, max_queries=6)
    base = HttpxTransport()
    try:
        return collect(
            plan,
            RetryingTransport(base),
            keywords_for_scoring=profile.keywords,
            timeout=timeout,
            max_total=MAX_TOTAL_WORKS,
        )
    finally:
        base.close()


def collect_candidates(
    max_candidates: int,
    lookback_days: int,
    timeout: float,
    keywords: list[str] | None,
) -> list[CollectedWork]:
    """Collect bounded candidates via OpenAlex (sole discovery API).

    The whole bounded query plan is executed against a hard
    candidate-pool cap (200); ``max_candidates`` only bounds the final
    ranked output slice so later query branches still contribute when the
    output bound is small.
    """
    if not (1 <= max_candidates <= MAX_MAX_CANDIDATES):
        raise ValueError(f"--max-candidates must be within 1..{MAX_MAX_CANDIDATES}")
    return collect_pool(lookback_days, timeout, keywords)[:max_candidates]


def _snapshot_age_note(collected_at_utc: str) -> str:
    """Human-readable staleness disclosure for a cached snapshot timestamp."""
    import datetime as _dt

    try:
        taken = _dt.datetime.fromisoformat(collected_at_utc)
        if taken.tzinfo is None:
            taken = taken.replace(tzinfo=_dt.timezone.utc)
        age = _dt.datetime.now(_dt.timezone.utc) - taken
        seconds = max(0, int(age.total_seconds()))
    except (ValueError, TypeError):
        return f"snapshot collected at {collected_at_utc} (age unknown)"
    if seconds < 90:
        age_note = f"{seconds}s old"
    elif seconds < 5400:
        age_note = f"{seconds // 60}m old"
    else:
        age_note = f"{seconds // 3600}h old"
    return f"snapshot collected at {collected_at_utc} ({age_note})"


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if not (1 <= args.max_candidates <= MAX_MAX_CANDIDATES):
        return _fail(
            f"--max-candidates must be within 1..{MAX_MAX_CANDIDATES}", 4
        )
    try:
        analysis_timeout = _analyze_timeout(args.analysis_timeout)
    except ValueError as exc:
        return _fail(str(exc), 4)
    if not (isinstance(args.max_tokens, int) and 128 <= args.max_tokens <= 8000):
        return _fail("--max-tokens must be an integer within 128..8000", 4)
    disable_thinking = _freetoken.resolve_disable_thinking(args.disable_thinking)

    cached_works: list[CollectedWork] | None = None
    cached_meta: dict | None = None
    if args.from_snapshot:
        # Cached synthesis: zero OpenAlex calls; never combined with refresh.
        if args.refresh_dir:
            return _fail(
                "--from-snapshot cannot be combined with --refresh-dir; "
                "cached analysis never overwrites snapshot state.",
                4,
            )
        import os as _os

        if not _os.path.exists(args.from_snapshot):
            return _fail(f"snapshot not found: {args.from_snapshot}", 4)
        try:
            cached_works, cached_meta = _refresh.load_snapshot(args.from_snapshot)
        except _refresh.RefreshError as exc:
            return _fail(f"invalid snapshot source ({exc}). No model was called.", 3)
        pool = _refresh.select_topn(cached_works, args.max_candidates)
        print(_snapshot_age_note(str(cached_meta["collected_at_utc"])),
              file=sys.stderr)
    else:
        pool = None
        try:
            if args.refresh_dir:
                # Refresh persists the full pool, not just the topN slice.
                pool = collect_pool(args.lookback_days, args.timeout, args.keywords)
            else:
                pool = collect_candidates(
                    args.max_candidates, args.lookback_days, args.timeout, args.keywords
                )
        except ValueError as exc:
            return _fail(str(exc), 4)
        except Exception as exc:
            return _fail(
                f"OpenAlex collection failed ({type(exc).__name__}: {exc}). "
                "Check network access to https://api.openalex.org and retry; "
                "per-request timeouts are bounded by --timeout.",
                2,
            )
        assert pool is not None

    refresh_collected_at: str | None = None
    if args.refresh_dir:
        try:
            summary = _refresh.refresh_pool(pool, args.refresh_dir)
            refresh_collected_at = str(summary["collected_at_utc"])
        except _refresh.RefreshError as exc:
            return _fail(
                f"refresh failed ({exc}). Previous snapshot preserved.",
                3,
            )

    if args.collect_only:
        if cached_meta is not None:
            print(json.dumps({
                "snapshot": args.from_snapshot,
                "collected_at_utc": cached_meta["collected_at_utc"],
                "coverage": cached_meta["coverage"],
                "discovery": cached_meta["discovery"],
                "disclosure": _refresh.DISCLOSURE,
                "selected": len(pool),
                "works": [work_to_json(w) for w in pool],
            }, indent=2))
        elif args.refresh_dir:
            print(json.dumps(summary, indent=2))
        else:
            print(
                json.dumps(
                    [work_to_json(w) for w in pool[: args.max_candidates]],
                    indent=2,
                )
            )
        return 0

    works = pool[: args.max_candidates]
    if not works:
        print("No candidates collected; nothing to analyze.")
        return 0

    # Validate private-network config early for a fast, clear model error, then
    # resolve the model (FREETOKEN_MODEL env or local /models endpoint).
    # The session owns its HTTP client; it is closed deterministically below.
    try:
        config = _freetoken.FreeTokenConfig.resolve(base_url=args.base_url, model=args.model)
        session = _freetoken.build_session(config)
    except _freetoken.FreeTokenError as exc:
        return _fail(str(exc), 3)

    try:
        draft, _prompt = analyze_candidates(
            works,
            model=None,
            max_candidates=args.max_candidates,
            analysis_timeout_s=analysis_timeout,
            max_tokens=args.max_tokens,
            disable_thinking=disable_thinking,
            session=session,
        )
    except _freetoken.FreeTokenError as exc:
        return _fail(str(exc), 3)
    # Evidence must resolve against the same bounded candidate set supplied
    # to the LLM (shared helper guarantees no hardcoded-slice drift).
    evidence_candidates = bound_candidates(works, args.max_candidates)
    # Explicit coverage: only the bounded selection ever reaches the LLM,
    # never the whole pool.
    print(
        f"radar: coverage selected={len(evidence_candidates)} "
        f"analyzed={len(draft.opportunities)} "
        f"(pool={len(pool)}; bounded discovery sample, not all of OpenAlex)",
        file=sys.stderr,
    )
    try:
        report = attach_evidence(draft, evidence_candidates)
        print(render_markdown(report))
    except Exception as exc:
        return _fail(
            f"Report generation failed ({type(exc).__name__}: {exc}).",
            3,
        )
    if args.refresh_dir:
        # Best-effort, generation-aware: record LLM counts only in the
        # snapshot this run wrote. The metadata snapshot stands even when
        # this patch cannot run.
        try:
            _refresh.update_llm_coverage(
                args.refresh_dir,
                llm_selected=len(evidence_candidates),
                llm_analyzed=len(draft.opportunities),
                expect_collected_at=refresh_collected_at,
            )
        except _refresh.RefreshError as exc:
            print(f"radar: warning: could not update refresh coverage ({exc})",
                  file=sys.stderr)
    return 0
