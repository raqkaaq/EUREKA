"""One-command AI/ML opportunity radar CLI.

Thin boundary: argument parsing, request construction, printing the
pipeline result, and exit codes. All behavior lives behind the
:mod:`radar.pipeline` seam.

- ``uv run python -m radar --collect-only --max-candidates 8`` prints
  bounded real OpenAlex candidates as JSON (no LLM calls).
- ``uv run python -m radar`` collects, analyzes via the private-network Strata
  endpoint (PydanticAI only), and prints a Markdown report.

Exit codes: 0 ok, 2 external-service (OpenAlex) failure, 3 analysis/report/storage
failure (including Strata/Falkor), 4 usage/config error.
"""

from __future__ import annotations

import argparse
import os
import sys

from radar.config.documents import DOCUMENT_ANALYSIS_TIMEOUT_S
from radar.config.runtime import (
    ANALYSIS_MAX_TIMEOUT_S,
    ANALYSIS_TIMEOUT_S,
    ANALYSIS_MAX_TOKENS,
    CLEF_DEFAULT_MODEL,
    CLEF_DEFAULT_OVERALL_TIMEOUT_S,
    CLEF_DEFAULT_REQUEST_TIMEOUT_S,
    DEFAULT_MAX_CANDIDATES,
    MAX_MAX_CANDIDATES,
)
from radar.pipeline import PipelineRequest, run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="radar",
        description="AI/ML opportunity radar (OpenAlex discovery + local Strata analysis).",
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
        default=10.0,
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
        help="Strata base URL (loopback/private LAN only; or set STRATA_BASE_URL (legacy FREETOKEN_BASE_URL)).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Strata model id override (default: STRATA_MODEL env or local /models).",
    )
    parser.add_argument(
        "--storage-dir", default=None, metavar="PATH",
        help="SQLite and local Falkor storage directory (default: RADAR_STORAGE_DIR or data/radar).",
    )
    parser.add_argument(
        "--from-db", action="store_true",
        help="Analyze/inspect the latest SQLite paper pool with zero OpenAlex calls.",
    )
    parser.add_argument(
        "--rebuild-graph", action="store_true",
        help="Rebuild the Falkor graph from SQLite without discovery or model calls.",
    )
    parser.add_argument("--reports", action="store_true",
                        help="List the 20 most recent saved reports, read-only and without model calls.")
    parser.add_argument("--report", metavar="RUN_ID", default=None,
                        help="Reopen a saved report by run ID, or latest; no source/model/graph calls.")
    parser.add_argument(
        "--refresh-dir",
        default=None,
        help="Deprecated alias for --storage-dir; persist the full pool in databases, not JSON.",
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
        default=ANALYSIS_TIMEOUT_S,
        help=f"Overall analysis deadline in seconds, (0, {ANALYSIS_MAX_TIMEOUT_S}] "
        f"(default {ANALYSIS_TIMEOUT_S}).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=ANALYSIS_MAX_TOKENS,
        help=f"Per-run model output cap, 128..8000 (default {ANALYSIS_MAX_TOKENS}).",
    )
    parser.add_argument(
        "--disable-thinking",
        action="store_true",
        help="Opt-in: send the server-specific thinking-disable key "
        "(or set STRATA_DISABLE_THINKING=1). Default omits it; not every "
        "backend supports it.",
    )
    parser.add_argument(
        "--clef-base-url",
        default=None,
        help="Preferred CLEF/SystemOne base URL for full-pool screening "
        "(or set CLEF_BASE_URL; user-owned LAN server, never launched).",
    )
    parser.add_argument(
        "--clef-model",
        default=None,
        help=f"CLEF model id (default {CLEF_DEFAULT_MODEL}; or set CLEF_MODEL).",
    )
    parser.add_argument(
        "--clef-timeout",
        type=float,
        default=CLEF_DEFAULT_REQUEST_TIMEOUT_S,
        help="Per-request CLEF timeout in seconds, (0, 60] (default 10).",
    )
    parser.add_argument(
        "--triage-timeout",
        type=float,
        default=CLEF_DEFAULT_OVERALL_TIMEOUT_S,
        help="Overall CLEF screening deadline in seconds, (0, 300] (default 60).",
    )
    parser.add_argument(
        "--qwen-triage-timeout", type=float, default=None,
        help="Separate Qwen System1 fallback deadline, (0, 3600] seconds "
        "(default: packaged qwen_screening.yaml, 1800).",
    )
    parser.add_argument(
        "--triage-output",
        default=None,
        metavar="PATH",
        help="Deprecated storage-directory alias; screening is saved in SQLite, never a JSON sidecar.",
    )
    parser.add_argument(
        "--document-timeout",
        type=float,
        default=DOCUMENT_ANALYSIS_TIMEOUT_S,
        help="Full-PDF reading deadline in seconds, (0, 3600] (default 900).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse args, run the pipeline, print its result, return its exit code."""
    args = build_parser().parse_args(argv)
    if (args.reports or args.report is not None) and args.collect_only:
        print("radar: error: Report browsing cannot be combined with --collect-only.", file=sys.stderr)
        return 4
    directories = [path for path in (args.storage_dir, args.refresh_dir, args.triage_output) if path is not None]
    if len(set(directories)) > 1:
        print("radar: error: Choose a single database storage directory.", file=sys.stderr)
        return 4
    if args.from_snapshot is not None and args.refresh_dir is not None:
        print("radar: error: --from-snapshot cannot be combined with --refresh-dir; use --storage-dir for import.",
              file=sys.stderr)
        return 4
    storage_dir = directories[0] if directories else os.environ.get("RADAR_STORAGE_DIR", "data/radar")
    if not storage_dir.strip():
        print("radar: error: Storage directory must not be empty.", file=sys.stderr)
        return 4
    request = PipelineRequest(
        mode="collect" if args.collect_only else "analyze",
        max_candidates=args.max_candidates,
        lookback_days=args.lookback_days,
        timeout_s=args.timeout,
        keywords=tuple(args.keywords) if args.keywords else None,
        from_snapshot=args.from_snapshot,
        analysis_timeout_s=args.analysis_timeout,
        max_tokens=args.max_tokens,
        disable_thinking=args.disable_thinking,
        base_url=args.base_url,
        model=args.model,
        clef_base_url=args.clef_base_url,
        clef_model=args.clef_model,
        clef_timeout_s=args.clef_timeout,
        triage_timeout_s=args.triage_timeout,
        qwen_triage_timeout_s=args.qwen_triage_timeout,
        storage_dir=storage_dir,
        from_database=args.from_db,
        rebuild_graph=args.rebuild_graph,
        list_reports=args.reports,
        report_id=args.report,
        document_timeout_s=args.document_timeout,
    )
    result = run(request)
    for note in result.stderr_notes:
        print(note, file=sys.stderr)
    if result.stdout:
        print(result.stdout)
    return result.exit_code
