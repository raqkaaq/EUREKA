"""One-command AI/ML opportunity radar CLI.

Thin boundary: argument parsing, request construction, printing the
pipeline result, and exit codes. All behavior lives behind the
:mod:`radar.pipeline` seam.

- ``uv run python -m radar --collect-only --max-candidates 8`` prints
  bounded real OpenAlex candidates as JSON (no LLM calls).
- ``uv run python -m radar`` collects, analyzes via the private-network FreeToken
  endpoint (PydanticAI only), and prints a Markdown report.

Exit codes: 0 ok, 2 external-service (OpenAlex) failure, 3 analysis/report
failure (including FreeToken), 4 usage/config error.
"""

from __future__ import annotations

import argparse
import sys

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
        "(or set FREETOKEN_DISABLE_THINKING=1). Default omits it; not every "
        "backend supports it.",
    )
    parser.add_argument(
        "--clef-base-url",
        default=None,
        help="CLEF/SystemOne base URL for mandatory full-pool screening "
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
        "--triage-output",
        default=None,
        metavar="PATH",
        help="Write an atomic triage sidecar JSON beside the run; strict "
        "v1 metadata snapshots are never mutated.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse args, run the pipeline, print its result, return its exit code."""
    args = build_parser().parse_args(argv)
    request = PipelineRequest(
        mode="collect" if args.collect_only else "analyze",
        max_candidates=args.max_candidates,
        lookback_days=args.lookback_days,
        timeout_s=args.timeout,
        keywords=tuple(args.keywords) if args.keywords else None,
        refresh_dir=args.refresh_dir,
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
        triage_output=args.triage_output,
    )
    result = run(request)
    for note in result.stderr_notes:
        print(note, file=sys.stderr)
    if result.stdout:
        print(result.stdout)
    return result.exit_code
