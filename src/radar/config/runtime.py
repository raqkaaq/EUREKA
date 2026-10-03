"""Numeric runtime bounds (service-free: stdlib only).

Single home for every validated limit so CLI, pipeline, source, agent,
and storage share one definition. Validation helpers raise ValueError;
the pipeline maps those to usage errors (exit 4).
"""

from __future__ import annotations

import math as _math

# --- CLI selection bounds ---
DEFAULT_MAX_CANDIDATES = 12
MAX_MAX_CANDIDATES = 25

# --- OpenAlex discovery bounds ---
DEFAULT_TIMEOUT_S = 10.0
MAX_TIMEOUT_S = 30.0
MAX_TOTAL_WORKS = 200
MAX_ABSTRACT_CHARS = 20000
RESPONSE_READ_LIMIT = 2_000_000
MAX_429_RETRIES = 2
MAX_RETRY_AFTER_S = 45.0
DEFAULT_RETRY_AFTER_S = 2.0
QUOTA_BODY_READ_LIMIT = 4096
LONG_QUOTA_RESET_S = 3600.0

# --- Query-plan bounds ---
MAX_QUERIES = 6
MAX_PER_PAGE = 50
MAX_KEYWORDS = 20
MAX_TERM_CHARS = 300

# --- Analysis bounds (proven live 2026-10-03) ---
ANALYSIS_TIMEOUT_S = 90.0
ANALYSIS_MAX_TIMEOUT_S = 300.0
ANALYSIS_MAX_TOKENS = 2000
MIN_ANALYSIS_TOKENS = 128
MAX_ANALYSIS_TOKENS = 8000
ANALYSIS_REQUEST_TIMEOUT_S = 60.0
ANALYSIS_REQUEST_LIMIT = 2
# Single output-validation retry: real prompts sporadically emit malformed
# envelopes (live categories: ignore.0 string_type, opportunities list_type).
# Bounded by request_limit=2 (at most 2 provider requests) and the overall
# deadline; network-level SDK retries are separate and unchanged.
ANALYSIS_RETRIES = 1

# --- Prompt bounds ---
MAX_CANDIDATES_IN_PROMPT = 25
MAX_ABSTRACT_IN_PROMPT = 600
MAX_TITLE_IN_PROMPT = 200
MAX_PROMPT_CHARS = 12_000
MAX_ANALYSIS_OPPORTUNITIES = 2

# --- Specialist research team (three bounded contributions + synthesis) ---
SPECIALIST_CONCURRENCY = 2
SPECIALIST_MAX_TOKENS = 1000
SPECIALIST_MAX_REPORT_CHARS = 1000
SPECIALIST_CONTEXT_CHARS = 3500

# --- Analysis output schema caps ---
MAX_OPPORTUNITIES = 5
MAX_EVIDENCE_PER_OPP = 3

# --- CLEF triage bounds (mandatory main routing stage; server not yet running) ---
CLEF_DEFAULT_MODEL = "clef-flash"
CLEF_DEFAULT_REQUEST_TIMEOUT_S = 10.0
CLEF_DEFAULT_OVERALL_TIMEOUT_S = 60.0
CLEF_DEFAULT_CONCURRENCY = 4
CLEF_MAX_TEXT_JSON_BYTES = 65536
CLEF_MAX_REQUEST_TIMEOUT_S = 60.0
CLEF_MAX_OVERALL_TIMEOUT_S = 300.0
CLEF_MAX_CONCURRENCY = 16


def validate_analysis_timeout(value: float) -> float:
    """Overall analysis deadline: finite seconds in (0, 300]."""
    try:
        timeout = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"--analysis-timeout must be within (0, {ANALYSIS_MAX_TIMEOUT_S}]s"
        ) from exc
    if not _math.isfinite(timeout) or not (0 < timeout <= ANALYSIS_MAX_TIMEOUT_S):
        raise ValueError(
            f"--analysis-timeout must be within (0, {ANALYSIS_MAX_TIMEOUT_S}]s"
        )
    return timeout


def validate_max_tokens(value: int) -> int:
    """Per-run model output cap within 128..8000."""
    try:
        tokens = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"--max-tokens must be an integer within "
            f"{MIN_ANALYSIS_TOKENS}..{MAX_ANALYSIS_TOKENS}"
        ) from exc
    if not MIN_ANALYSIS_TOKENS <= tokens <= MAX_ANALYSIS_TOKENS:
        raise ValueError(
            f"--max-tokens must be an integer within "
            f"{MIN_ANALYSIS_TOKENS}..{MAX_ANALYSIS_TOKENS}"
        )
    return tokens


def validate_openalex_timeout(value: float) -> float:
    """Per-request OpenAlex timeout in (0, 30]."""
    try:
        timeout = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"--timeout must be within (0, {MAX_TIMEOUT_S}]") from exc
    if not _math.isfinite(timeout) or not (0 < timeout <= MAX_TIMEOUT_S):
        raise ValueError(f"--timeout must be within (0, {MAX_TIMEOUT_S}]")
    return float(timeout)


def validate_clef_request_timeout(value: float) -> float:
    """Per-request CLEF timeout in (0, 60]."""
    try:
        timeout = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"--clef-timeout must be within (0, {CLEF_MAX_REQUEST_TIMEOUT_S}]"
        ) from exc
    if not _math.isfinite(timeout) or not (0 < timeout <= CLEF_MAX_REQUEST_TIMEOUT_S):
        raise ValueError(
            f"--clef-timeout must be within (0, {CLEF_MAX_REQUEST_TIMEOUT_S}]"
        )
    return float(timeout)


def validate_clef_overall_timeout(value: float) -> float:
    """Overall CLEF batch timeout in (0, 300]."""
    try:
        timeout = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"--triage-timeout must be within (0, {CLEF_MAX_OVERALL_TIMEOUT_S}]"
        ) from exc
    if not _math.isfinite(timeout) or not (0 < timeout <= CLEF_MAX_OVERALL_TIMEOUT_S):
        raise ValueError(
            f"--triage-timeout must be within (0, {CLEF_MAX_OVERALL_TIMEOUT_S}]"
        )
    return float(timeout)


__all__ = [
    "ANALYSIS_MAX_TIMEOUT_S",
    "ANALYSIS_MAX_TOKENS",
    "ANALYSIS_REQUEST_LIMIT",
    "ANALYSIS_REQUEST_TIMEOUT_S",
    "ANALYSIS_RETRIES",
    "ANALYSIS_TIMEOUT_S",
    "DEFAULT_MAX_CANDIDATES",
    "DEFAULT_RETRY_AFTER_S",
    "DEFAULT_TIMEOUT_S",
    "LONG_QUOTA_RESET_S",
    "MAX_429_RETRIES",
    "MAX_ABSTRACT_CHARS",
    "MAX_ANALYSIS_OPPORTUNITIES",
    "MAX_ANALYSIS_TOKENS",
    "MAX_CANDIDATES_IN_PROMPT",
    "MAX_ABSTRACT_IN_PROMPT",
    "MAX_EVIDENCE_PER_OPP",
    "MAX_KEYWORDS",
    "MAX_MAX_CANDIDATES",
    "MAX_OPPORTUNITIES",
    "MAX_PER_PAGE",
    "MAX_PROMPT_CHARS",
    "MAX_QUERIES",
    "MAX_RETRY_AFTER_S",
    "MAX_TERM_CHARS",
    "MAX_TITLE_IN_PROMPT",
    "MAX_TIMEOUT_S",
    "MAX_TOTAL_WORKS",
    "MIN_ANALYSIS_TOKENS",
    "QUOTA_BODY_READ_LIMIT",
    "RESPONSE_READ_LIMIT",
    "SPECIALIST_CONCURRENCY",
    "SPECIALIST_CONTEXT_CHARS",
    "SPECIALIST_MAX_REPORT_CHARS",
    "SPECIALIST_MAX_TOKENS",
    "validate_analysis_timeout",
    "validate_clef_overall_timeout",
    "validate_clef_request_timeout",
    "validate_max_tokens",
    "validate_openalex_timeout",
]
