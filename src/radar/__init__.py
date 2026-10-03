"""OpenAlex collection seam for an AI/ML-centered cross-domain radar.

Scope: query-plan representation, OpenAlex request construction,
abstract reconstruction, normalization, dedup, provenance, and
deterministic cheap pre-scoring. No LLM calls. OpenAlex is the sole
scholarly discovery API here; arXiv may appear only as a location
string returned by OpenAlex, never as a queried API.
"""

from radar.analyze import analyze_candidates, build_agent
from radar.models import (
    CollectedWork,
    EvidenceLink,
    LocationInfo,
    Opportunity,
    OpportunityDraft,
    PlannedQuery,
    QueryPlan,
    RadarDraft,
    RadarProfile,
    RadarReport,
)
from radar.openalex import (
    BASE_URL,
    HttpxTransport,
    RetryingTransport,
    build_query_plan,
    build_request,
    cheap_score,
    collect,
    get_api_key,
    normalize_work,
    reconstruct_abstract,
)
from radar.profile import default_profile
from radar.prompt import build_prompt
from radar.report import attach_evidence, render_markdown

__all__ = [
    "BASE_URL",
    "CollectedWork",
    "EvidenceLink",
    "HttpxTransport",
    "LocationInfo",
    "Opportunity",
    "OpportunityDraft",
    "PlannedQuery",
    "QueryPlan",
    "RadarDraft",
    "RadarProfile",
    "RadarReport",
    "RetryingTransport",
    "analyze_candidates",
    "attach_evidence",
    "build_agent",
    "build_prompt",
    "build_query_plan",
    "build_request",
    "cheap_score",
    "collect",
    "default_profile",
    "get_api_key",
    "normalize_work",
    "reconstruct_abstract",
    "render_markdown",
]
