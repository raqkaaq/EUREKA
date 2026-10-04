"""AI/ML opportunity radar: OpenAlex discovery + local Strata analysis.

Caller-facing seams are re-exported lazily (PEP 562) so importing leaf
modules such as ``radar.schema.papers`` never pulls service dependencies
(httpx, pydantic_ai) into the importing process.
"""

from __future__ import annotations

_LAZY: dict[str, str] = {
    # schema/papers.py
    "CollectedWork": "radar.schema.papers",
    "LocationInfo": "radar.schema.papers",
    "PlannedQuery": "radar.schema.papers",
    "QueryPlan": "radar.schema.papers",
    # schema/opportunities.py
    "EvidenceLink": "radar.schema.opportunities",
    "Opportunity": "radar.schema.opportunities",
    "OpportunityDraft": "radar.schema.opportunities",
    "RadarDraft": "radar.schema.opportunities",
    "RadarReport": "radar.schema.opportunities",
    # config/interests.py
    "RadarProfile": "radar.config.interests",
    "default_profile": "radar.config.interests",
    "build_query_plan": "radar.config.searches",
    # source/openalex.py
    "BASE_URL": "radar.source.openalex",
    "DictTransport": "radar.source.openalex",
    "HttpxTransport": "radar.source.openalex",
    "OpenAlexHttpError": "radar.source.openalex",
    "OpenAlexQuotaError": "radar.source.openalex",
    "RetryingTransport": "radar.source.openalex",
    "build_request": "radar.source.openalex",
    "collect": "radar.source.openalex",
    "default_client": "radar.source.openalex",
    "get_api_key": "radar.source.openalex",
    "normalize_work": "radar.source.openalex",
    "reconstruct_abstract": "radar.source.openalex",
    # processing/ranking.py
    "bound_candidates": "radar.processing.ranking",
    "cheap_score": "radar.processing.ranking",
    "rank_works": "radar.processing.ranking",
    "select_topn": "radar.processing.ranking",
    # processing/evidence.py
    "attach_evidence": "radar.processing.evidence",
    # storage/snapshots.py
    "load_snapshot": "radar.storage.snapshots",
    "refresh_pool": "radar.storage.snapshots",
    # agent/opportunity_analysis.py
    "analyze_candidates": "radar.agent.opportunity_analysis",
    "build_agent": "radar.agent.opportunity_analysis",
    # provider/strata.py (retired-provider exception alias remains compatible)
    "StrataError": "radar.provider.strata",
    "FreeTokenError": "radar.provider.freetoken",
    # output/markdown.py
    "render_markdown": "radar.output.markdown",
    # output/json.py
    "work_to_json": "radar.output.json",
}

__all__ = sorted(_LAZY)


def __getattr__(name: str) -> object:
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module = importlib.import_module(module_name)
    value = getattr(module, name)
    globals()[name] = value
    return value
