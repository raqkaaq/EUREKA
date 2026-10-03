"""Qwen endpoint configuration for the existing native SystemOne client.

No prompt conversion, chat interface, or separate scoring algorithm. The
CLEF-compatible HTTPX client sends/validates the identical SystemOne contract
at the fallback endpoint, using the model id served by that endpoint.
"""

from __future__ import annotations

import os

import httpx

from radar.config.interests import RadarProfile
from radar.provider.clef import ClefConfig, ClefError, screen_works as _screen
from radar.schema.papers import CollectedWork
from radar.schema.triage import TriageBatch


def resolve_config(
    *, base_url: str | None = None, model: str | None = None,
    freetoken_base_url: str | None = None, freetoken_model: str | None = None,
    request_timeout_s: float = 10.0, overall_timeout_s: float = 60.0,
) -> ClefConfig:
    """Explicit SystemOne overrides, then configured FreeToken endpoint/model.

    No guessed host, model name, model discovery, or contact with a service
    during resolution. Invalid configuration is not a reason to bypass safety.
    """
    base = (base_url if base_url is not None else
            (os.environ.get("QWEN_SYSTEMONE_BASE_URL", "").strip()
             or freetoken_base_url or os.environ.get("FREETOKEN_BASE_URL", ""))).strip()
    name = (model if model is not None else
            (os.environ.get("QWEN_SYSTEMONE_MODEL", "").strip()
             or freetoken_model or os.environ.get("FREETOKEN_MODEL", ""))).strip()
    if not base or not name or len(name) > 200:
        raise ValueError(
            "CLEF is unavailable and Qwen SystemOne fallback is not configured; "
            "set QWEN_SYSTEMONE_BASE_URL and QWEN_SYSTEMONE_MODEL, or the "
            "FREETOKEN_BASE_URL/FREETOKEN_MODEL equivalents. The Qwen server "
            "must support native POST /v1/systemone; chat-only is not compatible."
        )
    try:
        return ClefConfig.resolve(
            base_url=base, model=name, request_timeout_s=request_timeout_s,
            overall_timeout_s=overall_timeout_s)
    except (ClefError, ValueError):
        raise ValueError(
            "Qwen SystemOne endpoint/model configuration is invalid; use a "
            "plain private-LAN http(s) URL with an empty path or /v1, "
            "no credentials, query, or fragment, and valid routing bounds."
        ) from None


def screen_works(
    works: list[CollectedWork], profile: RadarProfile, *, config: ClefConfig,
    fallback_reason: str, transport: httpx.BaseTransport | None = None,
) -> TriageBatch:
    """Reuse exactly the native request, result schema, bounds and lifecycle."""
    batch = _screen(works, profile, config=config, transport=transport)
    return TriageBatch(
        model_id=batch.model_id, rubric_version=batch.rubric_version,
        backend="qwen", probability_kind="native_noul",
        fallback_reason=fallback_reason, results=batch.results)
