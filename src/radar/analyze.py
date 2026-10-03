"""PydanticAI analysis wiring: candidates -> structured RadarDraft.

All LLM inference goes through PydanticAI ``Agent`` with a Pydantic
``output_type``. Tests inject a ``TestModel``/``FunctionModel`` via the
``model`` parameter; production builds the private-network FreeToken model via
:mod:`radar.freetoken`. No direct OpenAI SDK calls anywhere on this path.

Live-reliability bounds (proven 2026-10-03): the overall run carries a hard
deadline enforced by cancelling async ``Agent.run`` (blocking ``run_sync``
cannot be interrupted), with per-run ``max_tokens``, a small request limit,
and zero validation retries. The thinking-disable key is opt-in only.
"""

from __future__ import annotations

import asyncio as _asyncio
import math as _math
from typing import TYPE_CHECKING

from radar import freetoken as _freetoken
from radar.models import CollectedWork, RadarDraft
from radar.prompt import build_prompt

if TYPE_CHECKING:
    from pydantic_ai.models import Model as _Model
    from pydantic_ai.settings import ModelSettings as _ModelSettings
    from pydantic_ai.usage import UsageLimits as _UsageLimits


def _check_timeout(value: float, name: str = "--analysis-timeout") -> float:
    try:
        timeout = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be within (0, {_freetoken.ANALYSIS_MAX_TIMEOUT_S}]s") from exc
    if not _math.isfinite(timeout) or not (
        0 < timeout <= _freetoken.ANALYSIS_MAX_TIMEOUT_S
    ):
        raise ValueError(f"{name} must be within (0, {_freetoken.ANALYSIS_MAX_TIMEOUT_S}]s")
    return timeout


def run_bounds(
    disable_thinking: bool = False,
    max_tokens: int = _freetoken.ANALYSIS_MAX_TOKENS,
    request_timeout_s: float = _freetoken.ANALYSIS_REQUEST_TIMEOUT_S,
    request_limit: int = _freetoken.ANALYSIS_REQUEST_LIMIT,
    analysis_timeout_s: float = _freetoken.ANALYSIS_TIMEOUT_S,
) -> tuple["_ModelSettings", "_UsageLimits", int]:
    """Build the bounded per-run settings: ``(model_settings, usage_limits, retries)``.

    Pure seam: no network, no model. ``request_timeout_s`` must fit inside
    the remaining overall budget; the thinking-disable key is included only
    when explicitly opted in.
    """
    analysis_timeout = _check_timeout(analysis_timeout_s)
    try:
        tokens = int(max_tokens)
    except (TypeError, ValueError) as exc:
        raise ValueError("--max-tokens must be an integer within 128..8000") from exc
    if not 128 <= tokens <= 8000:
        raise ValueError("--max-tokens must be an integer within 128..8000")
    try:
        limit = int(request_limit)
    except (TypeError, ValueError) as exc:
        raise ValueError("request_limit must be 1 or 2") from exc
    if limit not in (1, 2):
        raise ValueError("request_limit must be 1 or 2")
    try:
        per_request = float(request_timeout_s)
    except (TypeError, ValueError) as exc:
        raise ValueError("request timeout must be within (0, overall deadline]s") from exc
    if not _math.isfinite(per_request) or not (0 < per_request <= analysis_timeout):
        raise ValueError("request timeout must be within (0, overall deadline]s")
    settings: _ModelSettings = {"max_tokens": tokens, "timeout": per_request}  # type: ignore[typeddict-item]
    if disable_thinking:
        settings["extra_body"] = _freetoken.thinking_extra_body()  # type: ignore[typeddict-unknown-key]
    from pydantic_ai.usage import UsageLimits

    return settings, UsageLimits(request_limit=limit), _freetoken.ANALYSIS_RETRIES


def build_agent(model: "_Model | None" = None, output_type: type[RadarDraft] = RadarDraft):
    """Build the analysis agent.

    ``model`` is a PydanticAI model instance (injectable for tests). When
    ``None``, the private-network FreeToken model is resolved and built (may raise
    :class:`freetoken.FreeTokenError` with an actionable message).
    """
    from pydantic_ai import Agent

    if model is None:
        config = _freetoken.FreeTokenConfig.resolve()
        model = _freetoken.build_model(config)
    return Agent(model, output_type=output_type, retries=_freetoken.ANALYSIS_RETRIES)


def _validation_categories(exc: BaseException) -> list[str]:
    """Summarize output-validation failures as ``loc: error-type`` entries.

    Only field paths and error types are recorded -- never offending values,
    prompts, or secrets -- so the summary is safe for errors and logs.
    """
    categories: list[str] = []
    node: BaseException | None = exc
    seen = 0
    while node is not None and seen < 8:
        if type(node).__name__ == "ValidationError":
            errors = getattr(node, "errors", None)
            if callable(errors):
                try:
                    for entry in errors(include_url=False):
                        if isinstance(entry, dict):
                            loc = ".".join(str(p) for p in entry.get("loc", ()))
                            categories.append(f"{loc or '?'}: {entry.get('type', '?')}")
                except Exception:
                    pass
        node = node.__cause__
        seen += 1
    return categories[:12]


def _actionable(exc: Exception, disable_thinking: bool) -> _freetoken.FreeTokenError:
    text = str(exc)
    lowered = text.lower()
    if disable_thinking and (
        "400" in text or "bad request" in lowered or "extra_body" in lowered
    ):
        return _freetoken.FreeTokenError(
            f"FreeToken server rejected the optional thinking-disable key "
            f"({type(exc).__name__}: {exc}). Rerun without --disable-thinking; "
            "that server-specific key is not supported by every backend."
        )
    categories = _validation_categories(exc)
    detail = (
        f" Output failed validation ({'; '.join(categories)})"
        if categories
        else ""
    )
    return _freetoken.FreeTokenError(
        f"FreeToken inference failed ({type(exc).__name__}: {exc}).{detail} "
        "Check that your user-owned FreeToken server is serving "
        "OpenAI-compatible Chat Completions at the configured private-network "
        "endpoint from FREETOKEN_BASE_URL/--base-url and that "
        "FREETOKEN_MODEL names a served model."
    )


async def analyze_candidates_async(
    candidates: list[CollectedWork],
    model: "_Model | None" = None,
    max_candidates: int = 12,
    analysis_timeout_s: float = _freetoken.ANALYSIS_TIMEOUT_S,
    max_tokens: int = _freetoken.ANALYSIS_MAX_TOKENS,
    request_limit: int = _freetoken.ANALYSIS_REQUEST_LIMIT,
    disable_thinking: bool = False,
    session: "_freetoken.FreeTokenSession | None" = None,
) -> tuple[RadarDraft, str]:
    """Run the radar analysis under a hard overall deadline.

    The deadline cancels the async ``Agent.run`` itself (a blocked sync run
    could not be interrupted). When ``session`` is given, its model is used
    and its HTTP client is closed in the same event loop once the run
    settles (including on cancellation), so cleanup is deterministic.
    Raises :class:`freetoken.FreeTokenError` on deadline breach or
    inference failure.
    """
    settings, limits, retries = run_bounds(
        disable_thinking=disable_thinking,
        max_tokens=max_tokens,
        request_timeout_s=min(
            _freetoken.ANALYSIS_REQUEST_TIMEOUT_S, float(analysis_timeout_s)
        ),
        analysis_timeout_s=analysis_timeout_s,
        request_limit=request_limit,
    )
    prompt = build_prompt(candidates, max_candidates=max_candidates)
    if session is not None:
        model = session.model
    agent = build_agent(model)
    try:
        async with _asyncio.timeout(float(analysis_timeout_s)):
            result = await agent.run(
                prompt,
                model_settings=settings,  # type: ignore[arg-type]
                usage_limits=limits,
                retries=retries,
            )
    except (TimeoutError, _asyncio.CancelledError) as exc:
        raise _freetoken.FreeTokenError(
            f"FreeToken analysis exceeded the overall {analysis_timeout_s}s "
            f"deadline ({type(exc).__name__}); no partial report was produced. "
            "Retry with fewer candidates, a smaller --max-tokens, or a larger "
            "--analysis-timeout."
        ) from exc
    except _freetoken.FreeTokenError:
        raise
    except Exception as exc:
        raise _actionable(exc, disable_thinking) from exc
    finally:
        if session is not None:
            # Same-loop deterministic cleanup, including on cancellation.
            # A close failure must not mask the analysis outcome.
            try:
                await session.http_client.aclose()
            except Exception:
                pass
    output = result.output
    if not isinstance(output, RadarDraft):
        # Defensive: Agent(output_type=RadarDraft) must return RadarDraft;
        # coerce when a test double returns a mapping.
        try:
            output = RadarDraft.model_validate(output)
        except Exception as exc:
            raise _freetoken.FreeTokenError(
                f"Model returned output that does not validate as RadarDraft: {exc}"
            ) from exc
    return output, prompt


def analyze_candidates(
    candidates: list[CollectedWork],
    model: "_Model | None" = None,
    max_candidates: int = 12,
    analysis_timeout_s: float = _freetoken.ANALYSIS_TIMEOUT_S,
    max_tokens: int = _freetoken.ANALYSIS_MAX_TOKENS,
    request_limit: int = _freetoken.ANALYSIS_REQUEST_LIMIT,
    disable_thinking: bool = False,
    session: "_freetoken.FreeTokenSession | None" = None,
) -> tuple[RadarDraft, str]:
    """Run the radar analysis; return ``(draft, prompt)``.

    Synchronous wrapper around :func:`analyze_candidates_async` (fresh event
    loop per call, never nested). Raises :class:`freetoken.FreeTokenError`
    when the local model endpoint is unreachable/misconfigured or the
    overall deadline is breached.
    """
    return _asyncio.run(
        analyze_candidates_async(
            candidates,
            model=model,
            max_candidates=max_candidates,
            analysis_timeout_s=analysis_timeout_s,
            max_tokens=max_tokens,
            request_limit=request_limit,
            disable_thinking=disable_thinking,
            session=session,
        )
    )
