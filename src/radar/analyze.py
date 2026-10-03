"""PydanticAI analysis wiring: candidates -> structured RadarDraft.

All LLM inference goes through PydanticAI ``Agent`` with a Pydantic
``output_type``. Tests inject a ``TestModel``/``FunctionModel`` via the
``model`` parameter; production builds the private-network FreeToken model via
:mod:`radar.freetoken`. No direct OpenAI SDK calls anywhere on this path.
"""

from __future__ import annotations

from typing import Any

from radar import freetoken as _freetoken
from radar.models import CollectedWork, RadarDraft
from radar.prompt import build_prompt


def build_agent(model: Any | None = None, output_type: type[RadarDraft] = RadarDraft):
    """Build the analysis agent.

    ``model`` is a PydanticAI model instance (injectable for tests). When
    ``None``, the private-network FreeToken model is resolved and built (may raise
    :class:`freetoken.FreeTokenError` with an actionable message).
    """
    from pydantic_ai import Agent

    if model is None:
        config = _freetoken.FreeTokenConfig.resolve()
        model = _freetoken.build_model(config)
    return Agent(model, output_type=output_type, retries=_freetoken.MODEL_MAX_RETRIES)


def analyze_candidates(
    candidates: list[CollectedWork],
    model: Any | None = None,
    max_candidates: int = 12,
) -> tuple[RadarDraft, str]:
    """Run the radar analysis; return ``(draft, prompt)``.

    Raises :class:`freetoken.FreeTokenError` when the local model endpoint
    is unreachable/misconfigured, wrapping lower-level failures with an
    actionable message naming the endpoint.
    """
    prompt = build_prompt(candidates, max_candidates=max_candidates)
    agent = build_agent(model)
    try:
        result = agent.run_sync(prompt)
    except _freetoken.FreeTokenError:
        raise
    except Exception as exc:
        raise _freetoken.FreeTokenError(
            f"FreeToken inference failed ({type(exc).__name__}: {exc}). "
            "Check that your user-owned FreeToken server is serving "
            "OpenAI-compatible Chat Completions at the configured private-network "
            "endpoint from FREETOKEN_BASE_URL/--base-url and that "
            "FREETOKEN_MODEL names a served model."
        ) from exc
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
