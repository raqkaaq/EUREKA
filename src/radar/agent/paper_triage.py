"""Same screening input/answer contracts over Strata PydanticAI chat.

The only adaptation is transport framing: complete native screening inputs
are grouped with work IDs in the chat user message; output is validated
against the same SystemOneResponse model used by CLEF. No rewritten rubric.
"""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING

from radar.config.interests import RadarProfile
from radar.config.runtime import MAX_TOTAL_WORKS, validate_qwen_overall_timeout
from radar.processing.triage_input import build_input
from radar.prompts.catalog import paper_triage_prompt, screening_questions
from radar.provider import strata
from radar.schema.papers import CollectedWork
from radar.schema.triage import (
    QwenResponses,
    ScreeningFailureKind,
    TriageBatch,
    TriageResult,
)

if TYPE_CHECKING:
    from pydantic_ai.models import Model

MAX_PROMPT_BYTES = 64 * 1024
MAX_REQUEST_TOKENS = 4096


def _prompt(chunk: list[CollectedWork], profile: RadarProfile, model_id: str) -> str:
    return json.dumps({"papers": [
        {"work_id": w.openalex_id, "input": build_input(w, profile, model_id)}
        for w in chunk]}, ensure_ascii=False)


def _plan(
    works: list[CollectedWork],
    profile: RadarProfile,
    model_id: str,
    *,
    batch_size: int | None = None,
):
    from radar.config.triage import qwen_screening_policy

    if batch_size is None:
        batch_size = int(qwen_screening_policy().batch_size)
    results: dict[str, TriageResult] = {}
    batches: list[list[CollectedWork]] = []
    pending: list[CollectedWork] = []
    for work in works:
        if not work.abstract.strip():
            results[work.openalex_id] = TriageResult(work_id=work.openalex_id, status="missing_abstract")
        elif len(_prompt([work], profile, model_id).encode("utf-8")) > MAX_PROMPT_BYTES:
            results[work.openalex_id] = TriageResult(work_id=work.openalex_id, status="oversized")
        else:
            candidate = pending + [work]
            if pending and (len(candidate) > batch_size or
                            len(_prompt(candidate, profile, model_id).encode("utf-8")) > MAX_PROMPT_BYTES):
                batches.append(pending)
                pending = []
            pending.append(work)
    if pending:
        batches.append(pending)
    return batches, results


def build_agent(model: "Model"):
    from pydantic_ai import Agent, ModelRetry, RunContext

    agent = Agent(model, output_type=QwenResponses, deps_type=tuple,
                  instructions=paper_triage_prompt().instructions, retries=1)

    @agent.output_validator
    def exact_contract(ctx: RunContext[tuple[str, tuple[str, ...]]], output: QwenResponses):
        name, expected = ctx.deps
        ids = [r.work_id for r in output.responses]
        if len(ids) != len(expected) or set(ids) != set(expected):
            raise ModelRetry("Return exactly one response for each supplied work_id; no other IDs.")
        if any(r.model != name for r in output.responses):
            raise ModelRetry("Copy the exact model name from each supplied screening input.")
        return output

    return agent


def _walk_chain(exc: BaseException):
    """Yield an exception and its causes/contexts without touching messages."""
    seen: set[int] = set()
    stack: list[BaseException] = [exc]
    while stack:
        current = stack.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        yield current
        if isinstance(current, BaseExceptionGroup):
            stack.extend(current.exceptions)
        for attr in ("__cause__", "__context__"):
            nxt = getattr(current, attr, None)
            if isinstance(nxt, BaseExceptionGroup):
                stack.extend(nxt.exceptions)
            elif isinstance(nxt, BaseException):
                stack.append(nxt)


def _classify_failure(exc: BaseException) -> ScreeningFailureKind:
    """Classify typed causes without reading messages or upstream bodies."""
    import httpx
    import httpx2
    from pydantic import ValidationError
    from pydantic_ai.exceptions import (
        ModelAPIError, ModelHTTPError, ModelRetry,
        UnexpectedModelBehavior, UsageLimitExceeded,
    )

    nodes = tuple(_walk_chain(exc))
    categories = (
        ((httpx.ConnectTimeout, httpx2.ConnectTimeout), ScreeningFailureKind.CONNECT_TIMEOUT),
        ((httpx.ReadTimeout, httpx2.ReadTimeout), ScreeningFailureKind.READ_TIMEOUT),
        ((TimeoutError, httpx.TimeoutException, httpx2.TimeoutException),
         ScreeningFailureKind.REQUEST_TIMEOUT),
        ((UnexpectedModelBehavior, ValidationError, ModelRetry, UsageLimitExceeded),
         ScreeningFailureKind.INVALID_RESPONSE),
        ((ModelHTTPError, ModelAPIError, strata.StrataError,
          httpx.TransportError, httpx2.TransportError,
          httpx.HTTPStatusError, httpx2.HTTPStatusError), ScreeningFailureKind.SERVICE_ERROR),
    )
    for types, kind in categories:
        if any(isinstance(node, types) for node in nodes):
            return kind
    return ScreeningFailureKind.UNEXPECTED_ERROR


async def screen_works_async(
    works: list[CollectedWork], profile: RadarProfile, *,
    model: "Model | None" = None, model_id: str | None = None,
    base_url: str | None = None, configured_model: str | None = None,
    overall_timeout_s: float | None = None, disable_thinking: bool = False,
    fallback_reason: str | None = None,
) -> TriageBatch:
    from pydantic_ai.usage import UsageLimits

    from radar.config.triage import qwen_screening_policy

    policy = qwen_screening_policy()
    if overall_timeout_s is None:
        budget = float(policy.overall_timeout_s)
    else:
        budget = validate_qwen_overall_timeout(overall_timeout_s)
    request_timeout = float(policy.request_timeout_s)
    batch_size = int(policy.batch_size)
    concurrency = int(policy.concurrency)
    per_request = min(budget, request_timeout)
    ids = [w.openalex_id for w in works]
    if len(ids) > MAX_TOTAL_WORKS or len(set(ids)) != len(ids):
        raise ValueError("routing pool exceeds its bound or contains duplicate IDs")
    paper_triage_prompt()  # Validate before provider resolution or network I/O.
    name = model_id or configured_model or strata.model_from_env() or "qwen-not-called"
    batches, results = _plan(works, profile, name, batch_size=batch_size)
    session = None
    try:
        if batches:
            try:
                async with asyncio.timeout(budget):
                    if model is None:
                        try:
                            config = await strata.StrataConfig.resolve_async(
                                base_url=base_url, model=configured_model,
                                timeout_s=min(budget, request_timeout))
                            name = config.model
                            session = strata.build_session(config)
                            model = session.model
                            batches, results = _plan(works, profile, name, batch_size=batch_size)
                        except asyncio.CancelledError:
                            raise
                        except Exception as exc:
                            kind = _classify_failure(exc)
                            for wid in ids:
                                results.setdefault(
                                    wid,
                                    TriageResult(work_id=wid, status="failed", failure_kind=kind),
                                )
                            batches = []
                    if batches and model is not None:
                        agent = build_agent(model)
                        semaphore = asyncio.Semaphore(concurrency)
                        settings = {"max_tokens": MAX_REQUEST_TOKENS,
                                    "timeout": per_request}
                        if disable_thinking:
                            settings["extra_body"] = strata.thinking_extra_body()

                        async def score(chunk: list[CollectedWork]):
                            async with semaphore:
                                try:
                                    async with asyncio.timeout(per_request):
                                        output = (await agent.run(
                                            _prompt(chunk, profile, name),
                                            deps=(name, tuple(w.openalex_id for w in chunk)),
                                            model_settings=settings,
                                            usage_limits=UsageLimits(request_limit=2),
                                        )).output
                                    for response in output.responses:
                                        results[response.work_id] = response.to_result(response.work_id)
                                except asyncio.CancelledError:
                                    raise
                                except TimeoutError:
                                    for work in chunk:
                                        results[work.openalex_id] = TriageResult(
                                            work_id=work.openalex_id, status="failed",
                                            failure_kind=ScreeningFailureKind.REQUEST_TIMEOUT)
                                except Exception as exc:
                                    kind = _classify_failure(exc)
                                    for work in chunk:
                                        results[work.openalex_id] = TriageResult(
                                            work_id=work.openalex_id, status="failed",
                                            failure_kind=kind)

                        async with asyncio.TaskGroup() as group:
                            for chunk in batches:
                                group.create_task(score(chunk))
            except TimeoutError:
                for wid in ids:
                    if wid not in results:
                        results[wid] = TriageResult(
                            work_id=wid, status="deadline",
                            failure_kind=ScreeningFailureKind.BUDGET_EXHAUSTED)
            except asyncio.CancelledError:
                for wid in ids:
                    if wid not in results:
                        results[wid] = TriageResult(
                            work_id=wid, status="deadline",
                            failure_kind=ScreeningFailureKind.BUDGET_EXHAUSTED)
                raise
            except Exception as exc:
                # Never echo upstream prompts, bodies, auth material, or URLs.
                kind = _classify_failure(exc)
                for wid in ids:
                    results.setdefault(
                        wid, TriageResult(work_id=wid, status="failed", failure_kind=kind))
    finally:
        if session is not None:
            await session.http_client.aclose()
    return TriageBatch(
        model_id=name, rubric_version=screening_questions().rubric_version,
        rubric_hash=screening_questions().fingerprint, backend="qwen",
        probability_kind="prompted_estimate", fallback_reason=fallback_reason,
        results=[results[wid] for wid in ids])


def screen_works(works: list[CollectedWork], profile: RadarProfile, **kwargs) -> TriageBatch:
    """One loop owns routing tasks and deterministic provider cleanup."""
    return asyncio.run(screen_works_async(works, profile, **kwargs))
