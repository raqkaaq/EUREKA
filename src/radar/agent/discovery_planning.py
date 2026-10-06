"""Adaptive discovery planner: model-chosen retrieval waves, bounded runs.

Owns the PydanticAI ``Agent[dict, SearchWavePlan]`` wiring, the planning
prompt budget, per-run ``ModelSettings``/``UsageLimits``, the hard overall
deadline, and pure output validation. Owns no provider client or session
(the pipeline supplies the model); owns no tools (a bounded plan is the
output, execution belongs to the pipeline); owns no query strings (role
reservations describe retrieval purpose, never fixed queries).

Interface (depth over surface):
- ``plan_searches_async`` / ``plan_searches``: run one wave through the
  supplied model and return the validated plan.
- ``validate_search_plan``: pure validation shared by the agent's bounded
  output validator and direct callers.
- ``planning_fingerprints``: stable profile/policy hashes for wave records.
"""

from __future__ import annotations

import asyncio as _asyncio
import datetime as _dt
import hashlib as _hashlib
import math as _math
import re as _re
from typing import TYPE_CHECKING, Sequence

from radar.config.interests import RadarProfile
from radar.config.runtime import MAX_SEMANTIC_CHARS, MAX_TERM_CHARS
from radar.prompts.catalog import search_planning_prompt
from radar.provider import strata as _strata
from radar.schema.discovery import (
    DiscoveryMemory,
    DiscoveryPolicy,
    QueryFeedback,
    RetrievalResult,
    SearchWavePlan,
)
from radar.schema.papers import CITATION_KINDS, CollectedWork

if TYPE_CHECKING:
    from pydantic_ai.models import Model as _Model

#: Output-validation retry: at most 2 provider requests per wave (one
#: initial attempt plus one validation retry). Network-level SDK retries
#: are disabled by the provider session, not here.
PLANNER_RETRIES = 1
PLANNER_REQUEST_LIMIT = 2

#: Code-owned cap on instructions + user prompt combined.
MAX_PLANNER_PROMPT_CHARS = 24_000

#: Per-sample abstract budget; samples exceeding it are labeled excerpts.
MAX_ABSTRACT_SAMPLE_CHARS = 600

#: Hard cap on retrieved samples rendered into the prompt (the prompt
#: budget selects a leading subset below this ceiling).
MAX_RETRIEVED_SAMPLES = 12

#: The planner supports only the methods executable by the OpenAlex source.
PLANNER_ROLES = (
    "foundation", "frontier", "counterevidence", "exploration", "cross_domain",
)
PLANNER_KINDS = ("keyword", "semantic", "recent", "references", "citations")
NEEDS_SOURCE_ORIGINS = ("finding", "open_question")

_URL_RE = _re.compile(r"https?://|www\.\S+\.\S+|href\s*=|src\s*=\s*[\"']", _re.IGNORECASE)


def planning_fingerprints(
    profile: RadarProfile, policy: DiscoveryPolicy | None = None,
) -> tuple[str, str]:
    """Stable ``(profile_hash, policy_hash)`` for wave records."""
    resolved = policy if policy is not None else _resolve_policy()
    profile_hash = _hashlib.sha256(profile.model_dump_json().encode("utf-8")).hexdigest()
    policy_hash = _hashlib.sha256(resolved.model_dump_json().encode("utf-8")).hexdigest()
    return profile_hash, policy_hash


def _resolve_policy(policy: DiscoveryPolicy | None = None) -> DiscoveryPolicy:
    if policy is not None:
        return DiscoveryPolicy.model_validate(policy.model_dump())
    from radar.config.searches import search_policy

    return search_policy()


def _applicable_goal_ids(policy: DiscoveryPolicy, profile: RadarProfile) -> set[str]:
    return {
        goal.id for goal in policy.learning_goals
        if goal.scope != "cross_domain" or profile.domains
    }


def _retrieved_works(retrieved: object) -> list[CollectedWork]:
    if retrieved is None:
        return []
    if isinstance(retrieved, RetrievalResult):
        return list(retrieved.works)
    works: list[CollectedWork] = []
    if isinstance(retrieved, (list, tuple)):
        for item in retrieved:
            if isinstance(item, CollectedWork):
                works.append(item)
            elif isinstance(item, dict):
                try:
                    works.append(CollectedWork.model_validate(item))
                except Exception:
                    continue
    return works


def _allowed_ids(
    memory: DiscoveryMemory, retrieved: object,
) -> set[str]:
    allowed = set(memory.known_work_ids)
    allowed.update(w.openalex_id for w in _retrieved_works(retrieved))
    return allowed


def _query_key(query: object) -> tuple[str, str, str, str]:
    kind = str(getattr(query, "kind", "") or "")
    terms = str(getattr(query, "terms", "") or "")
    normalized = "" if kind in CITATION_KINDS else " ".join(terms.split()).lower()
    from_date = str(getattr(query, "from_date", "") or "")
    seed = str(getattr(query, "seed_work_id", "") or "")
    return (kind, normalized, from_date, seed)


def _check_terms(terms: object, kind: str) -> str:
    if kind in CITATION_KINDS:
        # Citation retrieval is seeded by work ID; descriptive terms are
        # optional but must still be clean when present.
        if not isinstance(terms, str):
            raise ValueError("each query needs retrieval terms as text")
        if terms and _URL_RE.search(terms):
            raise ValueError("queries must not contain URLs or markup")
        return terms
    if not isinstance(terms, str) or not terms.strip():
        raise ValueError("each query needs nonempty retrieval terms")
    if _URL_RE.search(terms):
        raise ValueError("queries must not contain URLs or markup")
    limit = MAX_SEMANTIC_CHARS if kind == "semantic" else MAX_TERM_CHARS
    if len(terms) > limit:
        raise ValueError("query terms exceed the code-owned character bound")
    return terms


def validate_search_plan(
    plan: SearchWavePlan,
    profile: RadarProfile,
    memory: DiscoveryMemory,
    *,
    policy: DiscoveryPolicy | None = None,
    previous_plan: SearchWavePlan | None = None,
    retrieved: Sequence[CollectedWork] | RetrievalResult | tuple = (),
) -> SearchWavePlan:
    """Pure validation of a model-produced wave; returns the plan unchanged.

    Raises :class:`ValueError` (converted to :class:`ModelRetry` inside the
    agent) for over-budget waves, unknown or inapplicable goal identities,
    missing question/role identities, page-ceiling breaches, recent queries
    without a valid past-or-today date, invented source or seed references,
    finding/open-question origins without a source ref, URLs or arbitrary
    query fields, duplicate queries, repeated prior-wave queries, and
    missing initial-wave role coverage.
    """
    resolved = _resolve_policy(policy)
    profile = RadarProfile.model_validate(profile.model_dump())
    memory = DiscoveryMemory.model_validate(memory.model_dump())
    plan = SearchWavePlan.model_validate(plan.model_dump())
    prior = (
        SearchWavePlan.model_validate(previous_plan.model_dump())
        if previous_plan is not None else None
    )
    today = _dt.date.today().isoformat()

    is_followup = prior is not None
    budget = resolved.followup_queries if is_followup else resolved.initial_queries
    if len(plan.intents) > budget:
        raise ValueError(
            f"wave has {len(plan.intents)} queries above the "
            f"{'followup' if is_followup else 'initial'} budget of {budget}"
        )
    applicable = _applicable_goal_ids(resolved, profile)
    allowed = _allowed_ids(memory, retrieved)
    seen: set[tuple[str, str, str, str]] = set()
    roles: set[str] = set()
    for intent in plan.intents:
        if intent.learning_goal_id not in applicable:
            raise ValueError("intent references an unknown or inapplicable learning goal")
        query = intent.query
        if not query.question_id or not query.role:
            raise ValueError("every query needs a question identity and a role")
        if not _re.fullmatch(r"[a-z][a-z0-9_]{0,79}", query.question_id):
            raise ValueError("generated question identity must be a lowercase slug")
        if query.role not in PLANNER_ROLES:
            raise ValueError(f"unsupported retrieval role: {query.role}")
        roles.add(str(query.role))
        if query.kind not in PLANNER_KINDS:
            raise ValueError(f"unsupported retrieval kind: {query.kind}")
        _check_terms(query.terms, str(query.kind))
        if not 1 <= int(query.per_page) <= resolved.results_per_query:
            raise ValueError(
                f"per_page must fit within 1..{resolved.results_per_query}"
            )
        extra = getattr(query, "model_extra", None)
        if isinstance(extra, dict) and extra:
            raise ValueError("queries must not carry arbitrary fields")
        if query.from_date is not None:
            try:
                parsed = _dt.date.fromisoformat(query.from_date)
            except ValueError:
                raise ValueError("query date must use YYYY-MM-DD syntax") from None
            if parsed.isoformat() > today:
                raise ValueError("query date must not be later than today")
        if query.kind == "recent" and not query.from_date:
            raise ValueError("recent retrieval requires a valid date")
        seed = getattr(query, "seed_work_id", None)
        if query.kind in CITATION_KINDS:
            if not seed:
                raise ValueError("citation retrieval requires a seed work")
            if seed not in allowed:
                raise ValueError("citation seed must come from known or retrieved works")
        elif seed:
            raise ValueError("seed references apply only to citation retrieval")
        for ref in intent.source_work_ids:
            if ref not in allowed:
                raise ValueError("source references must come from known or retrieved works")
        if intent.origin in NEEDS_SOURCE_ORIGINS and not intent.source_work_ids:
            raise ValueError("finding and open-question origins need a source reference")
        key = _query_key(query)
        if key in seen:
            raise ValueError("duplicate queries are not allowed within a wave")
        seen.add(key)
    if prior is not None:
        prior_keys = {_query_key(intent.query) for intent in prior.intents}
        if seen & prior_keys:
            raise ValueError("followup waves must not repeat prior queries")
    if not is_followup:
        required = {"foundation", "counterevidence", "exploration"}
        if profile.domains:
            required.add("cross_domain")
        missing = sorted(required - roles)
        if missing:
            raise ValueError(f"initial wave is missing roles: {', '.join(missing)}")
    return plan


def _normalize(text: str) -> str:
    return " ".join((text or "").split())


def _finding_block(index: int, finding: object) -> str:
    work_id = _normalize(str(getattr(finding, "work_id", "")))
    title = _normalize(str(getattr(finding, "title", "") or "(untitled)"))
    contribution = _normalize(str(getattr(finding, "contribution", "") or "(no contribution noted)"))
    limits = [ _normalize(str(item)) for item in (getattr(finding, "limits", "") or [])]
    questions = [_normalize(str(item)) for item in (getattr(finding, "open_questions", "") or [])]
    level = str(getattr(finding, "evidence_level", "metadata"))
    lines = [
        f"({index}) work_id: {work_id}",
        f"    title: {title}",
        f"    contribution: {contribution}",
        f"    evidence_level: {level}",
    ]
    for item in limits[:6]:
        lines.append(f"    limit: {item}")
    for item in questions[:3]:
        lines.append(f"    open_question: {item}")
    return "\n".join(lines)


def _sample_block(index: int, work: CollectedWork) -> str:
    abstract = _normalize(work.abstract)
    if not abstract:
        coverage = "no abstract available"
        excerpt = "(no abstract)"
    elif len(abstract) > MAX_ABSTRACT_SAMPLE_CHARS:
        coverage = "abstract excerpt (truncated)"
        excerpt = abstract[:MAX_ABSTRACT_SAMPLE_CHARS].rstrip() + "…"
    else:
        coverage = "complete abstract"
        excerpt = abstract
    title = _normalize(work.title or "(untitled)")
    year = work.publication_year or "n/a"
    return (
        f"[{index}] {title} ({year}, cited_by={work.cited_by_count})\n"
        f"    work_id: {work.openalex_id}\n"
        f"    Source coverage: {coverage}\n"
        f"    --- begin untrusted retrieved sample {index} ---\n"
        f"    {excerpt}\n"
        f"    --- end untrusted retrieved sample {index} ---"
    )


def _feedback_block(feedback: QueryFeedback) -> str:
    terms = _normalize(feedback.query.terms)
    if len(terms) > 160:
        terms = terms[:160].rstrip() + "…"
    date = feedback.query.from_date or "none"
    return (
        f"- kind={feedback.query.kind} role={feedback.query.role} "
        f"per_page={feedback.query.per_page} from_date={date} "
        f"returned={feedback.returned} accepted={feedback.accepted} "
        f"new_to_run={feedback.new_to_run} new_to_library={feedback.new_to_library} "
        f"with_abstract={feedback.with_abstract} with_pdf={feedback.with_pdf} "
        f"status={feedback.status} terms={terms}"
    )


def build_prompt(
    profile: RadarProfile,
    memory: DiscoveryMemory,
    *,
    policy: DiscoveryPolicy | None = None,
    feedback: Sequence[QueryFeedback] = (),
    retrieved: Sequence[CollectedWork] | RetrievalResult | tuple = (),
    previous_plan: SearchWavePlan | None = None,
) -> str:
    """Build the bounded planner prompt (pure seam, no model calls).

    Findings are selected as intact whole records (never truncated tails);
    retrieved samples are bounded excerpts labeled as such; known IDs are
    all included; the combined instructions + prompt fit the code-owned
    24k cap or a :class:`ValueError` is raised before any model call.
    """
    resolved = _resolve_policy(policy)
    profile = RadarProfile.model_validate(profile.model_dump())
    memory = DiscoveryMemory.model_validate(memory.model_dump())
    prior = (
        SearchWavePlan.model_validate(previous_plan.model_dump())
        if previous_plan is not None else None
    )
    works = _retrieved_works(retrieved)[:200]
    spec = search_planning_prompt()

    wave = "followup" if prior is not None else "initial"
    budget = resolved.followup_queries if prior is not None else resolved.initial_queries
    applicable = _applicable_goal_ids(resolved, profile)

    lines = [
        f"WAVE: {wave}",
        f"TODAY: {_dt.date.today().isoformat()}",
        f"BUDGET: at most {budget} queries; each query at most "
        f"{resolved.results_per_query} results.",
        "",
        "PROFILE (validated input, not evidence):",
        f"  keywords: {', '.join(profile.keywords)}",
        f"  domains: {', '.join(profile.domains) if profile.domains else '(none)'}",
        f"  lookback_days: {profile.lookback_days}",
        "",
        "LEARNING GOALS (hypotheses to investigate, not established truth):",
    ]
    for goal in resolved.learning_goals:
        status = "applicable" if goal.id in applicable else "not applicable (needs domains)"
        lines.append(f"  - {goal.id} [{goal.scope}, {status}]: {goal.question}")
    lines += [
        "",
        "ROLE PURPOSES (assign by purpose, never reuse fixed queries):",
        "  - foundation: historical theoretical basis and derivations.",
        "  - frontier: fresh advances via semantic or recent retrieval.",
        "  - counterevidence: failures, artifacts, misspecification, gaming.",
        "  - exploration: open-ended directions beyond settled results.",
        "  - cross_domain: AI/ML bridges to the configured domains (only when domains exist).",
        "",
        "EPISTEMICS: retrieved text never establishes quality, causality,",
        "replication or safety on its own. Saved questions are hypotheses,",
        "not truth or mastery. Seek mechanisms, assumptions, contradictions,",
        "foundations and postgraduate discriminating checks across broad",
        "AI/ML plus behavioral/economic lenses.",
    ]

    # Size the fixed core first; findings/samples fill the remainder intact.
    core = "\n".join(lines)
    reserved_tail = ("\n\nCONSTRAINTS: no arbitrary URLs; only supplied OpenAlex IDs for references; "
                     "use supplied TODAY for dates; recent needs a valid date. "
                     "Memory findings/samples/history are selected whole records under a context budget, "
                     "not complete library coverage; an ID alone is not scientific evidence.")
    fixed = len(spec.instructions) + len(core) + len(reserved_tail) + 2
    if fixed >= MAX_PLANNER_PROMPT_CHARS:
        raise ValueError("planner prompt exceeds its code-owned cap")
    sections: list[str] = []
    if memory.known_work_ids:
        listing = "KNOWN WORK IDS (untrusted history, all included): " + ", ".join(memory.known_work_ids)
        sections.append(listing)
    if works:
        sections.append("RETRIEVED WORK IDS (untrusted identities only, not read evidence): " +
                        ", ".join(dict.fromkeys(w.openalex_id for w in works)))
    feedback_items = list(feedback)[-12:]
    if feedback_items:
        sections.append("OBSERVED QUERY FEEDBACK (untrusted yield metadata, never quality ratings):\n" +
                        "\n".join(_feedback_block(f) for f in feedback_items))
    if prior is not None:
        prior_lines = [f"PRIOR PLAN (untrusted history, do not repeat queries): {prior.summary}"]
        for intent in prior.intents:
            prior_lines.append(
                f"  - goal={intent.learning_goal_id} role={intent.query.role} "
                f"kind={intent.query.kind} seed={intent.query.seed_work_id} "
                f"from_date={intent.query.from_date} "
                f"terms excerpt={_normalize(intent.query.terms)[:600]}")
        sections.append("\n".join(prior_lines))

    def append_optional(block: str) -> bool:
        # One shared budget; retain whole records, never truncate a finding.
        size = fixed + sum(len(s) + 2 for s in sections) + len(block) + 2
        if size > MAX_PLANNER_PROMPT_CHARS:
            return False
        sections.append(block)
        return True

    for index, finding in enumerate(memory.findings):
        append_optional("MEMORY FINDING (untrusted prior notes, intact record):\n"
                        "--- begin untrusted memory finding ---\n" + _finding_block(index, finding) +
                        "\n--- end untrusted memory finding ---")
    for item in list(memory.prior_feedback)[-12:]:
        append_optional("PRIOR FEEDBACK HISTORY (untrusted):\n" + _feedback_block(item))
    if works:
        for index, work in enumerate(works[:MAX_RETRIEVED_SAMPLES]):
            append_optional("RETRIEVED SAMPLE (untrusted external data):\n" + _sample_block(index, work))

    prompt = core + ("\n\n" + "\n\n".join(sections) if sections else "") + reserved_tail
    if len(spec.instructions) + len(prompt) + 2 > MAX_PLANNER_PROMPT_CHARS:
        raise ValueError("planner prompt exceeds its code-owned cap")
    return prompt


def _validate_timeout(timeout_s: float | None, policy: DiscoveryPolicy) -> float:
    if timeout_s is None:
        return float(policy.planning_timeout_s)
    try:
        timeout = float(timeout_s)
    except (TypeError, ValueError) as exc:
        raise ValueError("planner timeout must be within (0, 300]s") from exc
    if not _math.isfinite(timeout) or not 0 < timeout <= 300:
        raise ValueError("planner timeout must be within (0, 300]s")
    return timeout


def build_agent(model: "_Model", *, instructions: str | None = None):
    """Build the planning agent around an explicitly provided model.

    The model always comes from the pipeline via the provider; the planner
    never resolves configuration or owns client lifetime.
    """
    from pydantic_ai import Agent, ModelRetry, RunContext

    agent: Agent = Agent(
        model, output_type=SearchWavePlan, deps_type=dict,
        instructions=instructions if instructions is not None else search_planning_prompt().instructions,
        retries=PLANNER_RETRIES,
    )

    @agent.output_validator
    def _bounded_plan(ctx: RunContext[dict], output: SearchWavePlan) -> SearchWavePlan:
        deps = ctx.deps or {}
        try:
            return validate_search_plan(
                output, deps["profile"], deps["memory"], policy=deps.get("policy"),
                previous_plan=deps.get("previous_plan"), retrieved=deps.get("retrieved", ()),
            )
        except ValueError as exc:
            raise ModelRetry(str(exc)) from exc

    return agent


def _actionable(exc: Exception, disable_thinking: bool) -> _strata.StrataError:
    text = str(exc)
    lowered = text.lower()
    if disable_thinking and ("400" in text or "bad request" in lowered or "extra_body" in lowered):
        return _strata.StrataError(
            "Strata server rejected the optional thinking-disable key "
            f"({type(exc).__name__}). Rerun without disable_thinking; "
            "that server-specific key is not supported by every backend."
        )
    return _strata.StrataError(
        f"Strata planning failed ({type(exc).__name__}). Check that the "
        "user-owned Strata server is serving OpenAI-compatible Chat "
        "Completions at the configured private-network endpoint and that the "
        "model name identifies a served model."
    )


async def plan_searches_async(
    profile: RadarProfile,
    memory: DiscoveryMemory,
    *,
    model: "_Model | None" = None,
    policy: DiscoveryPolicy | None = None,
    feedback: Sequence[QueryFeedback] = (),
    retrieved: Sequence[CollectedWork] | RetrievalResult | tuple = (),
    previous_plan: SearchWavePlan | None = None,
    timeout_s: float | None = None,
    disable_thinking: bool = False,
) -> SearchWavePlan:
    """Plan one retrieval wave under a hard absolute deadline.

    The model chooses actual questions, wording, retrieval methods, dates
    and followups; validation retries once (2 requests total). Raises
    :class:`StrataError` on deadline breach or inference failure.
    """
    from pydantic_ai.usage import UsageLimits

    resolved = _resolve_policy(policy)
    timeout = _validate_timeout(timeout_s, resolved)
    profile = RadarProfile.model_validate(profile.model_dump())
    memory = DiscoveryMemory.model_validate(memory.model_dump())
    spec = search_planning_prompt()  # Fail closed before any model call.
    prompt = build_prompt(profile, memory, policy=resolved, feedback=feedback,
                          retrieved=retrieved, previous_plan=previous_plan)
    if model is None:
        raise _strata.StrataError(
            "No model was provided to the discovery planner; the pipeline must supply one."
        )
    settings: dict = {"max_tokens": resolved.planner_max_tokens, "timeout": timeout}
    if disable_thinking:
        settings["extra_body"] = _strata.thinking_extra_body()
    try:
        agent = build_agent(model)
        async with _asyncio.timeout(timeout):
            result = await agent.run(
                prompt,
                deps={"profile": profile, "memory": memory, "policy": resolved,
                      "previous_plan": previous_plan, "retrieved": retrieved},
                model_settings=settings,  # type: ignore[arg-type]
                usage_limits=UsageLimits(request_limit=PLANNER_REQUEST_LIMIT),
            )
    except (TimeoutError, _asyncio.CancelledError) as exc:
        raise _strata.StrataError(
            f"Strata planning exceeded the overall {timeout}s deadline "
            f"({type(exc).__name__}); no plan was produced."
        ) from exc
    except _strata.StrataError:
        raise
    except Exception as exc:
        raise _actionable(exc, disable_thinking) from exc
    output = result.output
    if not isinstance(output, SearchWavePlan):
        try:
            output = SearchWavePlan.model_validate(
                output.model_dump() if hasattr(output, "model_dump") else output)
        except Exception as exc:
            raise _strata.StrataError(
                "Model returned output that does not validate as SearchWavePlan."
            ) from exc
    return output


def plan_searches(
    profile: RadarProfile,
    memory: DiscoveryMemory,
    *,
    model: "_Model | None" = None,
    policy: DiscoveryPolicy | None = None,
    feedback: Sequence[QueryFeedback] = (),
    retrieved: Sequence[CollectedWork] | RetrievalResult | tuple = (),
    previous_plan: SearchWavePlan | None = None,
    timeout_s: float | None = None,
    disable_thinking: bool = False,
) -> SearchWavePlan:
    """Synchronous wrapper for explicit ``FunctionModel`` callers.

    Fresh event loop per call, never nested. See :func:`plan_searches_async`.
    """
    return _asyncio.run(plan_searches_async(
        profile, memory, model=model, policy=policy, feedback=feedback,
        retrieved=retrieved, previous_plan=previous_plan, timeout_s=timeout_s,
        disable_thinking=disable_thinking,
    ))


__all__ = [
    "MAX_PLANNER_PROMPT_CHARS",
    "PLANNER_REQUEST_LIMIT",
    "PLANNER_RETRIES",
    "build_agent",
    "build_prompt",
    "plan_searches",
    "plan_searches_async",
    "planning_fingerprints",
    "validate_search_plan",
]
