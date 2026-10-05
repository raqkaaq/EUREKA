"""Offline discovery-planner doubles at the model/prompt boundary (tests only).

Provides a deterministic PydanticAI ``FunctionModel`` that returns fixed
initial/followup :class:`SearchWavePlan` outputs, plus a matching legacy
:class:`QueryPlan` fixture and canned OpenAlex pages keyed by the same
synthetic terms. No network, no filesystem writes, no production imports
beyond schema types.

Synthetic terms are simple, meaningful, and reserved for the fake API only;
the production planner never imports this module.
"""

from __future__ import annotations

import datetime as _dt

from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models.function import FunctionModel

# Goal identities mirror the packaged v3 policy goals (plain string
# constants; this fixture never calls the production planner).
_GOAL_GENERALIZATION = "learning_and_generalization"
_GOAL_UNCERTAINTY = "uncertainty_and_identification"
_GOAL_INCENTIVES = "objectives_and_incentives"
_GOAL_MEASUREMENT = "measurement_and_human_ai"

# Six fixed initial queries: all five planner roles plus one extra frontier.
# Each entry: (goal_id, question, rationale, value, origin, kind, terms,
#              role, per_page, from_date_mode).
# ``from_date_mode`` is "recent" (90-day window from today), None, or an
# explicit offset in days (frontier freshness uses a real calendar date).
_INITIAL_SPECS: tuple[tuple[str, ...], ...] = (
    (
        _GOAL_GENERALIZATION,
        "Which mechanisms explain generalization and transfer?",
        "Separate structural mechanisms from scale and data artifacts.",
        "Reconstruct the assumption-dependent argument.",
        "agenda", "keyword",
        "mechanism generalization transfer inductive bias",
        "foundation", "10", "",
    ),
    (
        _GOAL_GENERALIZATION,
        "When does scaling produce new capabilities versus better fitting?",
        "Discriminate emergence claims from evaluation artifacts.",
        "Identify a falsifiable scaling test.",
        "agenda", "semantic",
        "how scaling changes transfer and emergent capabilities",
        "frontier", "12", "",
    ),
    (
        _GOAL_GENERALIZATION,
        "Which reported advances fail under distribution shift?",
        "Seek boundary conditions and negative evidence.",
        "Name a concrete failure mode to check.",
        "agenda", "keyword",
        "shortcut learning spurious correlations failed replication",
        "counterevidence", "10", "",
    ),
    (
        _GOAL_UNCERTAINTY,
        "What unexplored regularization mechanisms exist beyond scaling?",
        "Probe open directions rather than settled results.",
        "Propose one discriminating check for a new mechanism.",
        "exploration", "semantic",
        "unexplored regularization mechanisms beyond scale",
        "exploration", "10", "",
    ),
    (
        _GOAL_INCENTIVES,
        "How do incentive mechanisms transfer to learning agents?",
        "Connect economic mechanisms to agent behavior.",
        "Sketch an incentive-compatible evaluation idea.",
        "agenda", "keyword",
        "mechanism design incentives reward shaping agents",
        "cross_domain", "10", "",
    ),
    (
        _GOAL_MEASUREMENT,
        "What recent evidence bears on evaluation validity?",
        "Track fresh measurement and human-AI findings.",
        "Summarize one recent validity result.",
        "agenda", "recent",
        "recent advances evaluation validity human AI",
        "frontier", "10", "recent",
    ),
)

# Six fixed followup queries: unique terms, no overlap with the initial set.
# Finding/open-question origins carry source references into the canned
# known IDs so grounded followups validate against a populated memory.
_FOLLOWUP_SPECS: tuple[tuple[str, ...], ...] = (
    (
        _GOAL_GENERALIZATION,
        "Which data ablations isolate the transfer mechanism?",
        "Follow up on thin initial mechanism evidence.",
        "Design a controlled comparison.",
        "finding", "keyword",
        "data ablation transfer mechanism controlled comparison",
        "foundation", "10", "", "0",
    ),
    (
        _GOAL_UNCERTAINTY,
        "Which identifiability conditions fail in practice?",
        "Pursue open questions about confounding.",
        "State a testable identification check.",
        "open_question", "semantic",
        "identifiability conditions confounding practical failure",
        "frontier", "10", "", "1",
    ),
    (
        _GOAL_GENERALIZATION,
        "Which replications contradict the scaling claim?",
        "Chase counterevidence surfaced in feedback.",
        "Cite the contradicting result precisely.",
        "finding", "keyword",
        "replication failure scaling claim benchmark leakage",
        "counterevidence", "10", "", "2",
    ),
    (
        _GOAL_UNCERTAINTY,
        "What calibration methods survive shift?",
        "Explore alternatives raised by retrieved work.",
        "Compare two calibration approaches.",
        "exploration", "semantic",
        "calibration methods distribution shift comparison",
        "exploration", "10", "", "",
    ),
    (
        _GOAL_INCENTIVES,
        "How does strategic behavior distort proxy objectives?",
        "Bridge to incentive literature via retrieved seeds.",
        "Name one gaming mechanism with evidence.",
        "finding", "keyword",
        "strategic behavior proxy objectives Goodhart gaming",
        "cross_domain", "10", "", "3",
    ),
    (
        _GOAL_MEASUREMENT,
        "What new measurement studies appeared this quarter?",
        "Refresh the validity branch with recent work.",
        "Summarize the newest construct validity test.",
        "agenda", "recent",
        "recent construct validity psychometric AI measurement",
        "frontier", "10", "recent", "",
    ),
)

_SUMMARIES = {
    "initial": "Initial wave covering foundation, frontier, counterevidence, exploration and cross-domain roles.",
    "followup": "Followup wave pursuing thin evidence and open questions without repeating prior queries.",
}


def _recent_date() -> str:
    return (_dt.date.today() - _dt.timedelta(days=90)).isoformat()


def _plan_dict(specs: tuple[tuple[str, ...], ...], summary: str) -> dict:
    intents = []
    for spec in specs:
        (goal, question, rationale, value, origin, kind, terms,
         role, per_page, date_mode) = spec[:10]
        source_index = spec[10] if len(spec) > 10 else ""
        from_date = _recent_date() if date_mode else None
        source_ids = (
            [known_work_ids()[int(source_index)]] if source_index != "" else []
        )
        intents.append({
            "learning_goal_id": goal,
            "question": question,
            "rationale": rationale,
            "expected_learning_value": value,
            "origin": origin,
            "source_work_ids": source_ids,
            "query": {
                "kind": kind,
                "terms": terms,
                "per_page": int(per_page),
                **({"from_date": from_date} if from_date else {}),
                "question_id": goal,
                "role": role,
            },
        })
    return {"summary": summary, "intents": intents}


def initial_plan_dict() -> dict:
    """Raw initial-wave plan matching :func:`planner_model` output."""
    return _plan_dict(_INITIAL_SPECS, _SUMMARIES["initial"])


def followup_plan_dict() -> dict:
    """Raw followup-wave plan matching :func:`planner_model` output."""
    return _plan_dict(_FOLLOWUP_SPECS, _SUMMARIES["followup"])


def _is_followup_prompt(messages: object) -> bool:
    try:
        texts = []
        for message in messages or []:
            for part in getattr(message, "parts", []) or []:
                content = getattr(part, "content", "")
                if isinstance(content, str):
                    texts.append(content)
        blob = "\n".join(texts)
    except Exception:
        return False
    return "WAVE: followup" in blob


def planner_model() -> FunctionModel:
    """Offline planner double: fixed initial/followup waves via FunctionModel.

    Selects the wave from the actual submitted prompt (``WAVE: followup``
    marker for followup, otherwise initial), so tests exercise the real
    prompt seam instead of planner internals. Returns six initial queries
    (all five roles plus one extra frontier) or six followup queries with
    unique terms.
    """

    def _impl(messages, info):
        payload = followup_plan_dict() if _is_followup_prompt(messages) else initial_plan_dict()
        import re
        text = "\n".join(p.content for m in messages for p in m.parts
                         if isinstance(getattr(p, "content", None), str))
        ids = list(dict.fromkeys(re.findall(r"https://openalex\.org/W\d+", text)))
        if "domains: (none)" in text:
            payload["intents"] = [i for i in payload["intents"] if i["query"]["role"] != "cross_domain"]
            for intent in payload["intents"]:
                if intent["learning_goal_id"] in ("objectives_and_incentives", "measurement_and_human_ai"):
                    intent["learning_goal_id"] = "learning_and_generalization"
        for intent in payload["intents"]:
            if intent["source_work_ids"]:
                if ids:
                    intent["source_work_ids"] = [ids[0]]
                else:
                    intent.update(origin="agenda", source_work_ids=[])
        return ModelResponse(parts=[ToolCallPart(
            info.output_tools[0].name, payload)])

    return FunctionModel(_impl)


def _spec_queries(specs: tuple[tuple[str, ...], ...], *, omit_cross_domain: bool = False) -> list[dict]:
    from radar.schema.papers import PlannedQuery

    queries = []
    for spec in specs:
        (goal, _q, _r, _v, _o, kind, terms, role, per_page, date_mode) = spec
        if omit_cross_domain and role == "cross_domain":
            continue
        from_date = _recent_date() if date_mode else None
        query = PlannedQuery(
            kind=kind, terms=terms, per_page=int(per_page),
            **({"from_date": from_date} if from_date else {}),
            question_id=goal, role=role,
        )
        queries.append(query)
    return queries


def test_query_plan(profile=None, *, max_queries=6):
    """Legacy first-wave :class:`QueryPlan` matching the fixture model.

    Hardcoded from the same synthetic terms (never calls the production
    planner). When ``profile`` carries no domains, the cross-domain query
    is omitted so the same plan stays applicable with domains absent.
    """
    from radar.schema.papers import QueryPlan

    omit = bool(profile is not None and not getattr(profile, "domains", []))
    queries = _spec_queries(_INITIAL_SPECS, omit_cross_domain=omit)
    return QueryPlan(
        queries=queries[:max_queries],
        profile_summary="fixture initial wave",
    )


def followup_query_plan():
    """Legacy followup :class:`QueryPlan` with unique non-overlapping terms."""
    from radar.schema.papers import QueryPlan

    return QueryPlan(
        queries=_spec_queries(_FOLLOWUP_SPECS),
        profile_summary="fixture followup wave",
    )


def _canned_work(index: int, terms: str) -> dict:
    day = (index % 27) + 1
    return {
        "id": f"https://openalex.org/W{10_000 + index}",
        "title": f"Fixture study {index} on {terms[:40]}",
        "publication_date": f"2026-09-{day:02d}",
        "publication_year": 2026,
        "cited_by_count": index,
    }


def fake_pages() -> dict[str, dict]:
    """Canned OpenAlex payloads keyed by every fixture synthetic term."""
    pages: dict[str, dict] = {}
    index = 0
    for specs in (_INITIAL_SPECS, _FOLLOWUP_SPECS):
        for spec in specs:
            terms = spec[6]
            pages[terms] = {"results": [_canned_work(index, terms)]}
            index += 1
    return pages


def known_work_ids() -> list[str]:
    """Retrieved work IDs backing the canned pages (valid seed/refs)."""
    ids = []
    for index in range(len(_INITIAL_SPECS) + len(_FOLLOWUP_SPECS)):
        ids.append(f"https://openalex.org/W{10_000 + index}")
    return ids


__all__ = [
    "followup_plan_dict",
    "followup_query_plan",
    "fake_pages",
    "initial_plan_dict",
    "known_work_ids",
    "planner_model",
    "test_query_plan",
]
