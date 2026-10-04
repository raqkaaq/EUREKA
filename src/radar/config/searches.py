"""Expand a learning agenda into bounded retrieval, without screening papers."""

from __future__ import annotations

import datetime as dt

from pydantic import ValidationError

from radar.config.interests import RadarProfile
from radar.config.runtime import MAX_QUERIES, MAX_TERM_CHARS
from radar.config.yaml import ConfigurationError, load_yaml
from radar.schema.configuration import SearchConfig
from radar.schema.papers import PlannedQuery, QueryPlan


def search_config() -> SearchConfig:
    return load_yaml("radar.config", "searches.yaml", SearchConfig)


def _phrase(term: str) -> str:
    # Profile terms are literal phrases, not injected Boolean expressions.
    return '"' + term.replace('\\', '\\\\').replace('"', '\\"') + '"'


def build_query_plan(
    profile: RadarProfile,
    max_queries: int = MAX_QUERIES,
    *,
    configuration: SearchConfig | None = None,
    today: dt.date | None = None,
) -> QueryPlan:
    """Render historical/fresh research branches (or legacy v1 templates).

    Invalid/oversized terms fail rather than truncating Boolean syntax. Empty
    domain branches are omitted; exact duplicate requests are deduplicated.
    Configuration cannot raise the code-owned request/page/pool caps.
    """
    profile = RadarProfile.model_validate(profile.model_dump())
    config = configuration if configuration is not None else search_config()
    bound = max(1, min(int(max_queries), 6 if config.version == 1 else MAX_QUERIES))
    date = (
        (today or dt.date.today()) - dt.timedelta(days=profile.lookback_days)
    ).isoformat()
    values = {
        "keywords": " OR ".join(_phrase(term) for term in profile.keywords),
        "domains": " OR ".join(_phrase(term) for term in profile.domains),
    }
    queries: list[PlannedQuery] = []
    if config.version == 2:
        # Round-robin roles so smaller caps cover more questions.
        questions = [q for q in config.learning_questions
                     if q.scope != "cross_domain" or profile.domains]
        prose = {"keywords": ", ".join(profile.keywords), "domains": ", ".join(profile.domains)}
        for role in ("frontier", "foundation", "counterevidence"):
            for question in questions:
                definition = next(s for s in question.searches if s.role == role)
                rendered_values = prose if definition.kind == "semantic" else values
                try:
                    queries.append(PlannedQuery(
                        kind=definition.kind, terms=definition.terms.format_map(rendered_values),
                        per_page=definition.per_page,
                        from_date=date if role == "frontier" else None,
                        question_id=question.id, role=role,
                    ))
                except ValidationError:
                    raise ConfigurationError("searches.yaml: rendered research search exceeds its bound") from None
                if len(queries) == bound:
                    break
            if len(queries) == bound:
                break
        if not queries:
            raise ConfigurationError("searches.yaml: no research questions apply to the active profile")
        return QueryPlan(queries=queries, profile_summary=f"searches=v2 research_questions={len(questions)}")
    seen: set[tuple[str, str]] = set()
    for definition in config.queries:
        if definition.scope == "cross_domain" and not profile.domains:
            continue
        variants = (
            profile.keywords[:definition.limit]
            if definition.repeat == "keywords" else [None]
        )
        for keyword in variants:
            rendered = definition.terms.format_map(
                values | ({"keyword": _phrase(keyword)} if keyword is not None else {})
            ).strip()
            identity = (definition.kind, rendered)
            if identity in seen:
                continue
            try:
                query = PlannedQuery(
                    # v1 used the misleading "semantic" name for keyword search.
                    kind="keyword" if definition.kind == "semantic" else definition.kind,
                    terms=rendered,
                    per_page=definition.per_page, from_date=date,
                )
            except ValidationError:
                raise ConfigurationError(
                    f"searches.yaml: rendered query is empty or exceeds {MAX_TERM_CHARS} characters; "
                    "shorten the profile or split the configured search"
                ) from None
            queries.append(query)
            seen.add(identity)
            if len(queries) == bound:
                break
        if len(queries) == bound:
            break
    if not queries:
        raise ConfigurationError("searches.yaml: no searches apply to the active profile")
    return QueryPlan(
        queries=queries,
        profile_summary=(f"searches=v{config.version} keywords={len(profile.keywords)} "
                         f"domains={len(profile.domains)}"),
    )
