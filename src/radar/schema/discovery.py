"""Service-free adaptive discovery policy, model plans and observed retrieval memory."""

from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator
from radar.schema.papers import CollectedWork, PlannedQuery, QueryPlan


class DiscoveryModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


class LearningGoal(DiscoveryModel):
    id: str = Field(pattern=r"^[a-z][a-z0-9_]{0,79}$")
    question: str = Field(min_length=1, max_length=1000)
    scope: Literal["broad", "cross_domain"] = "broad"


class DiscoveryPolicy(DiscoveryModel):
    version: Literal[3]
    learning_goals: tuple[LearningGoal, ...] = Field(min_length=1, max_length=8)
    initial_queries: StrictInt = Field(default=6, ge=4, le=6)
    followup_queries: StrictInt = Field(default=6, ge=1, le=6)
    results_per_query: StrictInt = Field(default=15, ge=1, le=15)
    planning_timeout_s: float = Field(default=120, gt=0, le=300, allow_inf_nan=False)
    planner_max_tokens: StrictInt = Field(default=3000, ge=128, le=4000)

    @model_validator(mode="after")
    def _unique_goals(self):
        if len({g.id for g in self.learning_goals}) != len(self.learning_goals):
            raise ValueError("Learning goal identities must be unique")
        return self


class SearchIntent(DiscoveryModel):
    learning_goal_id: str = Field(min_length=1, max_length=80)
    question: str = Field(min_length=1, max_length=500)
    rationale: str = Field(min_length=1, max_length=400)
    expected_learning_value: str = Field(min_length=1, max_length=300)
    origin: Literal["agenda", "finding", "open_question", "exploration"]
    source_work_ids: list[str] = Field(default_factory=list, max_length=3)
    query: PlannedQuery


class SearchWavePlan(DiscoveryModel):
    summary: str = Field(min_length=1, max_length=500)
    intents: list[SearchIntent] = Field(min_length=1, max_length=6)

    @property
    def query_plan(self) -> QueryPlan:
        return QueryPlan(queries=[i.query for i in self.intents], profile_summary=self.summary)


class QueryFeedback(DiscoveryModel):
    """Observed metadata availability/yield, never scientific quality ratings."""
    query: PlannedQuery
    returned: StrictInt = Field(ge=0, le=50)
    accepted: StrictInt = Field(ge=0, le=50)
    new_to_run: StrictInt = Field(ge=0, le=50)
    new_to_library: StrictInt = Field(ge=0, le=50)
    with_abstract: StrictInt = Field(ge=0, le=50)
    with_pdf: StrictInt = Field(ge=0, le=50)
    status: Literal["ok", "malformed", "skipped_budget"] = "ok"


class RetrievalResult(DiscoveryModel):
    works: list[CollectedWork] = Field(default_factory=list, max_length=200)
    feedback: list[QueryFeedback] = Field(default_factory=list, max_length=12)


class SavedFinding(DiscoveryModel):
    work_id: str = Field(min_length=1, max_length=500)
    title: str = Field(default="", max_length=2000)
    contribution: str = Field(default="", max_length=600)
    limits: list[str] = Field(default_factory=list, max_length=6)
    open_questions: list[str] = Field(default_factory=list, max_length=3)
    evidence_level: Literal["abstract", "pdf_text", "metadata"] = "metadata"


class DiscoveryMemory(DiscoveryModel):
    findings: list[SavedFinding] = Field(default_factory=list, max_length=6)
    known_work_ids: list[str] = Field(default_factory=list, max_length=100)
    prior_feedback: list[QueryFeedback] = Field(default_factory=list, max_length=24)


class SearchWaveRecord(DiscoveryModel):
    wave: StrictInt = Field(ge=1, le=2)
    profile_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    policy_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    origin: Literal["generated", "cached"] = "generated"
    plan: SearchWavePlan
    feedback: list[QueryFeedback] = Field(default_factory=list, max_length=6)
    retrieval_failure: Literal["source_error"] | None = None


class CachedSearchPlan(DiscoveryModel):
    run_id: str = Field(min_length=1, max_length=128)
    started_at: str = Field(min_length=1, max_length=100)
    record: SearchWaveRecord
