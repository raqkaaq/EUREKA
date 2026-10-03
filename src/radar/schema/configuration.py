"""Service-free contracts for editable search plans, questions and prompts."""

from __future__ import annotations

import hashlib
from string import Formatter
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from radar.config.runtime import (
    MAX_ANALYSIS_OPPORTUNITIES,
    MAX_PER_PAGE,
    MAX_PROMPT_CHARS,
    MAX_QUERIES,
    MAX_TERM_CHARS,
)
from radar.schema.opportunities import SpecialistRole

Text = Annotated[str, Field(strict=True, min_length=1, max_length=8000)]
Version = Annotated[int, Field(strict=True, ge=1, le=1)]


def validate_template(text: str, allowed: set[str]) -> str:
    """Allow named substitutions only: no attribute access, conversions or specs."""
    for _, name, spec, conversion in Formatter().parse(text):
        if name is not None and (name not in allowed or spec or conversion):
            raise ValueError("unsupported template placeholder")
    return text


class ConfigurationModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


class QuestionCriteria(ConfigurationModel):
    true_criterion: Text = Field(alias="true")
    false_criterion: Text = Field(alias="false")

    @field_validator("true_criterion", "false_criterion")
    @classmethod
    def _templates(cls, value: str) -> str:
        return validate_template(value, {"keywords", "domains"})


class ScreeningQuestion(ConfigurationModel):
    type: Literal["noul"]
    instructions: Text
    criteria: QuestionCriteria

    @field_validator("instructions")
    @classmethod
    def _template(cls, value: str) -> str:
        return validate_template(value, {"keywords", "domains"})


class ScreeningQuestions(ConfigurationModel):
    ai_ml_relevance: ScreeningQuestion
    cross_domain_potential: ScreeningQuestion


class ScreeningConfig(ConfigurationModel):
    version: Version
    rubric_version: Annotated[str, Field(strict=True, min_length=1, max_length=200)]
    questions: ScreeningQuestions

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(self.model_dump_json().encode("utf-8")).hexdigest()

    def render(self, *, keywords: str, domains: str) -> dict:
        values = {"keywords": keywords, "domains": domains}
        rendered = self.questions.model_dump(by_alias=True)
        for question in rendered.values():
            question["instructions"] = question["instructions"].format_map(values)
            question["criteria"] = {
                name: text.format_map(values)
                for name, text in question["criteria"].items()
            }
        return rendered


class AgentPrompt(ConfigurationModel):
    version: Version
    instructions: Text


class AnalysisPrompt(AgentPrompt):
    candidate_header: Text
    task_template: Text

    @property
    def max_opportunities(self) -> int:
        return MAX_ANALYSIS_OPPORTUNITIES

    @field_validator("task_template")
    @classmethod
    def _template(cls, value: str) -> str:
        return validate_template(value, {"max_opportunities", "valid_range"})

    @model_validator(mode="after")
    def _fits_empty_prompt(self) -> AnalysisPrompt:
        task = self.task_template.format(
            max_opportunities=self.max_opportunities,
            valid_range="none (no candidates)",
        )
        if len(self.instructions) + len(self.candidate_header) + len(task) + 4 > MAX_PROMPT_CHARS:
            raise ValueError("instructions and task exceed the prompt budget")
        return self


class SpecialistPrompt(AnalysisPrompt):
    role: SpecialistRole

    @property
    def max_opportunities(self) -> int:
        return 1


class SearchDefinition(ConfigurationModel):
    name: Annotated[str, Field(strict=True, min_length=1, max_length=80)]
    kind: Literal["semantic", "recent"]
    scope: Literal["broad", "cross_domain"] = "broad"
    terms: Annotated[str, Field(strict=True, min_length=1, max_length=MAX_TERM_CHARS)]
    repeat: Literal["once", "keywords"] = "once"
    limit: Annotated[int, Field(strict=True, ge=1, le=MAX_QUERIES)] = 1
    per_page: Annotated[int, Field(strict=True, ge=1, le=MAX_PER_PAGE)] = 25

    @model_validator(mode="after")
    def _template(self) -> SearchDefinition:
        allowed = {"keywords", "domains"}
        if self.repeat == "keywords":
            allowed.add("keyword")
        validate_template(self.terms, allowed)
        if self.repeat == "once" and self.limit != 1:
            raise ValueError("once searches have limit 1")
        return self


class SearchConfig(ConfigurationModel):
    version: Version
    queries: tuple[SearchDefinition, ...] = Field(min_length=1, max_length=MAX_QUERIES)

    @model_validator(mode="after")
    def _unique_names(self) -> SearchConfig:
        if len({query.name for query in self.queries}) != len(self.queries):
            raise ValueError("search names must be unique")
        return self
