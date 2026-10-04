"""Triage screening contracts (service-free: pydantic only).

CLEF/SystemOne per-paper screening outcomes. Probabilities are strict
0..1 floats: bools, strings, NaN/infinities, and out-of-range values are
rejected. A ``scored`` result carries both probabilities; every other
status carries none, so unknown works stay explicitly unknown.
"""

from __future__ import annotations

import math as _math
from enum import Enum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def _probability(value: object) -> object:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("probability must be a JSON number, not bool/string")
    number = float(value)
    if not _math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError("probability must be finite within 0..1")
    return number


TriageStatus = Literal[
    "scored", "missing_abstract", "failed", "deadline", "oversized"
]


class ScreeningFailureKind(str, Enum):
    """Typed screening failure diagnosis (local Qwen path only)."""

    CONNECT_TIMEOUT = "connect_timeout"
    READ_TIMEOUT = "read_timeout"
    REQUEST_TIMEOUT = "request_timeout"
    INVALID_RESPONSE = "invalid_response"
    SERVICE_ERROR = "service_error"
    UNEXPECTED_ERROR = "unexpected_error"
    BUDGET_EXHAUSTED = "budget_exhausted"


class TriageResult(BaseModel):
    """One work's screening outcome (model-set fields only)."""

    work_id: str = Field(min_length=1, max_length=500)
    status: TriageStatus
    ai_ml_relevance: float | None = None
    cross_domain_potential: float | None = None
    failure_kind: ScreeningFailureKind | None = None

    @field_validator(
        "ai_ml_relevance", "cross_domain_potential", mode="before"
    )
    @classmethod
    def _strict_probability(cls, value: object) -> object:
        return _probability(value)

    @model_validator(mode="after")
    def _status_probability_agreement(self) -> "TriageResult":
        probs = (self.ai_ml_relevance, self.cross_domain_potential)
        if self.status == "scored":
            if any(p is None for p in probs):
                raise ValueError("scored results require both probabilities")
        elif any(p is not None for p in probs):
            raise ValueError(f"{self.status} results must not carry probabilities")
        if self.status in ("scored", "missing_abstract", "oversized"):
            if self.failure_kind is not None:
                raise ValueError(f"{self.status} results must not carry failure_kind")
        return self


def _finite_timeout(value: object, name: str, low: float, high: float) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number within ({low}, {high}]")
    if not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number within ({low}, {high}]")
    number = float(value)
    if not _math.isfinite(number) or not (low < number <= high):
        raise ValueError(f"{name} must be a finite number within ({low}, {high}]")
    return number


class QwenScreeningPolicy(BaseModel):
    """Immutable packaged Qwen screening reliability policy."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: Annotated[int, Field(strict=True, ge=1, le=1)]
    batch_size: Annotated[int, Field(strict=True, ge=1, le=24)] = 2
    concurrency: Annotated[int, Field(strict=True, ge=1, le=2)] = 1
    request_timeout_s: float = 90.0
    overall_timeout_s: float = 1800.0

    @field_validator("request_timeout_s", mode="before")
    @classmethod
    def _check_request_timeout(cls, value: object) -> float:
        return _finite_timeout(value, "request_timeout_s", 0, 300)

    @field_validator("overall_timeout_s", mode="before")
    @classmethod
    def _check_overall_timeout(cls, value: object) -> float:
        return _finite_timeout(value, "overall_timeout_s", 0, 3600)


class TriageBatch(BaseModel):
    """Whole-pool screening outcome with model/rubric provenance."""

    model_id: str = Field(min_length=1, max_length=200)
    rubric_version: str = Field(min_length=1, max_length=200)
    rubric_hash: str = Field(default="", pattern=r"^(?:[a-f0-9]{64})?$")
    backend: Literal["clef", "qwen"] = "clef"
    probability_kind: Literal["native_noul", "prompted_estimate"] = "native_noul"
    fallback_reason: Literal[
        "missing_endpoint", "screening_failed", "deadline", "screening_error"
    ] | None = None
    results: list[TriageResult] = Field(default_factory=list)


class NoulAnswer(BaseModel):
    """Shared strict yes/no probability answer, regardless of transport."""

    type: Literal["noul"]
    noul: float

    @field_validator("noul", mode="before")
    @classmethod
    def _strict_probability(cls, value: object) -> object:
        return _probability(value)


class SystemOneAnswers(BaseModel):
    ai_ml_relevance: NoulAnswer
    cross_domain_potential: NoulAnswer


class SystemOneResponse(BaseModel):
    """Same validated answer contract for CLEF and Qwen chat fallback."""

    model: str = Field(min_length=1, max_length=200)
    answers: SystemOneAnswers

    def to_result(self, work_id: str) -> TriageResult:
        return TriageResult(
            work_id=work_id, status="scored",
            ai_ml_relevance=self.answers.ai_ml_relevance.noul,
            cross_domain_potential=self.answers.cross_domain_potential.noul)


class QwenPaperResponse(SystemOneResponse):
    """Batch correlation metadata around an unchanged SystemOne response."""

    work_id: str = Field(min_length=1, max_length=500)


class QwenResponses(BaseModel):
    model_config = ConfigDict(extra="forbid")
    responses: list[QwenPaperResponse] = Field(min_length=1, max_length=24)
