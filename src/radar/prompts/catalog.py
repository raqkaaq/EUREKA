"""Typed access to the packaged prompt documents."""

from radar.config.yaml import ConfigurationError, load_yaml
from radar.schema.configuration import (
    AgentPrompt, AnalysisPrompt, ScreeningConfig, SpecialistPrompt, SpecialistRole,
)

SPECIALIST_ROLES: tuple[SpecialistRole, ...] = (
    "ml_methods", "behavioral_economics", "evidence_review",
)


def screening_questions() -> ScreeningConfig:
    return load_yaml("radar.prompts", "screening_questions.yaml", ScreeningConfig)


def paper_triage_prompt() -> AgentPrompt:
    return load_yaml("radar.prompts", "paper_triage.yaml", AgentPrompt)


def opportunity_analysis_prompt() -> AnalysisPrompt:
    return load_yaml("radar.prompts", "opportunity_analysis.yaml", AnalysisPrompt)


def specialist_prompt(role: SpecialistRole) -> SpecialistPrompt:
    if role not in SPECIALIST_ROLES:
        raise ConfigurationError("unsupported specialist role")
    prompt = load_yaml("radar.prompts", f"{role}.yaml", SpecialistPrompt)
    if prompt.role != role:
        raise ConfigurationError("specialist document has the wrong role")
    return prompt
