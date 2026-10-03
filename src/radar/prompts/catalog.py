"""Typed access to the packaged prompt documents."""

from radar.config.yaml import load_yaml
from radar.schema.configuration import AgentPrompt, AnalysisPrompt, ScreeningConfig


def screening_questions() -> ScreeningConfig:
    return load_yaml("radar.prompts", "screening_questions.yaml", ScreeningConfig)


def paper_triage_prompt() -> AgentPrompt:
    return load_yaml("radar.prompts", "paper_triage.yaml", AgentPrompt)


def opportunity_analysis_prompt() -> AnalysisPrompt:
    return load_yaml("radar.prompts", "opportunity_analysis.yaml", AnalysisPrompt)
