"""Packaged Qwen screening reliability policy (service-free loader)."""

from __future__ import annotations

from radar.config.yaml import load_yaml
from radar.schema.triage import QwenScreeningPolicy


def qwen_screening_policy() -> QwenScreeningPolicy:
    """Load the immutable packaged screening policy once per process."""
    return load_yaml("radar.config", "qwen_screening.yaml", QwenScreeningPolicy)
