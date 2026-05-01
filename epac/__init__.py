"""
EPAC – Expert-Planner-Actor-Critic Framework
=============================================

An open-source Python SDK for building enterprise-grade agentic AI pipelines
with governed, auditable, human-in-the-loop workflows.

Pattern: Expert → Planner → Actor ⇌ Critic → Expert

Usage::

    from epac import EPACPipeline, EPACConfig
    from epac.roles import ExpertRole, PlannerRole, ActorRole, CriticRole

    config = EPACConfig(llm_model="openai/gpt-4o")
    pipeline = EPACPipeline(config=config)
    result = await pipeline.run(goal="Add OAuth2 login to the user service")
"""

from epac.artifacts import EPACSpec, EPACPlan, EPACImplementation, EPACReview
from epac.pipeline import EPACPipeline, EPACConfig, EPACResult
from epac.state import EPACState, StageStatus, RiskLevel, AutonomyLevel

__all__ = [
    # Artifacts
    "EPACSpec",
    "EPACPlan",
    "EPACImplementation",
    "EPACReview",
    # Pipeline
    "EPACPipeline",
    "EPACConfig",
    "EPACResult",
    # State
    "EPACState",
    "StageStatus",
    "RiskLevel",
    "AutonomyLevel",
]

__version__ = "0.1.4"
__author__ = "James Thomason"
