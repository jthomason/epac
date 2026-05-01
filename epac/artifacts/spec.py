"""EPACSpec – the typed artifact produced by the Expert and consumed by the Planner."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class RiskLevel(str, Enum):
    """Risk classification that drives autonomy gating and audit requirements."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AcceptanceCriterion(BaseModel):
    """A single GIVEN/WHEN/THEN acceptance criterion."""

    given: str = Field(..., description="Precondition / context")
    when: str = Field(..., description="Triggering action")
    then: str = Field(..., description="Expected outcome")

    def to_text(self) -> str:
        return f"GIVEN {self.given} WHEN {self.when} THEN {self.then}"


class Constraint(BaseModel):
    """A non-functional requirement or policy constraint."""

    category: str = Field(
        ...,
        description="E.g. 'security', 'performance', 'compliance', 'style'",
    )
    description: str
    mandatory: bool = True


class EPACSpec(BaseModel):
    """
    Expert Specification Artifact.

    The Expert (human or human-assisted) produces this artifact.  It flows to the
    Planner as the authoritative source of intent, context, and acceptance criteria.

    Follows AGENTS.md conventions for structure and Kiro-style EARS notation for
    acceptance criteria.
    """

    # Identity
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    version: int = Field(default=1, description="Incremented on each Expert revision")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    created_by: str = Field(..., description="Expert identifier (e.g. email or username)")

    # Goal
    title: str = Field(..., description="Short imperative title, e.g. 'Add OAuth2 login'")
    goal: str = Field(..., description="High-level description of what must be achieved")
    background: str = Field(
        default="",
        description="Domain context, prior art, architectural notes",
    )

    # Scope
    in_scope: list[str] = Field(default_factory=list)
    out_of_scope: list[str] = Field(default_factory=list)

    # Requirements
    acceptance_criteria: list[AcceptanceCriterion] = Field(
        default_factory=list,
        description="GIVEN/WHEN/THEN statements that define Done",
    )
    constraints: list[Constraint] = Field(
        default_factory=list,
        description="Non-functional requirements and policy constraints",
    )

    # Risk & governance
    risk_level: RiskLevel = Field(
        default=RiskLevel.MEDIUM,
        description="Drives autonomy gating (Level 1–5)",
    )
    requires_expert_plan_approval: bool = Field(
        default=True,
        description="If True, Expert must approve EPACPlan before Actor runs",
    )
    requires_expert_final_approval: bool = Field(
        default=True,
        description="If True, Expert must approve before merge/deploy",
    )

    # Metadata passthrough
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = {
        "json_schema_extra": {
            "example": {
                "created_by": "james@example.com",
                "title": "Add OAuth2 login to user service",
                "goal": (
                    "Replace username/password auth with OAuth2 (Google + GitHub) "
                    "for the public-facing user service API."
                ),
                "acceptance_criteria": [
                    {
                        "given": "a user visits /login",
                        "when": "they click 'Login with Google'",
                        "then": "they are redirected to Google OAuth and returned with a valid JWT",
                    }
                ],
                "constraints": [
                    {
                        "category": "security",
                        "description": "No OAuth tokens stored in plaintext",
                        "mandatory": True,
                    }
                ],
                "risk_level": "high",
            }
        }
    }
