"""EPACPlan – the typed artifact produced by the Planner and consumed by the Actor."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    BLOCKED = "blocked"


class PlanArtifact(BaseModel):
    """A file or resource the Actor should produce or modify."""

    path: str = Field(..., description="Relative path within the repository")
    action: str = Field(..., description="'create' | 'modify' | 'delete'")
    description: str = Field(default="", description="What the change should accomplish")


class PlanTask(BaseModel):
    """A single, independently implementable unit of work for the Actor."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    description: str
    depends_on: list[str] = Field(
        default_factory=list,
        description="IDs of tasks that must complete before this one starts",
    )
    artifacts: list[PlanArtifact] = Field(default_factory=list)
    test_strategy: str = Field(
        default="",
        description="How Actor should validate this task is complete",
    )
    estimated_complexity: str = Field(
        default="medium",
        description="'trivial' | 'low' | 'medium' | 'high'",
    )
    status: TaskStatus = TaskStatus.PENDING
    metadata: dict[str, Any] = Field(default_factory=dict)


class DesignDecision(BaseModel):
    """An architectural or design choice the Planner made explicit."""

    title: str
    decision: str
    rationale: str
    alternatives_considered: list[str] = Field(default_factory=list)


class EPACPlan(BaseModel):
    """
    Planner Artifact.

    Produced by the Planner from an EPACSpec.  Flows to the Actor as a structured
    implementation specification — tasks with dependencies, affected files, design
    decisions, and non-functional guidance.

    If spec.requires_expert_plan_approval is True, this artifact is presented to the
    Expert before the Actor begins work.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    spec_id: str = Field(..., description="ID of the EPACSpec this plan implements")
    spec_version: int = Field(..., description="Version of the spec at planning time")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Summary
    summary: str = Field(..., description="Plain-language summary of the implementation approach")
    tasks: list[PlanTask] = Field(..., description="Ordered, dependency-linked task list")

    # Architecture
    design_decisions: list[DesignDecision] = Field(default_factory=list)
    affected_components: list[str] = Field(
        default_factory=list,
        description="System components / services touched by this plan",
    )
    api_contracts: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Interface definitions (OpenAPI snippets, TypedDicts, etc.)",
    )

    # Non-functional guidance for Actor
    security_guidance: str = Field(default="")
    performance_guidance: str = Field(default="")
    testing_guidance: str = Field(default="")

    # Expert approval
    expert_approved: bool | None = Field(
        default=None,
        description="None = awaiting review, True = approved, False = rejected",
    )
    expert_feedback: str = Field(default="")

    metadata: dict[str, Any] = Field(default_factory=dict)
