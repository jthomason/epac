"""EPAC pipeline state – the shared LangGraph state object."""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Any

from pydantic import BaseModel, Field

from epac.artifacts import EPACSpec, EPACPlan, EPACImplementation, EPACReview


class StageStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    AWAITING_APPROVAL = "awaiting_approval"
    APPROVED = "approved"
    REJECTED = "rejected"
    COMPLETED = "completed"
    FAILED = "failed"


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AutonomyLevel(int, Enum):
    """
    Five-level autonomy model from the EPAC framework.

    Level 1: Autonomous – read-only / informational; Actor executes, Critic monitors
    Level 2: Monitored – low-impact modifications; Critic reviews post-hoc
    Level 3: Gated – medium-impact; Expert approves before Actor executes
    Level 4: Dual Control – high-impact; Expert approves plan AND final output
    Level 5: Human-Only – critical/irreversible; AI recommends, Expert executes
    """

    AUTONOMOUS = 1
    MONITORED = 2
    GATED = 3
    DUAL_CONTROL = 4
    HUMAN_ONLY = 5


class AuditEntry(BaseModel):
    """A single immutable audit log entry at an EPAC stage boundary."""

    timestamp: str  # ISO-8601 UTC
    stage: str      # 'expert' | 'planner' | 'actor' | 'critic'
    event: str      # E.g. 'spec_submitted', 'plan_generated', 'review_passed'
    actor_id: str   # Who or what performed the action
    artifact_id: str | None = None
    risk_level: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class EPACState(BaseModel):
    """
    Shared state object threaded through the LangGraph pipeline.

    Each node reads from and writes to this object.  LangGraph persists it
    to a checkpointer so the pipeline is resumable across human approval waits.
    """

    # Identity
    pipeline_id: str
    thread_id: str = ""  # LangGraph thread ID for checkpoint/resume

    # Core artifacts (None = not yet produced)
    spec: EPACSpec | None = None
    plan: EPACPlan | None = None
    implementations: list[EPACImplementation] = Field(default_factory=list)
    reviews: list[EPACReview] = Field(default_factory=list)

    # Stage status
    expert_status: StageStatus = StageStatus.PENDING
    planner_status: StageStatus = StageStatus.PENDING
    actor_status: StageStatus = StageStatus.PENDING
    critic_status: StageStatus = StageStatus.PENDING

    # Loop control
    actor_critic_iteration: int = 0
    max_actor_critic_iterations: int = 5

    # Governance
    autonomy_level: AutonomyLevel = AutonomyLevel.GATED
    current_stage: str = "expert"

    # Human decision slot (populated by Expert interrupt handlers)
    pending_decision: str | None = None    # 'approve' | 'reject' | 'edit'
    pending_decision_notes: str = ""

    # Audit trail
    audit_log: list[AuditEntry] = Field(default_factory=list)

    # Pipeline outcome
    completed: bool = False
    failed: bool = False
    failure_reason: str = ""

    # Pass-through metadata
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def latest_implementation(self) -> EPACImplementation | None:
        return self.implementations[-1] if self.implementations else None

    @property
    def latest_review(self) -> EPACReview | None:
        return self.reviews[-1] if self.reviews else None

    @property
    def critic_passed(self) -> bool:
        review = self.latest_review
        return review is not None and review.passed

    @property
    def actor_critic_loop_exhausted(self) -> bool:
        return self.actor_critic_iteration >= self.max_actor_critic_iterations
