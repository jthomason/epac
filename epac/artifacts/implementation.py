"""EPACImplementation – produced by the Actor, reviewed by the Critic."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


class FileChange(BaseModel):
    """A single file-level change produced by the Actor."""

    path: str
    action: str = Field(..., description="'created' | 'modified' | 'deleted'")
    diff: str = Field(default="", description="Unified diff of the change")
    content: str = Field(default="", description="Full file content (for new files)")
    language: str = Field(default="", description="Programming language")


class TestResult(BaseModel):
    """Outcome of a test suite run by the Actor."""

    suite: str = Field(..., description="Name of the test suite / runner")
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    duration_seconds: float = 0.0
    output: str = Field(default="", description="Raw test runner output")

    @property
    def all_passed(self) -> bool:
        return self.failed == 0


class BuildResult(BaseModel):
    success: bool
    command: str
    output: str = ""
    duration_seconds: float = 0.0


class EPACImplementation(BaseModel):
    """
    Actor Implementation Artifact.

    Produced by the Actor against a specific PlanTask (or set of tasks).
    Flows to the Critic for review.  Carries all evidence needed for the
    Critic to perform a complete review: diffs, test results, build logs.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    plan_id: str = Field(..., description="ID of the EPACPlan this implements")
    task_ids: list[str] = Field(..., description="IDs of PlanTasks completed in this batch")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    iteration: int = Field(
        default=1,
        description="Actor-Critic loop iteration count (increments on re-implementation)",
    )

    # Code changes
    file_changes: list[FileChange] = Field(default_factory=list)

    # Validation evidence
    test_results: list[TestResult] = Field(default_factory=list)
    build_results: list[BuildResult] = Field(default_factory=list)

    # Provenance
    pr_url: str = Field(default="", description="Pull request URL if GitHub integration enabled")
    branch_name: str = Field(default="")
    commit_sha: str = Field(default="")

    # Self-assessment by Actor
    actor_notes: str = Field(
        default="",
        description="Actor's notes on implementation decisions, trade-offs, or open questions",
    )
    known_issues: list[str] = Field(
        default_factory=list,
        description="Issues the Actor identified but did not resolve",
    )

    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def all_tests_passed(self) -> bool:
        return all(r.all_passed for r in self.test_results)

    @property
    def build_succeeded(self) -> bool:
        return all(r.success for r in self.build_results)
