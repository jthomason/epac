"""EPACReview – produced by the Critic, surfaces to Expert on pass."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class FindingSeverity(str, Enum):
    """Maps to SARIF 2.1.0 severity levels."""

    ERROR = "error"       # Must fix before passing
    WARNING = "warning"   # Should fix; Critic may allow pass with justification
    NOTE = "note"         # Informational
    NONE = "none"


class FindingCategory(str, Enum):
    SECURITY = "security"
    CORRECTNESS = "correctness"
    PERFORMANCE = "performance"
    STYLE = "style"
    POLICY = "policy"
    GOAL_ALIGNMENT = "goal_alignment"
    TEST_COVERAGE = "test_coverage"
    OTHER = "other"


class ReviewFinding(BaseModel):
    """A single Critic finding, serializable to SARIF 2.1.0."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    rule_id: str = Field(..., description="Critic rule or check identifier")
    severity: FindingSeverity
    category: FindingCategory = FindingCategory.OTHER
    message: str
    file_path: str = Field(default="")
    line_start: int | None = None
    line_end: int | None = None
    suggested_fix: str = Field(default="", description="Concrete code or action suggestion")
    references: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SARIFReport(BaseModel):
    """
    Minimal SARIF 2.1.0 wrapper.

    The full SARIF JSON is stored in `raw`; EPAC uses the parsed `findings` list
    for routing decisions.  SARIF output is natively supported by GitHub Code Scanning,
    SonarQube, and CodeQL.
    """

    schema_uri: str = "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/master/Schemata/sarif-schema-2.1.0.json"
    version: str = "2.1.0"
    findings: list[ReviewFinding] = Field(default_factory=list)
    raw: dict[str, Any] = Field(default_factory=dict)

    def to_sarif_dict(self) -> dict[str, Any]:
        """Serialize findings to a SARIF 2.1.0-compliant dict."""
        results = []
        for f in self.findings:
            result: dict[str, Any] = {
                "ruleId": f.rule_id,
                "level": f.severity.value,
                "message": {"text": f.message},
            }
            if f.file_path:
                location: dict[str, Any] = {
                    "physicalLocation": {
                        "artifactLocation": {"uri": f.file_path, "uriBaseId": "%SRCROOT%"}
                    }
                }
                if f.line_start is not None:
                    location["physicalLocation"]["region"] = {"startLine": f.line_start}
                result["locations"] = [location]
            if f.suggested_fix:
                result["fixes"] = [{"description": {"text": f.suggested_fix}}]
            results.append(result)

        return {
            "$schema": self.schema_uri,
            "version": self.version,
            "runs": [{"tool": {"driver": {"name": "epac-critic"}}, "results": results}],
        }


class EPACReview(BaseModel):
    """
    Critic Review Artifact.

    Produced by the Critic for each EPACImplementation.  Drives the Actor-Critic
    loop: if `passed` is False, findings are fed back to the Actor.  If `passed`
    is True, the implementation surfaces to the Expert for final approval.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    implementation_id: str
    plan_id: str
    spec_id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    iteration: int = Field(default=1, description="Matches EPACImplementation.iteration")

    # Verdict
    passed: bool = Field(
        ...,
        description="True when all mandatory checks pass and quality thresholds are met",
    )
    pass_rationale: str = Field(default="")
    fail_rationale: str = Field(default="", description="Summary of why review failed")

    # Quality scores (0.0–1.0)
    security_score: float | None = None
    correctness_score: float | None = None
    test_coverage_score: float | None = None

    # Findings
    findings: list[ReviewFinding] = Field(default_factory=list)
    sarif: SARIFReport = Field(default_factory=SARIFReport)

    # Goal alignment check
    goal_alignment_verified: bool = False
    goal_alignment_notes: str = ""

    # Prompt injection / security posture checks
    prompt_injection_risk: str = Field(
        default="none",
        description="'none' | 'low' | 'medium' | 'high'",
    )

    # Feedback for Actor on re-iteration
    actor_feedback: str = Field(
        default="",
        description="Actionable instructions for the Actor if passed=False",
    )

    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def blocking_findings(self) -> list[ReviewFinding]:
        return [f for f in self.findings if f.severity == FindingSeverity.ERROR]

    @property
    def warning_findings(self) -> list[ReviewFinding]:
        return [f for f in self.findings if f.severity == FindingSeverity.WARNING]
