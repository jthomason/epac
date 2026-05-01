"""Tests for EPAC typed artifacts."""

import pytest
from epac.artifacts.spec import EPACSpec, AcceptanceCriterion, Constraint, RiskLevel
from epac.artifacts.plan import EPACPlan, PlanTask, PlanArtifact
from epac.artifacts.implementation import EPACImplementation, FileChange, TestResult
from epac.artifacts.review import EPACReview, ReviewFinding, FindingSeverity, FindingCategory, SARIFReport


class TestEPACSpec:
    def test_basic_creation(self):
        spec = EPACSpec(
            created_by="test@example.com",
            title="Test feature",
            goal="Build something useful",
        )
        assert spec.title == "Test feature"
        assert spec.risk_level == RiskLevel.MEDIUM
        assert spec.requires_expert_plan_approval is True
        assert spec.id is not None

    def test_acceptance_criterion_text(self):
        ac = AcceptanceCriterion(
            given="a user is logged in",
            when="they click logout",
            then="their session is invalidated",
        )
        text = ac.to_text()
        assert "GIVEN" in text
        assert "WHEN" in text
        assert "THEN" in text

    def test_serialization_roundtrip(self):
        spec = EPACSpec(
            created_by="x@y.com",
            title="Roundtrip test",
            goal="Test serialization",
            acceptance_criteria=[
                AcceptanceCriterion(given="a", when="b", then="c")
            ],
            constraints=[
                Constraint(category="security", description="No secrets in logs")
            ],
        )
        data = spec.model_dump()
        spec2 = EPACSpec(**data)
        assert spec.id == spec2.id
        assert len(spec2.acceptance_criteria) == 1
        assert len(spec2.constraints) == 1


class TestEPACPlan:
    def test_basic_creation(self):
        task = PlanTask(
            title="Implement middleware",
            description="Add rate limiting middleware",
            artifacts=[PlanArtifact(path="app/middleware.py", action="create")],
        )
        plan = EPACPlan(
            spec_id="spec-123",
            spec_version=1,
            summary="Add rate limiting",
            tasks=[task],
        )
        assert len(plan.tasks) == 1
        assert plan.expert_approved is None

    def test_approval_state(self):
        plan = EPACPlan(
            spec_id="spec-123",
            spec_version=1,
            summary="Test plan",
            tasks=[],
        )
        approved_plan = plan.model_copy(update={"expert_approved": True})
        assert approved_plan.expert_approved is True


class TestEPACImplementation:
    def test_test_results(self):
        impl = EPACImplementation(
            plan_id="plan-123",
            task_ids=["task-1"],
            test_results=[
                TestResult(suite="pytest", passed=10, failed=0),
                TestResult(suite="mypy", passed=1, failed=1),
            ],
        )
        assert impl.all_tests_passed is False  # mypy failed

    def test_all_tests_passed(self):
        impl = EPACImplementation(
            plan_id="plan-123",
            task_ids=["task-1"],
            test_results=[TestResult(suite="pytest", passed=10, failed=0)],
        )
        assert impl.all_tests_passed is True


class TestEPACReview:
    def test_blocking_findings(self):
        review = EPACReview(
            implementation_id="impl-1",
            plan_id="plan-1",
            spec_id="spec-1",
            passed=False,
            findings=[
                ReviewFinding(
                    rule_id="sec/001",
                    severity=FindingSeverity.ERROR,
                    category=FindingCategory.SECURITY,
                    message="Hardcoded secret detected",
                    file_path="app/config.py",
                    line_start=42,
                ),
                ReviewFinding(
                    rule_id="style/001",
                    severity=FindingSeverity.NOTE,
                    category=FindingCategory.STYLE,
                    message="Line too long",
                ),
            ],
        )
        assert len(review.blocking_findings) == 1
        assert len(review.warning_findings) == 0

    def test_sarif_serialization(self):
        sarif = SARIFReport(
            findings=[
                ReviewFinding(
                    rule_id="test/001",
                    severity=FindingSeverity.ERROR,
                    category=FindingCategory.SECURITY,
                    message="Test finding",
                    file_path="src/main.py",
                    line_start=10,
                )
            ]
        )
        sarif_dict = sarif.to_sarif_dict()
        assert sarif_dict["version"] == "2.1.0"
        assert len(sarif_dict["runs"][0]["results"]) == 1
        result = sarif_dict["runs"][0]["results"][0]
        assert result["ruleId"] == "test/001"
        assert result["level"] == "error"
