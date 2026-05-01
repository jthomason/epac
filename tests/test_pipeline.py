"""Tests for EPACPipeline orchestrator and HITL logic.

Covers:
1. EPACPipeline compiles without error (LangGraph available check)
2. Simple sequential _run_simple completes with mocked Planner/Actor/Critic
3. approve_plan and approve_implementation correctly update state
4. actor_critic_loop_exhausted triggers failed=True
"""
from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from epac.artifacts.implementation import EPACImplementation, FileChange
from epac.artifacts.plan import EPACPlan, PlanTask
from epac.artifacts.review import EPACReview, ReviewFinding, FindingSeverity, FindingCategory
from epac.artifacts.spec import EPACSpec
from epac.pipeline import EPACConfig, EPACPipeline, EPACResult
from epac.state import EPACState, AutonomyLevel, StageStatus


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_spec() -> EPACSpec:
    return EPACSpec(
        created_by="test@example.com",
        title="Test feature",
        goal="Implement a test feature",
    )


def _make_plan(spec_id: str) -> EPACPlan:
    task = PlanTask(
        title="Write the code",
        description="Implement the feature",
        test_strategy="Run pytest",
    )
    return EPACPlan(
        spec_id=spec_id,
        spec_version=1,
        summary="Minimal test plan",
        tasks=[task],
    )


def _make_implementation(plan_id: str, task_ids: list[str], iteration: int = 1) -> EPACImplementation:
    return EPACImplementation(
        plan_id=plan_id,
        task_ids=task_ids,
        iteration=iteration,
        file_changes=[
            FileChange(path="app/main.py", action="created", content="# hello\n")
        ],
    )


def _make_review(
    impl_id: str,
    plan_id: str,
    spec_id: str,
    passed: bool = True,
    iteration: int = 1,
) -> EPACReview:
    return EPACReview(
        implementation_id=impl_id,
        plan_id=plan_id,
        spec_id=spec_id,
        passed=passed,
        pass_rationale="Looks good" if passed else "",
        fail_rationale="" if passed else "Has errors",
        actor_feedback="" if passed else "Fix the issue",
        iteration=iteration,
    )


# ── Test 1: Pipeline compiles without error ───────────────────────────────────

class TestPipelineCompilation:
    def test_pipeline_creates_without_langgraph(self):
        """EPACPipeline instantiates even when LangGraph is not available."""
        pipeline = EPACPipeline()
        assert pipeline._expert is not None
        assert pipeline._planner is not None
        assert pipeline._actor is not None
        assert pipeline._critic is not None

    def test_pipeline_config_defaults(self):
        config = EPACConfig()
        pipeline = EPACPipeline(config=config)
        assert pipeline.config.max_actor_critic_iterations == 5
        assert pipeline.config.autonomy_level == AutonomyLevel.GATED

    def test_pipeline_with_langgraph(self):
        """If LangGraph is installed, graph compiles without error."""
        try:
            from langgraph.graph import StateGraph  # noqa: F401
            langgraph_available = True
        except ImportError:
            langgraph_available = False

        pipeline = EPACPipeline()
        if langgraph_available:
            assert pipeline._graph is not None
        else:
            assert pipeline._graph is None


# ── Test 2: _run_simple completes with mocked LLM ────────────────────────────

class TestRunSimple:
    @pytest.mark.asyncio
    async def test_run_simple_completes(self):
        """_run_simple runs end-to-end with mocked LLM calls."""
        spec = _make_spec()

        plan = _make_plan(spec.id)
        impl = _make_implementation(plan.id, [t.id for t in plan.tasks])
        review = _make_review(impl.id, plan.id, spec.id, passed=True)

        # Planner returns a valid plan dict
        planner_return = {
            "plan": plan.model_dump(),
            "planner_status": StageStatus.COMPLETED.value,
            "current_stage": "expert",  # Needs plan approval gate
        }
        # Actor returns a valid implementation dict
        actor_return = {
            "implementations": [impl.model_dump()],
            "actor_status": StageStatus.COMPLETED.value,
            "actor_critic_iteration": 1,
            "current_stage": "critic",
        }
        # Critic returns a passing review dict
        critic_return = {
            "reviews": [review.model_dump()],
            "critic_status": StageStatus.COMPLETED.value,
            "current_stage": "complete",
            "completed": True,
        }

        planner_mock = AsyncMock(return_value=planner_return)
        actor_mock = AsyncMock(return_value=actor_return)
        critic_mock = AsyncMock(return_value=critic_return)

        pipeline = EPACPipeline()
        pipeline._planner.run = planner_mock
        pipeline._actor.run = actor_mock
        pipeline._critic.run = critic_mock

        result = await pipeline._run_simple("test-pid-001", spec)

        assert isinstance(result, EPACResult)
        assert result.completed is True
        assert result.failed is False
        planner_mock.assert_called_once()
        actor_mock.assert_called_once()
        critic_mock.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_simple_via_run(self):
        """pipeline.run() uses _run_simple when LangGraph is unavailable."""
        import epac.pipeline as pipeline_mod

        spec = _make_spec()
        plan = _make_plan(spec.id)
        impl = _make_implementation(plan.id, [t.id for t in plan.tasks])
        review = _make_review(impl.id, plan.id, spec.id, passed=True)

        planner_return = {
            "plan": plan.model_dump(),
            "planner_status": StageStatus.COMPLETED.value,
            "current_stage": "expert",
        }
        actor_return = {
            "implementations": [impl.model_dump()],
            "actor_status": StageStatus.COMPLETED.value,
            "actor_critic_iteration": 1,
            "current_stage": "critic",
        }
        critic_return = {
            "reviews": [review.model_dump()],
            "critic_status": StageStatus.COMPLETED.value,
            "current_stage": "complete",
            "completed": True,
        }

        pipeline = EPACPipeline()
        pipeline._planner.run = AsyncMock(return_value=planner_return)
        pipeline._actor.run = AsyncMock(return_value=actor_return)
        pipeline._critic.run = AsyncMock(return_value=critic_return)

        # Force non-LangGraph path
        original = pipeline_mod._LANGGRAPH_AVAILABLE
        pipeline_mod._LANGGRAPH_AVAILABLE = False
        try:
            result = await pipeline.run(goal="Implement a test feature")
        finally:
            pipeline_mod._LANGGRAPH_AVAILABLE = original

        assert result.completed is True


# ── Test 3: approve_plan and approve_implementation update state ──────────────

class TestHITLDecisions:
    @pytest.mark.asyncio
    async def test_approve_plan_updates_state(self):
        """approve_plan sets plan.expert_approved=True and current_stage='actor'."""
        spec = _make_spec()
        plan = _make_plan(spec.id)
        # plan.expert_approved is None (awaiting approval)

        pipeline_id = str(uuid.uuid4())
        state = EPACState(
            pipeline_id=pipeline_id,
            spec=spec,
            plan=plan,
            expert_status=StageStatus.AWAITING_APPROVAL,
        )
        pipeline = EPACPipeline()
        pipeline._thread_states[pipeline_id] = state.model_dump()

        # Simulate approve_plan in non-LangGraph mode
        import epac.pipeline as pipeline_mod
        original = pipeline_mod._LANGGRAPH_AVAILABLE
        pipeline_mod._LANGGRAPH_AVAILABLE = False
        pipeline._graph = None
        try:
            await pipeline.approve_plan(pipeline_id, notes="Looks great")
        finally:
            pipeline_mod._LANGGRAPH_AVAILABLE = original

        stored = pipeline._thread_states[pipeline_id]
        # pending_decision should be injected; stored is now an EPACState
        if hasattr(stored, "pending_decision"):
            assert stored.pending_decision == "approve"
            assert stored.pending_decision_notes == "Looks great"
        else:
            assert stored.get("pending_decision") == "approve"
            assert stored.get("pending_decision_notes") == "Looks great"

    @pytest.mark.asyncio
    async def test_approve_implementation_updates_pending_decision(self):
        """approve_implementation injects 'approve' into pending_decision."""
        pipeline_id = str(uuid.uuid4())
        pipeline = EPACPipeline()
        pipeline._thread_states[pipeline_id] = {"pipeline_id": pipeline_id}

        import epac.pipeline as pipeline_mod
        original = pipeline_mod._LANGGRAPH_AVAILABLE
        pipeline_mod._LANGGRAPH_AVAILABLE = False
        pipeline._graph = None
        try:
            await pipeline.approve_implementation(pipeline_id, notes="Ship it")
        finally:
            pipeline_mod._LANGGRAPH_AVAILABLE = original

        stored = pipeline._thread_states[pipeline_id]
        if hasattr(stored, "pending_decision"):
            assert stored.pending_decision == "approve"
            assert stored.pending_decision_notes == "Ship it"
        else:
            assert stored.get("pending_decision") == "approve"
            assert stored.get("pending_decision_notes") == "Ship it"

    @pytest.mark.asyncio
    async def test_reject_plan_updates_pending_decision(self):
        """reject_plan injects 'reject' into pending_decision."""
        pipeline_id = str(uuid.uuid4())
        pipeline = EPACPipeline()
        pipeline._thread_states[pipeline_id] = {"pipeline_id": pipeline_id}

        import epac.pipeline as pipeline_mod
        original = pipeline_mod._LANGGRAPH_AVAILABLE
        pipeline_mod._LANGGRAPH_AVAILABLE = False
        pipeline._graph = None
        try:
            await pipeline.reject_plan(pipeline_id, notes="Needs rework")
        finally:
            pipeline_mod._LANGGRAPH_AVAILABLE = original

        stored = pipeline._thread_states[pipeline_id]
        if hasattr(stored, "pending_decision"):
            assert stored.pending_decision == "reject"
        else:
            assert stored.get("pending_decision") == "reject"


# ── Test 4: actor_critic_loop_exhausted triggers failed=True ─────────────────

class TestActorCriticLoopExhaustion:
    @pytest.mark.asyncio
    async def test_loop_exhausted_sets_failed(self):
        """When Actor-Critic loop is exhausted, pipeline sets failed=True."""
        spec = _make_spec()
        plan = _make_plan(spec.id)
        plan_approved = plan.model_copy(update={"expert_approved": True})

        # Actor returns an implementation
        impl = _make_implementation(plan.id, [t.id for t in plan.tasks])

        # Planner returns a plan (auto-approved in simple mode when it has expert_approved=None)
        planner_return = {
            "plan": plan.model_dump(),
            "planner_status": StageStatus.COMPLETED.value,
            "current_stage": "expert",
        }

        iteration_counter = {"n": 0}

        def make_actor_return(iteration: int) -> dict:
            i = _make_implementation(plan.id, [t.id for t in plan.tasks], iteration=iteration)
            return {
                "implementations": [i.model_dump()],
                "actor_status": StageStatus.COMPLETED.value,
                "actor_critic_iteration": iteration,
                "current_stage": "critic",
            }

        async def actor_side_effect(state: dict) -> dict:
            iteration_counter["n"] += 1
            return make_actor_return(iteration_counter["n"])

        def make_critic_return(state: dict) -> dict:
            s = EPACState(**state)
            impl_list = s.implementations
            last_impl = impl_list[-1] if impl_list else impl
            r = _make_review(
                last_impl.id if hasattr(last_impl, "id") else impl.id,
                plan.id,
                spec.id,
                passed=False,
                iteration=s.actor_critic_iteration,
            )
            updates = {
                "reviews": [
                    (rv.model_dump() if hasattr(rv, "model_dump") else rv)
                    for rv in s.reviews
                ] + [r.model_dump()],
                "critic_status": StageStatus.COMPLETED.value,
                "current_stage": "actor",
            }
            if s.actor_critic_loop_exhausted:
                updates["failed"] = True
                updates["failure_reason"] = (
                    f"Actor-Critic loop exhausted after {s.max_actor_critic_iterations} "
                    "iterations without Critic passing."
                )
            return updates

        pipeline = EPACPipeline(config=EPACConfig(max_actor_critic_iterations=2))
        pipeline._planner.run = AsyncMock(return_value=planner_return)
        pipeline._actor.run = AsyncMock(side_effect=actor_side_effect)
        pipeline._critic.run = AsyncMock(side_effect=make_critic_return)

        result = await pipeline._run_simple("test-loop-exhaust", spec)

        assert result.failed is True
        assert "exhausted" in result.failure_reason.lower()
        assert result.completed is False

    @pytest.mark.asyncio
    async def test_loop_exhausted_via_critic_run(self):
        """CriticRole sets failed=True when actor_critic_loop_exhausted is True."""
        spec = _make_spec()
        plan = _make_plan(spec.id)
        plan_approved = plan.model_copy(update={"expert_approved": True})
        impl = _make_implementation(plan.id, [t.id for t in plan.tasks], iteration=5)
        review_fail = _make_review(impl.id, plan.id, spec.id, passed=False)

        # Build a state where loop is already exhausted (iteration == max)
        state = EPACState(
            pipeline_id="loop-test",
            spec=spec,
            plan=plan_approved,
            implementations=[impl],
            reviews=[review_fail],
            actor_critic_iteration=5,
            max_actor_critic_iterations=5,
        )
        assert state.actor_critic_loop_exhausted is True

        # Mock the LLM call in CriticRole to return a failing review
        failing_review_data = {
            "passed": False,
            "fail_rationale": "Critical bug",
            "actor_feedback": "Fix everything",
            "findings": [
                {
                    "rule_id": "test/001",
                    "severity": "error",
                    "category": "correctness",
                    "message": "Critical bug",
                }
            ],
        }

        with patch("epac._llm.call_llm_json", new=AsyncMock(return_value=failing_review_data)):
            from epac.roles.critic import CriticRole
            critic = CriticRole()
            updates = await critic.run(state.model_dump())

        assert updates.get("failed") is True
        assert "exhausted" in updates.get("failure_reason", "").lower()
