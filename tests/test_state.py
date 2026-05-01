"""Tests for EPACState."""

import pytest
from epac.state import EPACState, StageStatus, AutonomyLevel


class TestEPACState:
    def test_initial_state(self):
        state = EPACState(pipeline_id="test-pipeline")
        assert state.expert_status == StageStatus.PENDING
        assert state.completed is False
        assert state.failed is False
        assert state.actor_critic_iteration == 0

    def test_loop_exhaustion(self):
        state = EPACState(
            pipeline_id="test",
            actor_critic_iteration=5,
            max_actor_critic_iterations=5,
        )
        assert state.actor_critic_loop_exhausted is True

    def test_no_implementations(self):
        state = EPACState(pipeline_id="test")
        assert state.latest_implementation is None
        assert state.latest_review is None
        assert state.critic_passed is False

    def test_autonomy_levels(self):
        assert AutonomyLevel.AUTONOMOUS == 1
        assert AutonomyLevel.HUMAN_ONLY == 5
        assert AutonomyLevel.GATED == 3
