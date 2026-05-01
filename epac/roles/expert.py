"""
ExpertRole – the human-in-the-loop first-class citizen.

The Expert is the initiating and approving node in the EPAC loop.  Because
the Expert is human (or human-assisted), this role does not call an LLM by
default — instead it surfaces artifacts for human review and records decisions.

In LangGraph terms, Expert nodes use `interrupt_before` / `interrupt_after`
so the graph pauses and the human can inject a Command (approve / reject / edit).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from epac.artifacts.spec import EPACSpec
from epac.roles.base import BaseRole, RoleConfig
from epac.state import AuditEntry, StageStatus

logger = logging.getLogger(__name__)


EXPERT_SYSTEM_PROMPT = """You are the Expert assistant in an EPAC pipeline.

Your responsibilities:
- Help the human Expert write high-quality EPACSpec artifacts
- Clarify goals, acceptance criteria, and constraints
- Summarize Planner output for Expert review
- Surface Critic-approved implementations for Expert sign-off
- Record Expert decisions and their rationale

You must NEVER:
- Change the Expert's stated goals without explicit approval
- Approve artifacts on the Expert's behalf
- Bypass an approval gate
"""


class ExpertRole(BaseRole):
    """
    Expert node — pauses for human review at defined approval gates.

    This role handles two distinct interactions:

    1. **Spec submission**: The Expert provides a goal; the role structures it
       into an EPACSpec and logs the submission.

    2. **Approval gates**: After plan generation or Critic pass, the pipeline
       pauses here.  The human injects an approval decision via LangGraph Command.
    """

    def __init__(self, config: RoleConfig | None = None) -> None:
        if config is None:
            config = RoleConfig(
                role_name="expert",
                system_prompt=EXPERT_SYSTEM_PROMPT,
            )
        super().__init__(config)

    async def run(self, state: dict[str, Any]) -> dict[str, Any]:
        """
        Expert node logic.

        On first call (no spec): surfaces a prompt for goal input.
        On approval gate call: reads pending_decision and routes accordingly.
        """
        from epac.state import EPACState

        s = EPACState(**state)
        updates: dict[str, Any] = {}

        # ── Gate 1: spec submission ──────────────────────────────────────────
        if s.spec is None:
            # In a real pipeline the LangGraph interrupt fires here;
            # the human provides the spec via Command or ExpertRole.submit_spec().
            logger.info("[Expert] Awaiting spec submission — pipeline will interrupt.")
            updates["expert_status"] = StageStatus.AWAITING_APPROVAL.value
            updates["current_stage"] = "expert"
            return updates

        # ── Gate 2: plan approval ────────────────────────────────────────────
        if s.plan is not None and s.plan.expert_approved is None:
            logger.info("[Expert] Plan ready for approval — pipeline will interrupt.")
            updates["expert_status"] = StageStatus.AWAITING_APPROVAL.value
            updates["current_stage"] = "expert_plan_approval"

            decision = s.pending_decision
            if decision == "approve":
                updated_plan = s.plan.model_copy(
                    update={
                        "expert_approved": True,
                        "expert_feedback": s.pending_decision_notes,
                    }
                )
                updates["plan"] = updated_plan.model_dump()
                updates["expert_status"] = StageStatus.APPROVED.value
                updates["current_stage"] = "actor"
                updates["pending_decision"] = None
                self._log_audit(updates, s, "plan_approved")

            elif decision == "reject":
                updated_plan = s.plan.model_copy(
                    update={
                        "expert_approved": False,
                        "expert_feedback": s.pending_decision_notes,
                    }
                )
                updates["plan"] = updated_plan.model_dump()
                updates["expert_status"] = StageStatus.REJECTED.value
                updates["current_stage"] = "planner"  # Re-plan with feedback
                updates["pending_decision"] = None
                self._log_audit(updates, s, "plan_rejected")

            return updates

        # ── Gate 3: final implementation approval ────────────────────────────
        if s.critic_passed:
            logger.info("[Expert] Critic-approved implementation ready — pipeline will interrupt.")
            updates["expert_status"] = StageStatus.AWAITING_APPROVAL.value
            updates["current_stage"] = "expert_final_approval"

            decision = s.pending_decision
            if decision == "approve":
                updates["expert_status"] = StageStatus.APPROVED.value
                updates["completed"] = True
                updates["pending_decision"] = None
                self._log_audit(updates, s, "implementation_approved")

            elif decision == "reject":
                # Expert can send back to Planner for a new plan
                updates["expert_status"] = StageStatus.REJECTED.value
                updates["current_stage"] = "planner"
                updates["pending_decision"] = None
                self._log_audit(updates, s, "implementation_rejected")

            return updates

        return updates

    def submit_spec(self, state: dict[str, Any], spec: EPACSpec) -> dict[str, Any]:
        """
        Helper called by the pipeline to inject an Expert-provided EPACSpec.

        In LangGraph usage this is invoked via graph.update_state() or a Command.
        """
        audit_entry = AuditEntry(
            timestamp=datetime.now(timezone.utc).isoformat(),
            stage="expert",
            event="spec_submitted",
            actor_id=spec.created_by,
            artifact_id=spec.id,
            risk_level=spec.risk_level.value,
        )
        existing_log = state.get("audit_log", [])
        return {
            "spec": spec.model_dump(),
            "expert_status": StageStatus.COMPLETED.value,
            "current_stage": "planner",
            "audit_log": existing_log + [audit_entry.model_dump()],
        }

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _log_audit(
        self,
        updates: dict[str, Any],
        state: Any,
        event: str,
    ) -> None:
        entry = AuditEntry(
            timestamp=datetime.now(timezone.utc).isoformat(),
            stage="expert",
            event=event,
            actor_id=state.spec.created_by if state.spec else "unknown",
            artifact_id=state.plan.id if state.plan else None,
        )
        existing = list(state.audit_log)
        updates["audit_log"] = [e.model_dump() for e in existing] + [entry.model_dump()]
