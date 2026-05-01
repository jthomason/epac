"""
PlannerRole – expands the Expert's EPACSpec into a detailed EPACPlan.

The Planner is an LLM agent that performs task decomposition, dependency
analysis, and architecture documentation.  It cannot write code or call
repository APIs — its only output is the EPACPlan artifact.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from epac.artifacts.plan import EPACPlan, PlanTask, PlanArtifact, DesignDecision
from epac.roles.base import BaseRole, RoleConfig
from epac.state import AuditEntry, StageStatus

logger = logging.getLogger(__name__)


PLANNER_SYSTEM_PROMPT = """You are the Planner agent in an EPAC (Expert-Planner-Actor-Critic) pipeline.

Your purpose is to set the Actor up for high-quality work. A good plan is the
single biggest lever on output quality: it eliminates ambiguity before the Actor
starts, surfaces design decisions that the Expert should weigh in on, and gives
the Critic a clear baseline to review against.

Your mandate:
- Read the Expert's EPACSpec carefully and identify any ambiguities before planning
- Decompose the goal into concrete, independently implementable tasks
- For each task: write a description detailed enough that the Actor needs no additional context
- Make design decisions explicit with rationale so the Expert can approve them at Gate 1
- Provide security, performance, and testing guidance that will make the Actor's first attempt better
- Flag risks and open questions rather than resolving them silently

Constraints:
- You MUST NOT write any implementation code
- You MUST NOT change the Expert's acceptance criteria or constraints
- Each task must be independently reviewable by the Critic
- Identify dependencies between tasks clearly
- Ambiguity in the plan becomes defects in the implementation — be specific

Output format: Return ONLY a flat JSON object (no wrapper keys) with this exact structure:
{
  "summary": "<plain-language description of the implementation approach>",
  "tasks": [
    {
      "title": "<short imperative title>",
      "description": "<detailed description>",
      "files_to_change": ["<relative/path/to/file.py>"],
      "test_strategy": "<how to verify this task is complete>"
    }
  ],
  "security_guidance": "<security notes for Actor>",
  "testing_guidance": "<testing notes for Actor>"
}
Do NOT wrap in any outer key. Return the JSON object directly.
"""


class PlannerRole(BaseRole):
    """
    Planner node — converts EPACSpec → EPACPlan via LLM reasoning.

    Supports re-planning with Expert feedback when a plan is rejected.
    """

    def __init__(self, config: RoleConfig | None = None) -> None:
        if config is None:
            config = RoleConfig(
                role_name="planner",
                system_prompt=PLANNER_SYSTEM_PROMPT,
                llm_temperature=0.1,
            )
        super().__init__(config)

    async def run(self, state: dict[str, Any]) -> dict[str, Any]:
        from epac.state import EPACState

        s = EPACState(**state)
        updates: dict[str, Any] = {"planner_status": StageStatus.IN_PROGRESS.value}

        if s.spec is None:
            raise ValueError("PlannerRole.run() called with no spec in state")

        spec = s.spec
        expert_feedback = ""
        if s.plan is not None and s.plan.expert_approved is False:
            expert_feedback = s.plan.expert_feedback

        prompt = self._build_planning_prompt(spec, expert_feedback)
        logger.info("[Planner] Generating plan for spec '%s'", spec.title)

        try:
            plan = await self._call_llm_for_plan(prompt, spec)
        except Exception as exc:  # noqa: BLE001
            logger.error("[Planner] LLM call failed: %s", exc)
            updates["planner_status"] = StageStatus.FAILED.value
            updates["failed"] = True
            updates["failure_reason"] = f"Planner LLM error: {exc}"
            return updates

        audit_entry = AuditEntry(
            timestamp=datetime.now(timezone.utc).isoformat(),
            stage="planner",
            event="plan_generated",
            actor_id="planner-agent",
            artifact_id=plan.id,
        )
        existing_log = [e.model_dump() if hasattr(e, "model_dump") else e for e in s.audit_log]

        updates["plan"] = plan.model_dump()
        updates["planner_status"] = StageStatus.COMPLETED.value
        # If spec requires Expert plan approval, route back to expert gate
        updates["current_stage"] = "expert" if spec.requires_expert_plan_approval else "actor"
        updates["audit_log"] = existing_log + [audit_entry.model_dump()]
        return updates

    # ── Private helpers ──────────────────────────────────────────────────────

    def _build_planning_prompt(self, spec: Any, expert_feedback: str) -> str:
        lines = [
            "## Expert Specification",
            f"Title: {spec.title}",
            f"Goal: {spec.goal}",
        ]
        if spec.background:
            lines += ["", "### Background", spec.background]
        if spec.in_scope:
            lines += ["", "### In Scope", *[f"- {i}" for i in spec.in_scope]]
        if spec.out_of_scope:
            lines += ["", "### Out of Scope", *[f"- {i}" for i in spec.out_of_scope]]
        if spec.acceptance_criteria:
            lines += ["", "### Acceptance Criteria"]
            for ac in spec.acceptance_criteria:
                lines.append(f"- GIVEN {ac.given} WHEN {ac.when} THEN {ac.then}")
        if spec.constraints:
            lines += ["", "### Constraints"]
            for c in spec.constraints:
                mandatory = "MANDATORY" if c.mandatory else "OPTIONAL"
                lines.append(f"- [{c.category.upper()}] [{mandatory}] {c.description}")
        if expert_feedback:
            lines += [
                "",
                "### Expert Feedback on Previous Plan (Address These Points)",
                expert_feedback,
            ]
        lines += [
            "",
            "---",
            "Generate a detailed EPACPlan JSON. Be explicit about task dependencies,",
            "file paths, and design decisions. Do not write implementation code.",
        ]
        return "\n".join(lines)

    async def _call_llm_for_plan(self, prompt: str, spec: Any) -> EPACPlan:
        """
        Call the configured LLM to generate a plan.

        In production this uses LangChain / LiteLLM.  The implementation here
        provides a clear hook for users to swap in their preferred LLM client.
        """
        from epac._llm import call_llm_json

        raw = await call_llm_json(
            model=self.config.llm_model,
            system_prompt=self.config.system_prompt,
            user_prompt=prompt,
            temperature=self.config.llm_temperature,
            max_tokens=self.config.max_tokens,
        )

        # Hydrate into EPACPlan; fall back to a minimal scaffold if parsing fails
        try:
            plan_data = raw if isinstance(raw, dict) else json.loads(raw)

            # Unwrap common LLM envelope patterns: find the dict that looks like a plan
            # (has 'summary' and/or 'tasks' keys, possibly nested under an envelope key)
            if "summary" not in plan_data and "tasks" not in plan_data:
                # Try common envelope keys
                for key in ("plan", "epac_plan", "data", "result", "output"):
                    if key in plan_data and isinstance(plan_data[key], dict):
                        plan_data = plan_data[key]
                        break
                else:
                    # Try any dict value that has 'tasks' or 'summary'
                    for v in plan_data.values():
                        if isinstance(v, dict) and ("tasks" in v or "summary" in v):
                            plan_data = v
                            break

            plan_data.setdefault("id", str(uuid.uuid4()))
            plan_data.setdefault("spec_id", spec.id)
            plan_data.setdefault("spec_version", spec.version)
            plan_data.setdefault("created_at", datetime.now(timezone.utc).isoformat())
            plan_data.setdefault("summary", "Generated plan")
            plan_data.setdefault("tasks", [])

            # Normalize tasks: convert LLM field names to PlanTask schema field names
            normalized_tasks = []
            for t in plan_data.get("tasks", []):
                if not isinstance(t, dict):
                    continue
                t.setdefault("id", str(uuid.uuid4()))
                t.setdefault("title", t.pop("name", "Task"))
                t.setdefault("description", "")
                # Convert files_to_change / affected_files / files into artifacts
                if "artifacts" not in t:
                    files = (
                        t.pop("files_to_change", None)
                        or t.pop("affected_files", None)
                        or t.pop("files", None)
                        or []
                    )
                    t["artifacts"] = [
                        {"path": f, "action": "modify", "description": ""}
                        for f in (files if isinstance(files, list) else [])
                    ]
                normalized_tasks.append(t)
            plan_data["tasks"] = normalized_tasks

            return EPACPlan(**plan_data)
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"Planner returned invalid plan JSON: {exc}") from exc
