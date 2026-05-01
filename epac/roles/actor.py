"""
ActorRole – implements the EPACPlan as concrete code changes.

The Actor is a code-generation agent that works task-by-task from the EPACPlan,
producing file changes, running tests, and (optionally) opening pull requests.
It operates in an isolated, permission-scoped environment and hands its output
to the Critic for review.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from epac.artifacts.implementation import EPACImplementation, FileChange, TestResult, BuildResult
from epac.roles.base import BaseRole, RoleConfig
from epac.state import AuditEntry, StageStatus

logger = logging.getLogger(__name__)


ACTOR_SYSTEM_PROMPT = """You are the Actor agent in an EPAC (Expert-Planner-Actor-Critic) pipeline.

Your purpose is to produce the highest-quality implementation you can, knowing
that a Critic will review your work and return it if it falls short. The
Actor-Critic loop exists to improve output quality through iteration, not to
catch you making mistakes. Treat Critic feedback as a collaborator's notes, not
as a compliance checklist.

Your mandate:
- Implement exactly what the Planner specified in the EPACPlan
- Work task-by-task; produce minimal, focused changes that are easy to review
- Write tests for every task: the Critic will assess coverage, and good tests
  are the clearest signal that the implementation is correct
- Document your implementation decisions in actor_notes so the Critic understands
  your reasoning, not just your output
- Explicitly list known issues or deviations from the plan

When you receive Critic feedback on a previous iteration:
- Read it carefully and address every blocking finding
- Do not just patch the flagged lines; consider whether the feedback reveals a
  deeper issue in the approach
- Explain in actor_notes what you changed and why

Constraints:
- You MUST NOT change acceptance criteria or plan-level design decisions
- You MUST NOT access systems, APIs, or credentials not listed in your tool set
- You MUST NOT deploy to production
- If you cannot implement a task safely, set it to BLOCKED and explain why
- All file changes must be minimal and purposeful

IMPORTANT — Test and build results:
- You MUST NOT fabricate or hallucinate test_results or build_results.
- No code is actually executed in this environment; you cannot know real test outcomes.
- Leave test_results and build_results as empty lists unless you have actually run the
  tests or build in a real sandbox and have real output to report.
- The Critic will assess test coverage from your code; do not invent passing numbers.

Output format: Return ONLY a flat JSON object (no wrapper keys) with this exact structure:
{
  "file_changes": [
    {
      "path": "<relative/path/to/file.py>",
      "action": "created|modified|deleted",
      "content": "<full file content as a string>",
      "diff": "<unified diff if modifying existing file, else empty string>"
    }
  ],
  "actor_notes": "<notes on implementation decisions>",
  "known_issues": ["<any issues not resolved>"],
  "test_results": [],
  "build_results": []
}
Note: test_results should only be populated if the Actor actually ran tests in a real
execution environment — never hallucinate test outcomes.
Do NOT wrap in any outer key. Return the JSON object directly.
"""


class ActorRole(BaseRole):
    """
    Actor node — generates code / artifacts from the EPACPlan.

    Supports iterative re-implementation when the Critic returns feedback.
    """

    def __init__(self, config: RoleConfig | None = None) -> None:
        if config is None:
            config = RoleConfig(
                role_name="actor",
                system_prompt=ACTOR_SYSTEM_PROMPT,
                llm_temperature=0.05,  # Low temperature for deterministic code
            )
        super().__init__(config)

    async def run(self, state: dict[str, Any]) -> dict[str, Any]:
        from epac.state import EPACState

        s = EPACState(**state)
        updates: dict[str, Any] = {"actor_status": StageStatus.IN_PROGRESS.value}

        if s.plan is None:
            raise ValueError("ActorRole.run() called with no plan in state")

        plan = s.plan
        previous_review = s.latest_review
        iteration = s.actor_critic_iteration + 1

        prompt = self._build_implementation_prompt(plan, previous_review)
        logger.info(
            "[Actor] Implementing plan '%s' (iteration %d)", plan.id, iteration
        )

        try:
            implementation = await self._call_llm_for_implementation(
                prompt, plan, iteration
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("[Actor] LLM call failed: %s", exc)
            updates["actor_status"] = StageStatus.FAILED.value
            updates["failed"] = True
            updates["failure_reason"] = f"Actor LLM error: {exc}"
            return updates

        audit_entry = AuditEntry(
            timestamp=datetime.now(timezone.utc).isoformat(),
            stage="actor",
            event="implementation_produced",
            actor_id="actor-agent",
            artifact_id=implementation.id,
            metadata={"iteration": iteration},
        )
        existing_impls = [
            i.model_dump() if hasattr(i, "model_dump") else i
            for i in s.implementations
        ]
        existing_log = [
            e.model_dump() if hasattr(e, "model_dump") else e for e in s.audit_log
        ]

        updates["implementations"] = existing_impls + [implementation.model_dump()]
        updates["actor_status"] = StageStatus.COMPLETED.value
        updates["actor_critic_iteration"] = iteration
        updates["current_stage"] = "critic"
        updates["audit_log"] = existing_log + [audit_entry.model_dump()]
        return updates

    # ── Private helpers ──────────────────────────────────────────────────────

    def _build_implementation_prompt(
        self, plan: Any, previous_review: Any | None
    ) -> str:
        lines = [
            "## EPACPlan to Implement",
            f"Summary: {plan.summary}",
            "",
            "### Tasks",
        ]
        for task in plan.tasks:
            lines += [
                f"#### Task: {task.title}",
                task.description,
                f"Test strategy: {task.test_strategy}",
            ]
            if task.artifacts:
                lines.append("Files to touch:")
                for a in task.artifacts:
                    lines.append(f"  - [{a.action}] {a.path}: {a.description}")
            lines.append("")

        if plan.security_guidance:
            lines += ["### Security Guidance", plan.security_guidance, ""]
        if plan.testing_guidance:
            lines += ["### Testing Guidance", plan.testing_guidance, ""]

        if previous_review is not None and not previous_review.passed:
            feedback = previous_review.actor_feedback
            fb_cap = self.config.max_actor_feedback_chars
            if fb_cap > 0 and len(feedback) > fb_cap:
                feedback = feedback[:fb_cap] + f"\n... [feedback truncated at {fb_cap} chars]"
            lines += [
                "### Critic Feedback (Previous Iteration — Address All Points)",
                feedback,
                "",
                "Blocking findings to fix:",
            ]
            findings_cap = self.config.max_actor_feedback_findings
            findings_to_show = (
                previous_review.blocking_findings[:findings_cap]
                if findings_cap > 0
                else previous_review.blocking_findings
            )
            for f in findings_to_show:
                lines.append(
                    f"- [{f.category.value.upper()}] {f.file_path}:{f.line_start or '?'} — {f.message}"
                )
            lines.append("")

        lines += [
            "---",
            "Return your implementation as EPACImplementation JSON.",
        ]
        return "\n".join(lines)

    async def _call_llm_for_implementation(
        self, prompt: str, plan: Any, iteration: int
    ) -> EPACImplementation:
        # TODO: sandbox execution — currently the Actor only generates code; no tests
        # or builds are actually run. A future version should execute code in an
        # isolated sandbox and populate test_results / build_results from real output.
        from epac._llm import call_llm_json

        raw = await call_llm_json(
            model=self.config.llm_model,
            system_prompt=self.config.system_prompt,
            user_prompt=prompt,
            temperature=self.config.llm_temperature,
            max_tokens=self.config.max_tokens,
        )

        try:
            impl_data = raw if isinstance(raw, dict) else __import__("json").loads(raw)

            # Unwrap common LLM envelope patterns
            if "file_changes" not in impl_data and "files" not in impl_data and "changes" not in impl_data:
                for key in ("implementation", "epac_implementation", "result", "output", "data"):
                    if key in impl_data and isinstance(impl_data[key], dict):
                        impl_data = impl_data[key]
                        break
                else:
                    for v in impl_data.values():
                        if isinstance(v, dict) and any(k in v for k in ("file_changes", "files", "changes")):
                            impl_data = v
                            break

            impl_data.setdefault("id", str(uuid.uuid4()))
            impl_data.setdefault("plan_id", plan.id)
            impl_data.setdefault("task_ids", [t.id for t in plan.tasks])
            impl_data.setdefault("created_at", datetime.now(timezone.utc).isoformat())
            impl_data.setdefault("iteration", iteration)

            # Normalize file_changes: handle alternate field names and per-change fields
            raw_changes = (
                impl_data.pop("file_changes", None)
                or impl_data.pop("files", None)
                or impl_data.pop("changes", None)
                or impl_data.pop("modified_files", None)
                or []
            )
            normalized_changes = []
            for fc in raw_changes:
                if not isinstance(fc, dict):
                    continue
                # Normalize action/change_type field
                if "action" not in fc:
                    fc["action"] = (
                        fc.pop("change_type", None)
                        or fc.pop("operation", None)
                        or fc.pop("type", "modified")
                    )
                # Normalize path field
                if "path" not in fc:
                    fc["path"] = fc.pop("file_path", fc.pop("filename", "unknown"))
                fc.setdefault("content", fc.pop("new_content", fc.pop("code", "")))
                fc.setdefault("diff", "")
                normalized_changes.append(fc)
            impl_data["file_changes"] = normalized_changes

            return EPACImplementation(**impl_data)
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"Actor returned invalid implementation JSON: {exc}") from exc
