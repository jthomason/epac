"""
EPAC Quickstart Example – Software Development Pipeline
=======================================================

This example shows how to run a complete EPAC pipeline for a software
development task using the Python SDK.

Prerequisites:
  pip install "epac[litellm,langgraph]"
  export OPENAI_API_KEY=sk-...
  export ANTHROPIC_API_KEY=sk-ant-...

Run:
  python examples/software_dev/run_pipeline.py
"""

import asyncio
import json
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

from epac import EPACPipeline, EPACConfig
from epac.artifacts import EPACSpec
from epac.artifacts.spec import AcceptanceCriterion, Constraint, RiskLevel
from epac.state import AutonomyLevel


async def main() -> None:
    # ── 1. Expert writes the specification ────────────────────────────────────
    spec = EPACSpec(
        created_by="alice@example.com",
        title="Add rate limiting to the public API",
        goal=(
            "Implement per-client rate limiting on all public API endpoints "
            "to prevent abuse and protect downstream services."
        ),
        background=(
            "We use FastAPI on Python 3.12. Redis is available as a shared cache. "
            "The API currently has no rate limiting. We're seeing spikes of 10k req/s "
            "from badly behaved clients. Target: 100 req/min per API key."
        ),
        in_scope=[
            "Middleware-level rate limiting using Redis + sliding window algorithm",
            "429 Too Many Requests response with Retry-After header",
            "Per-API-key limits configurable via environment variables",
            "Unit tests for rate limit logic",
        ],
        out_of_scope=[
            "UI for managing rate limit configs",
            "Per-endpoint customization (v2 scope)",
            "DDoS protection (infrastructure team owns this)",
        ],
        acceptance_criteria=[
            AcceptanceCriterion(
                given="a client sends 101 requests within 60 seconds",
                when="they use the same API key",
                then="the 101st request returns HTTP 429 with a Retry-After header",
            ),
            AcceptanceCriterion(
                given="a client is rate limited",
                when="they wait for the window to reset",
                then="subsequent requests are processed normally",
            ),
            AcceptanceCriterion(
                given="Redis is unavailable",
                when="a request arrives",
                then="the middleware fails open (allows the request) and logs a warning",
            ),
        ],
        constraints=[
            Constraint(
                category="performance",
                description="Rate limit check must add < 5ms p99 latency",
                mandatory=True,
            ),
            Constraint(
                category="security",
                description="API keys must never be logged in plaintext",
                mandatory=True,
            ),
            Constraint(
                category="style",
                description="Follow existing project code style (Black, isort)",
                mandatory=False,
            ),
        ],
        risk_level=RiskLevel.MEDIUM,
        requires_expert_plan_approval=True,
        requires_expert_final_approval=True,
        tags=["api", "rate-limiting", "security"],
    )

    # ── 2. Configure the pipeline ─────────────────────────────────────────────
    config = EPACConfig(
        llm_model="openai/gpt-4o",
        critic_llm_model="anthropic/claude-3-7-sonnet-20250219",
        autonomy_level=AutonomyLevel.GATED,
        max_actor_critic_iterations=3,
        critic_tools=["bandit"],   # Add "semgrep", "github_code_scanning" for more coverage
    )
    pipeline = EPACPipeline(config=config)

    # ── 3. Run (non-interactive mode — auto-approves all gates) ───────────────
    print("\n" + "=" * 60)
    print("EPAC Pipeline Starting")
    print("Expert:  alice@example.com")
    print(f"Goal:    {spec.title}")
    print("=" * 60 + "\n")

    result = await pipeline.run(spec=spec, expert_id=spec.created_by)

    # ── 4. Inspect results ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    if result.completed:
        print("✓ Pipeline COMPLETED")
    else:
        print(f"✗ Pipeline FAILED: {result.failure_reason}")

    print(f"Actor-Critic iterations: {result.actor_critic_iterations}")
    print(f"Audit log entries:       {len(result.audit_log)}")

    if result.final_implementation:
        impl = result.final_implementation
        print(f"\nImplementation #{impl.iteration}:")
        print(f"  Files changed: {len(impl.file_changes)}")
        for change in impl.file_changes:
            print(f"    [{change.action.upper()}] {change.path}")
        if impl.pr_url:
            print(f"  PR URL: {impl.pr_url}")

    if result.final_review:
        review = result.final_review
        print(f"\nCritic Review:")
        print(f"  Passed:    {review.passed}")
        print(f"  Findings:  {len(review.findings)} total")
        print(f"  Blocking:  {len(review.blocking_findings)} errors")

    print("\nAudit trail:")
    for entry in result.audit_log:
        print(f"  [{entry.get('stage', '?'):8s}] {entry.get('event', '?')}")

    # ── 5. Save SARIF output for CI ───────────────────────────────────────────
    if result.final_review and result.final_review.sarif:
        sarif_path = "epac-critic.sarif"
        import json as _json
        with open(sarif_path, "w") as f:
            _json.dump(result.final_review.sarif.to_sarif_dict(), f, indent=2)
        print(f"\nSARIF output saved to {sarif_path}")

    print("=" * 60 + "\n")


# ── HITL example (interactive) ────────────────────────────────────────────────

async def hitl_example() -> None:
    """
    Demonstrates the human-in-the-loop approval workflow.

    In a real application the pipeline is paused and the Expert is notified
    (e.g. via Slack, email, or a web UI).  They then call approve_plan() or
    approve_implementation() after reviewing the artifact.
    """
    spec = EPACSpec(
        created_by="bob@example.com",
        title="Migrate database from SQLite to PostgreSQL",
        goal="Replace the SQLite backend with PostgreSQL for the production deployment.",
        risk_level=RiskLevel.HIGH,
    )

    config = EPACConfig(
        llm_model="openai/gpt-4o",
        autonomy_level=AutonomyLevel.DUAL_CONTROL,
    )
    pipeline = EPACPipeline(config=config)

    # Start pipeline — it runs until the first approval gate, then pauses
    thread_id = await pipeline.start(spec=spec)
    print(f"Pipeline started: thread_id={thread_id}")
    print("Pipeline paused — Expert reviewing plan...")

    # Expert reviews the plan (in practice: show plan in UI, wait for human input)
    # state = pipeline._get_state(thread_id)
    # print(state.plan.model_dump_json(indent=2))

    # Expert approves
    await pipeline.approve_plan(thread_id, notes="Plan looks good. Proceed.")
    print("Plan approved. Actor-Critic loop running...")

    # Pipeline pauses again at final approval gate
    await pipeline.approve_implementation(thread_id, notes="LGTM. Merge when ready.")
    print("Implementation approved.")

    result = pipeline.get_result(thread_id)
    print(f"Pipeline completed: {result.completed}")


if __name__ == "__main__":
    asyncio.run(main())
