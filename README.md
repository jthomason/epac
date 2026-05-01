# EPAC: The Expert-Planner-Actor-Critic Framework

[![PyPI version](https://badge.fury.io/py/epac.svg)](https://badge.fury.io/py/epac)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)
[![CI](https://github.com/thomason-io/epac/actions/workflows/ci.yml/badge.svg)](https://github.com/thomason-io/epac/actions)

*Structure your agentic workflows to produce better outputs through iteration and human judgment*

> **Status: Concept / Early Development.** EPAC is not production-ready. The framework is under active development and the API will change. Use it to explore the pattern, build on the ideas, or contribute — not to run production workloads.

---

## What is EPAC?

EPAC is a Python framework for running agentic AI workflows that actually produce reliable outputs.

You define what success looks like. Four roles do the work:

```
Expert → Planner → Actor ↔ Critic → Expert
```

- **Expert** (you): writes a typed spec with acceptance criteria, sets the risk level, approves the plan and the final output
- **Planner** (AI): turns the spec into a concrete, ordered task plan — no code, just structure
- **Actor** (AI): executes the plan and produces the output
- **Critic** (AI, different model): independently reviews the Actor's output against the original spec, returns specific feedback if it falls short
- The Actor-Critic loop repeats until the output passes or escalates to you

The loop is the point. A single-shot prompt gives you one attempt. EPAC gives you structured iteration where each pass is grounded in your original definition of success, reviewed by an agent with no stake in the Actor's approach.

**Why developers care:** drop-in Python library, LangGraph orchestration, LiteLLM for any model provider, Pydantic v2 typed artifacts, CLI included. One command to turn a prompt into a spec: `epac spec init`.

**Why decision makers care:** every run produces versioned artifacts at each stage, human approval gates before work starts and before output is used, and a full audit trail of what the agents did and why. You know what you got and how you got it.

```bash
pip install "epac[langgraph,litellm]"
epac spec init  # turns your prompt into a typed spec
epac run        # runs the full pipeline
```

---

## Quickstart

**Requirements:**
- Python 3.11+
- API keys for at least one LLM provider. Two providers are recommended: one for the Actor (does the work), one for the Critic (reviews the work). Using different providers catches more errors.

**Install:**

```bash
pip install "epac[langgraph,litellm]"
```

**Set your API keys** (use whichever providers you have):

```bash
# Anthropic (recommended default)
export ANTHROPIC_API_KEY=sk-ant-...

# OpenAI
export OPENAI_API_KEY=sk-...

# Google
export GOOGLE_API_KEY=...

# Any other LiteLLM-compatible provider works too
```

**Turn a prompt into a spec:**

```bash
epac spec init
# paste or describe your task — EPAC extracts structure and asks for what it can't infer
# outputs a validated YAML spec file
```

Or skip the wizard and write one directly (see [Chapter 3](#chapter-3-the-expert-role) for the full schema).

**Run the pipeline:**

```python
import asyncio
from epac import EPACPipeline, EPACConfig
from epac.artifacts import EPACSpec
from epac.artifacts.spec import AcceptanceCriterion, RiskLevel

async def main():
    spec = EPACSpec(
        created_by="you@example.com",
        title="Add rate limiting to the API",
        goal="Implement per-client rate limiting using Redis.",
        acceptance_criteria=[
            AcceptanceCriterion(
                given="a client sends 101 requests in 60 seconds",
                when="using the same API key",
                then="the 101st request returns HTTP 429",
            )
        ],
        risk_level=RiskLevel.MEDIUM,
    )
    config = EPACConfig(
        llm_model="anthropic/claude-haiku-4-5",          # Actor: fast, cheap
        critic_llm_model="anthropic/claude-sonnet-4-5",  # Critic: more thorough
    )
    pipeline = EPACPipeline(config=config)
    result = await pipeline.run(spec=spec)
    print(f"Status: {result.status}")

asyncio.run(main())
```

The pipeline will pause at Gate 1 (approve the plan) and Gate 2 (approve the output) and prompt you in the terminal.

**Optional extras:**

```bash
pip install "epac[github]"   # GitHub PR integration
pip install "epac[all]"      # everything including SAST tools (bandit, semgrep)
```

---

## Table of Contents

1. [The Agentic AI Problem](#chapter-1-the-agentic-ai-problem)
2. [Introducing EPAC](#chapter-2-introducing-epac)
3. [The Expert Role](#chapter-3-the-expert-role)
4. [The Planner Role](#chapter-4-the-planner-role)
5. [The Actor Role](#chapter-5-the-actor-role)
6. [The Critic Role](#chapter-6-the-critic-role)
7. [Orchestration and State](#chapter-7-orchestration-and-state)
8. [Governance and Compliance](#chapter-8-governance-and-compliance)
9. [Getting Started](#chapter-9-getting-started)

---

## Chapter 1: The Agentic AI Problem

Agentic AI is no longer a research curiosity. Large language models now drive autonomous pipelines that read codebases, call APIs, write and execute code, and propose changes to production systems. The productivity potential is real. So is the quality problem.

### Why Agentic Outputs Disappoint

The standard pattern for deploying an AI agent is: write a prompt, run the agent, review the output. When the output is wrong or incomplete, you adjust the prompt and try again. This loop is implicit, unstructured, and entirely dependent on how observant the person reviewing the output happens to be that day.

The problem is not that the models are bad. The problem is that a single-shot prompt gives the model no structure to work within, no explicit success criteria to optimize for, and no second perspective to catch what it missed. The result is agentic workflows that produce outputs ranging from excellent to dangerously wrong, with no reliable way to tell which you got until someone reads the output carefully.

### The Three Quality Failures

Most poor agentic outputs trace back to one of three structural problems.

**No shared definition of success.** The agent interprets the goal in good faith but not in the way the human intended. A coding agent asked to "improve test coverage" might delete the assertions that would catch regressions. A contract review agent asked to "flag risky clauses" might apply a risk threshold the reviewer would never have chosen. Without explicit acceptance criteria, the model fills the gaps with its own judgment.

**No second perspective.** The same model that produced the output is a poor reviewer of that output. It tends to rationalize its own choices rather than challenge them. Errors of omission, false assumptions, and subtle goal misalignment are invisible to the model that made them.

**No structured iteration.** When a human reviewer finds a problem, the typical response is to re-run the agent with a modified prompt. There is no record of what changed, why the original output failed, or whether the revised output actually addressed the root issue. Quality improvements are not accumulating; they are being discarded at the end of every session.

### What Actually Improves Output Quality

Three things reliably improve agentic output quality.

First, an explicit specification written by a human domain expert before the agent runs. Not a prompt, but a structured definition of what success looks like, written in terms that can be verified.

Second, independent review by a second agent on a different model, using the original specification as the benchmark, with findings fed back to the first agent for revision.

Third, a human in the loop at the points where human judgment actually matters: approving the approach before work begins, and approving the output before it is acted on.

EPAC is a framework that makes these three things structural rather than optional.

---

## Chapter 2: Introducing EPAC

EPAC stands for Expert-Planner-Actor-Critic. It is a role-based framework for structuring agentic AI pipelines so they produce better outputs: through explicit specifications, independent review, and iterative refinement with a human at the key decision points. Auditability and governance follow naturally from the same structure.

### The Core Pattern

```
Expert --> Planner --> Actor <--> Critic --> Expert
```

The pipeline begins and ends with a human. In between, three AI agents perform distinct, bounded functions: planning, implementation, and review. No AI role can perform another AI role's function. No AI role can override a human decision.

```
┌─────────────────────────────────────────────────────────┐
│                      Expert                             │
│  (Domain knowledge, acceptance criteria, final sign-off) │
└────────────────────┬─────────────────▲────────────────┘
                     │ EPACSpec         │ EPACImplementation
                     ▼                 │  (Critic-approved)
              ┌──────────────┐         │
              │   Planner    │         │
              │  (AI agent)  │         │
              └──────┬───────┘         │
                     │ EPACPlan        │
                     ▼                 │
              ┌──────────────┐    ┌────┴────────┐
              │    Actor     │◄──►│   Critic    │
              │  (AI agent)  │    │  (AI agent) │
              └──────────────┘    └─────────────┘
               EPACImplementation   EPACReview
```

### The Four Roles

**Expert** is the human domain authority. Experts own the business objective. They write the specification that launches every pipeline run and they hold the final approval authority over both the plan and the implementation.

**Planner** is an AI agent that transforms the Expert's specification into a structured, ordered task plan. The Planner reasons about design, dependencies, and risk, but it never writes code.

**Actor** is an AI agent that implements the Planner's task plan as concrete code changes. The Actor works in an isolated sandbox and cannot deploy to production without passing through the Critic and receiving Expert approval.

**Critic** is an AI agent that independently reviews the Actor's implementation against the original specification and a set of security and policy rules. The Critic produces a formal findings report but cannot modify the code it reviews.

### The Typed Artifacts

Every stage boundary in EPAC produces a typed, schema-validated artifact. These artifacts are the foundation of the audit trail.

| Artifact | Produced By | Consumed By |
|---|---|---|
| `EPACSpec` | Expert | Planner |
| `EPACPlan` | Planner | Expert (Gate 1), Actor |
| `EPACImplementation` | Actor | Critic |
| `EPACReview` | Critic | Expert (Gate 2) |

All four artifacts are Pydantic models. They are immutable once produced. Any modification creates a new versioned artifact with a new hash. This design makes tampering detectable and makes the full provenance of any deployment reconstructible from first principles.

### The Approval Gates

Two mandatory human checkpoints exist in every EPAC pipeline run.

**Gate 1** sits between the Planner and the Actor. The Expert reviews the `EPACPlan` and either approves it (allowing the Actor to begin implementation) or rejects it (returning the Planner for revision). No code is written until a human approves the plan.

**Gate 2** sits between the Critic and the final deployment. The Expert reviews the `EPACReview` findings alongside the `EPACImplementation`. Only after explicit Expert approval can the implementation advance toward production.

| Gate | Position | Reviewer | Passes To |
|---|---|---|---|
| Gate 1 | After Planner | Expert | Actor (on approval) or Planner (on rejection) |
| Gate 2 | After Critic | Expert | Deployment (on approval) or Actor (on rejection) |

### Why This Structure Works

EPAC works because it separates the concerns that prior frameworks collapse together. The Expert provides intent without performing implementation. The Planner provides structure without performing execution. The Actor performs execution without self-review. The Critic performs review without being able to rationalize away its own findings. Each role has a single function and a single set of artifacts it is authorized to produce.

This separation is the structural source of quality improvement. The Actor knows exactly what success looks like before starting, because the Expert wrote it down and the Planner decomposed it into verifiable tasks. The Critic catches what the Actor missed because it has no investment in the Actor's approach. The iterative loop between Actor and Critic means quality compounds across revisions rather than being reset on every run. The human approval gates ensure that domain judgment is applied at the moments that matter: before significant work begins, and before output is acted on.

The audit trail and governance properties are real, but they are downstream consequences of doing this correctly, not the primary goal.

---

## Chapter 3: The Expert Role

The Expert is the only human role in the EPAC pipeline, and it is the most important one. Every EPAC run begins with an Expert and ends with an Expert. The Expert is Accountable, in the RACI sense, for every activity in the pipeline.

### Who Is an Expert?

An Expert is a person with domain authority over the business objective being pursued. In a software engineering context, that might be a product manager, a technical lead, or an engineering manager. The Expert does not need to be a developer, but they need to understand the business goal well enough to write a meaningful acceptance criterion and to evaluate whether the final implementation achieves it.

The Expert role cannot be delegated to an AI. This is a design constraint, not a limitation. When an AI generates the specification that another AI implements, the accountability chain is broken. No human has exercised judgment at the point where judgment matters most: the definition of what success means.

### Writing the EPACSpec

The Expert's primary output is the `EPACSpec`. This Pydantic model captures everything the pipeline needs to operate correctly.

```python
class EPACSpec(BaseModel):
    created_by: str          # Expert identifier (email or user ID)
    title: str               # Short, descriptive title
    goal: str                # Business objective in plain language
    acceptance_criteria: list[AcceptanceCriterion]  # GIVEN/WHEN/THEN
    constraints: list[str]   # Technical or policy constraints
    risk_level: RiskLevel    # LOW, MEDIUM, HIGH, or CRITICAL
```

The acceptance criteria use a structured GIVEN/WHEN/THEN format borrowed from behavior-driven development. This format forces precision. A well-written acceptance criterion leaves no room for the Planner or Actor to substitute their own interpretation of success.

**Example:**

```
GIVEN:  a client sends 101 requests in 60 seconds
WHEN:   using the same API key
THEN:   the 101st request returns HTTP 429
```

This criterion is measurable, unambiguous, and testable. The Critic can evaluate the Actor's implementation against it directly.

### Expert Constraints on AI Behavior

The Expert also sets the `risk_level`, which controls the pipeline's autonomy settings. Higher risk levels require more human checkpoints and more restrictive Actor sandboxing. The Expert can add explicit constraints to the `EPACSpec` that bind both the Planner and the Actor: no third-party dependencies, no changes to authentication logic, no modifications outside a specified directory.

These constraints are not advisory. They are enforced by the Critic as mandatory policy rules. If the Actor violates a constraint, the Critic flags it as a blocking finding and Gate 2 will not pass without explicit Expert acknowledgment.

### The Expert at the Gates

At Gate 1, the Expert reviews the `EPACPlan`. They are looking for whether the Planner has understood the goal, whether the proposed approach is architecturally sound, and whether the plan introduces risks the `EPACSpec` did not anticipate. The Expert can approve, reject with comments, or reject and request a revised specification.

At Gate 2, the Expert reviews the `EPACReview`. They examine the Critic's findings, the test evidence, and the build artifacts. Only the Expert can make the final judgment that the implementation is acceptable for deployment. This judgment is recorded in the audit trail with a timestamp and the Expert's identifier.

The Expert is not a rubber stamp. The EPAC framework is designed to ensure that by the time a decision reaches Gate 2, the Expert has meaningful, well-organized information on which to base a real decision.

---

## Chapter 4: The Planner Role

The Planner is the first AI agent in the pipeline. It receives the `EPACSpec` from the Expert and produces the `EPACPlan`: a structured decomposition of the work required to meet the acceptance criteria.

### What the Planner Does

The Planner's job is to translate intent into structure. Given a business goal and a set of acceptance criteria, it must reason about what tasks are required, in what order they should be performed, what dependencies exist between them, what technical design decisions are implied, and what risks are worth flagging before implementation begins.

The Planner operates at the design level. It produces task descriptions, API contracts, data model sketches, and testing guidance. It does not produce code. This constraint is fundamental. A Planner that writes code is not a Planner; it is an Actor with planning responsibilities, and the separation of concerns that makes EPAC auditable collapses.

### The EPACPlan Structure

```python
class EPACPlan(BaseModel):
    spec_id: str                     # Reference to the originating EPACSpec
    tasks: list[PlanTask]            # Ordered task list with dependencies
    design_decisions: list[str]      # Architectural and design choices
    api_contracts: list[APIContract] # Interface definitions
    security_guidance: list[str]     # Security considerations for the Actor
    testing_guidance: list[str]      # Test strategy for the Actor
    risk_flags: list[str]            # Issues requiring Expert awareness
```

Each `PlanTask` has a unique identifier, a description, a list of prerequisite task IDs, and an estimated complexity. The dependency graph is explicit: the Actor cannot begin a task until all its prerequisites are marked complete.

### Design Decisions and Constraints

The Planner records its design decisions explicitly. This matters because design decisions represent the Planner's interpretation of the Expert's intent. When the Expert reviews the `EPACPlan` at Gate 1, they can see not just what the Planner proposes to do but why. If the Planner has misunderstood the goal, the design decisions section will reveal it before any code is written.

The Planner cannot change the acceptance criteria from the `EPACSpec`. It can flag a criterion as ambiguous or technically infeasible in the `risk_flags` field, but the Expert must resolve any such flag before the plan can be approved. The acceptance criteria belong to the Expert. The Planner works within them.

### Security and Testing Guidance

The Planner is expected to think ahead about security and testing even though it cannot execute either. For security, the Planner identifies which parts of the implementation touch authentication, authorization, data validation, or external APIs and flags them for heightened Critic scrutiny. For testing, the Planner specifies what kinds of tests the Actor should write: unit tests for individual functions, integration tests for API boundaries, and property-based tests for critical business logic.

This guidance is not binding in the same way that acceptance criteria are, but it is part of the formal `EPACPlan` artifact and the Expert reviews it. If the Actor ignores the Planner's testing guidance, the Critic will flag the deviation.

### What the Planner Cannot Do

The Planner has a clear set of prohibitions:

- It cannot write or modify source code.
- It cannot change acceptance criteria.
- It cannot approve its own plan.
- It cannot access production systems.
- It cannot communicate with the Actor directly. All communication is mediated through the typed artifacts and the pipeline orchestrator.

These constraints are enforced at the framework level, not by prompt engineering. The Planner's tool set simply does not include the capabilities required to violate them.

---

## Chapter 5: The Actor Role

The Actor is the implementation agent. It receives the approved `EPACPlan` and produces the `EPACImplementation`: a set of concrete file changes, test results, and build evidence that represents the system's response to the Expert's original specification.

### What the Actor Does

The Actor reads the `EPACPlan` task by task, following the dependency graph established by the Planner. For each task, it produces code changes, executes tests, and records the results. It works entirely within a sandboxed environment: isolated file system, no network access to production systems, short-lived credentials scoped to the minimum required permissions.

The Actor is the most powerful agent in the pipeline in the sense that it is the one that produces executable artifacts. This is precisely why its constraints are the most stringent.

### The EPACImplementation Structure

```python
class EPACImplementation(BaseModel):
    plan_id: str                    # Reference to the approved EPACPlan
    file_changes: list[FileChange]  # Diffs with before/after content
    test_results: list[TestResult]  # Per-test pass/fail with output
    build_evidence: BuildEvidence   # Build logs, coverage, artifact hashes
    task_log: list[TaskExecution]   # Execution trace per PlanTask
```

Every file change is recorded as a diff with the original and modified content. Every test run is recorded with its output. The build evidence includes the hash of every produced artifact. The Critic uses all of this to perform its review.

### The Sandbox Constraint

The Actor runs in an isolated sandbox. This is not a soft policy. The EPAC framework provisions the Actor's execution environment with:

- A containerized file system containing only the target codebase.
- Short-lived credentials (default TTL: 15 minutes) scoped to specific repositories and resources.
- No outbound network access except to explicitly whitelisted package registries.
- No access to production databases, secrets stores, or deployment infrastructure.

If the Actor needs a capability that would require breaking these constraints, it must flag the requirement in the `EPACImplementation` as an unresolved dependency. The Expert then decides whether to grant the capability, modify the `EPACSpec`, or abandon the run.

### Autonomy Levels

The degree of autonomy the Actor exercises is controlled by the pipeline's autonomy level, set based on the `risk_level` in the `EPACSpec`.

| Level | Name | Description |
|---|---|---|
| 1 | Autonomous | Actor operates without checkpoints. Suitable only for trivial, low-risk tasks. |
| 2 | Monitored | Actor logs all actions in real time. Humans can observe but do not block. |
| 3 | Gated | Actor pauses at defined decision points for human review. Default level. |
| 4 | Dual Control | Every significant action requires approval from two humans before proceeding. |
| 5 | Human-Only | AI assists with analysis and drafting; all actions are performed by humans. |

Most enterprise deployments use Level 3 (Gated) as the default, escalating to Level 4 for high-risk operations and Level 5 for critical systems where any autonomous action is unacceptable.

### What the Actor Cannot Do

The Actor's prohibitions are equally explicit:

- It cannot deploy to production. Deployment is a separate, post-Gate-2 action.
- It cannot bypass the Critic. The pipeline orchestrator enforces the sequence.
- It cannot modify the `EPACSpec` or `EPACPlan`.
- It cannot access systems outside its sandboxed environment.
- It cannot self-review its own implementation.

The Actor knows its job is to implement, not to judge. Judgment belongs to the Critic and, ultimately, to the Expert.

---

## Chapter 6: The Critic Role

The Critic is the pipeline's independent review agent. It receives the `EPACImplementation` and evaluates it against the original `EPACSpec`, the approved `EPACPlan`, security standards, and organizational policy. It produces the `EPACReview`: a formal findings report in SARIF 2.1.0 format.

### Why the Critic Must Be Independent

The Critic's independence is a structural requirement, not a preference. Two design decisions enforce this independence.

First, the Critic uses a different LLM provider than the Actor. If both agents run on the same model, a systematic bias or a model-level vulnerability affects both simultaneously. Using different providers (for example, an OpenAI model for the Actor and an Anthropic model for the Critic) breaks this correlation. A failure mode that causes the Actor to miss a security issue is unlikely to cause the same failure in a model from a different architecture and training lineage.

Second, the Critic cannot modify the code it reviews. It has read access to the `EPACImplementation` and nothing else. This constraint prevents a Critic that identifies a problem from quietly fixing it rather than reporting it. Every finding must surface in the `EPACReview` where the Expert can see it.

### The EPACReview Structure

```python
class EPACReview(BaseModel):
    implementation_id: str          # Reference to the EPACImplementation
    sarif_report: SARIFReport       # SARIF 2.1.0 findings
    goal_alignment_score: float     # 0.0 to 1.0
    criteria_results: list[CriterionResult]  # Pass/fail per acceptance criterion
    policy_violations: list[str]    # Constraint violations from EPACSpec
    recommendation: ReviewOutcome   # APPROVE, APPROVE_WITH_NOTES, or REJECT
```

The SARIF 2.1.0 format means findings are machine-readable and integrate directly with standard security tooling: GitHub Advanced Security, Azure Defender for DevOps, and similar platforms.

### What the Critic Evaluates

The Critic evaluates the implementation across five dimensions.

**Security (OWASP Agentic Top 10).** The Critic checks for prompt injection vulnerabilities, insecure tool use, privilege escalation paths, data exfiltration risks, and other issues specific to agentic systems. It also runs static analysis tools (Bandit, Semgrep) and records their output as part of the SARIF report.

**Correctness.** The Critic verifies that the test results in the `EPACImplementation` cover the acceptance criteria from the `EPACSpec`. If the Actor wrote tests that pass but do not actually validate the specified behavior, the Critic flags this as a blocking finding.

**Goal alignment.** The Critic evaluates whether the implementation achieves the business goal stated in the `EPACSpec`. It assigns a numeric score (0.0 to 1.0) that reflects its confidence in this assessment. A score below the configured threshold triggers a REJECT recommendation regardless of whether any individual findings are blocking.

**Policy compliance.** The Critic checks every constraint listed in the `EPACSpec`. If the Actor modified a file it was not authorized to touch, introduced a prohibited dependency, or accessed a disallowed resource, the Critic records the violation.

**Prompt injection detection.** Agentic systems that process external data are vulnerable to adversarial inputs that attempt to redirect the agent's behavior. The Critic specifically examines data paths where external content enters the system and evaluates whether the Actor's implementation handles such inputs safely.

### Review Outcomes

The Critic produces one of three recommendations: APPROVE, APPROVE_WITH_NOTES, or REJECT.

APPROVE means no blocking findings were identified. The `EPACReview` passes to the Expert at Gate 2 as a clean report.

APPROVE_WITH_NOTES means the Critic identified issues worth flagging but none that, in its judgment, prevent approval. The Expert sees the notes and decides whether to approve, require remediation, or reject.

REJECT means the Critic identified one or more blocking findings. The `EPACReview` returns to the Actor for remediation. The Actor must address the specific findings and produce a new `EPACImplementation`. The Critic then reviews again.

The Critic's recommendation is advisory. Only the Expert can make the final approval decision at Gate 2.

### Critic Tool Integrations

The Critic runs external security and quality tools before the LLM review:

```python
config = EPACConfig(
    critic_tools=[
        "bandit",                   # Python SAST (requires pip install bandit)
        "semgrep",                  # Multi-language SAST (requires semgrep CLI)
        "sarif",                    # Load pre-existing SARIF from CI
        "github_code_scanning",     # Read GitHub Code Scanning alerts (requires epac[github])
    ]
)
```

Register custom tools:

```python
from epac.integrations import register_tool
from epac.artifacts.review import ReviewFinding, FindingSeverity, FindingCategory

@register_tool("my_custom_linter")
async def run_my_linter(implementation, state) -> list[ReviewFinding]:
    return [
        ReviewFinding(
            rule_id="my_linter/001",
            severity=FindingSeverity.WARNING,
            category=FindingCategory.STYLE,
            message="Found a TODO comment in production code",
            file_path="src/main.py",
            line_start=42,
        )
    ]
```

---

## Chapter 7: Orchestration and State

The four EPAC roles do not coordinate themselves. A pipeline orchestrator manages the sequence, enforces the constraints, maintains the shared state, and records the immutable audit trail that governance requires.

### The Pipeline Orchestrator

EPAC ships with a LangGraph-based orchestrator that implements the state machine underlying the pipeline. The orchestrator is responsible for:

- Routing artifacts to the correct agent at each stage.
- Blocking transitions that would violate the pipeline sequence (for example, preventing the Actor from starting before Gate 1 is approved).
- Triggering human approval prompts at Gate 1 and Gate 2.
- Recording every state transition with a timestamp, a stage identifier, and the hash of the artifact produced.
- Handling failures: if an agent fails to produce a valid artifact within the configured timeout, the orchestrator records the failure and notifies the Expert.

### State Management

The pipeline maintains a `PipelineRun` object that tracks the complete state of a run from `EPACSpec` through final disposition.

```python
class PipelineRun(BaseModel):
    run_id: str
    spec: EPACSpec
    plan: EPACPlan | None
    implementation: EPACImplementation | None
    review: EPACReview | None
    gate_1_decision: GateDecision | None
    gate_2_decision: GateDecision | None
    status: RunStatus
    audit_trail: list[AuditEvent]
```

The `audit_trail` field is append-only. Every state transition, every agent output, every human decision is recorded as an `AuditEvent` with a cryptographic hash linking it to the previous event. This forms a chain of custody that is tamper-evident: modifying any event invalidates all subsequent hashes.

### Retry and Escalation Logic

The orchestrator implements configurable retry logic for agent failures. If the Planner fails to produce a valid `EPACPlan` after a configured number of attempts, the orchestrator escalates to the Expert with a failure summary. The Expert can choose to revise the `EPACSpec`, adjust the configuration, or abandon the run.

Rejection cycles at the gates follow a similar pattern. If the Actor fails to satisfy the Critic's findings after a configured number of revision cycles, the run is escalated to the Expert with the full history of attempts and findings. Infinite revision loops are not possible.

### Parallelism and Concurrency

For pipelines where the `EPACPlan` contains independent task groups, the orchestrator can run Actor instances in parallel against non-overlapping file sets. Each Actor instance operates in its own sandbox. The Critic reviews the merged `EPACImplementation` after all Actor instances complete.

Parallel execution requires that the Planner explicitly mark tasks as parallelizable in the `EPACPlan`. Tasks with shared dependencies are never run in parallel. The orchestrator enforces this through the dependency graph, not through prompt engineering.

### Integration Points

The orchestrator exposes webhook endpoints for integration with external systems: CI/CD pipelines, ticketing systems, code review platforms, and audit logging infrastructure. Gate approval decisions can be captured through these webhooks rather than requiring Experts to interact directly with the EPAC interface. This allows EPAC pipelines to integrate naturally into existing engineering workflows.

---

## Chapter 8: Governance and Compliance

EPAC is designed from the ground up to satisfy the governance requirements that enterprise AI deployments face from regulators, auditors, and internal risk teams.

### The RACI Model

Accountability in EPAC follows a strict RACI structure. Every activity in the pipeline has exactly one Accountable party and one or more Responsible parties.

| Activity | Responsible | Accountable | Consulted | Informed |
|---|---|---|---|---|
| Write EPACSpec | Expert | Expert | Domain SMEs | AI Owner |
| Produce EPACPlan | Planner (AI) | Expert | Expert | AI Owner |
| Gate 1 Approval | Expert | Expert | Security Team | AI Owner |
| Produce EPACImplementation | Actor (AI) | Expert | Expert | AI Owner |
| Produce EPACReview | Critic (AI) | AI Owner | Security Team | Expert |
| Gate 2 Approval | Expert | Expert | Security Team | AI Owner |
| Production Deployment | DevOps | AI Owner | Expert | Compliance |

The key principle: AI systems are Responsible for the quality of their outputs within their defined role. They are never Accountable. Accountability for every activity rests with a human: the Expert for domain decisions and the AI Owner for AI system behavior. This distinction is not semantic. It determines who answers to the regulator when something goes wrong.

### Immutable Audit Trail

Every EPAC pipeline run produces an immutable, hash-chained audit trail. This trail records:

- The full content of every typed artifact (EPACSpec, EPACPlan, EPACImplementation, EPACReview) at the time of production.
- The identity and timestamp of every human decision (Gate 1, Gate 2).
- The model identifiers and version strings of the Planner, Actor, and Critic at the time of execution.
- The configuration parameters (autonomy level, sandbox settings, tool versions) in effect during the run.

This evidence satisfies the documentation requirements of all major AI governance frameworks and is sufficient for post-incident forensic analysis.

```json
[
  {"stage": "expert",  "event": "spec_submitted",         "artifact_id": "...", "risk_level": "medium"},
  {"stage": "planner", "event": "plan_generated",          "artifact_id": "..."},
  {"stage": "expert",  "event": "plan_approved",           "artifact_id": "..."},
  {"stage": "actor",   "event": "implementation_produced", "artifact_id": "...", "iteration": 1},
  {"stage": "critic",  "event": "review_completed",        "artifact_id": "...", "passed": false},
  {"stage": "actor",   "event": "implementation_produced", "artifact_id": "...", "iteration": 2},
  {"stage": "critic",  "event": "review_completed",        "artifact_id": "...", "passed": true},
  {"stage": "expert",  "event": "implementation_approved", "artifact_id": "..."}
]
```

### Regulatory Alignment

**EU AI Act (Article 14: Human Oversight).** Article 14 requires that high-risk AI systems allow natural persons to effectively oversee and intervene in AI system outputs. EPAC's two-gate structure directly implements this requirement. Gate 1 provides oversight over the plan before implementation begins. Gate 2 provides intervention authority before any change reaches production. The immutable audit trail satisfies Article 14's documentation requirements.

**NIST AI RMF.** The NIST AI Risk Management Framework organizes AI governance into four functions: GOVERN, MAP, MEASURE, and MANAGE.

| NIST Function | EPAC Implementation |
|---|---|
| GOVERN | RACI structure, autonomy level policy, pipeline configuration |
| MAP | EPACSpec risk_level, Planner risk_flags, Critic policy_violations |
| MEASURE | EPACReview goal_alignment_score, SARIF findings, test coverage |
| MANAGE | Gate decisions, Critic rejection cycles, escalation logic |

**ISO 42001.** ISO 42001 is the AI management system standard. EPAC's typed artifacts, versioned configurations, and audit trail provide the documented evidence that ISO 42001 audits require. The framework's explicit role boundaries support ISO 42001's requirements for defined responsibilities and competencies.

### Security: OWASP Agentic Top 10

The Critic's security review is structured around the OWASP Agentic Top 10, the emerging standard for agentic AI security risks. Key categories that EPAC specifically addresses:

- **Prompt injection:** The Critic evaluates all external data entry points.
- **Insecure tool use:** The Actor's tool set is explicitly declared in the pipeline configuration and audited.
- **Excessive agency:** Autonomy levels and sandbox constraints limit what the Actor can do.
- **Privilege escalation:** Short-lived, scoped credentials prevent credential abuse.
- **Data exfiltration:** Network restrictions in the Actor's sandbox prevent unauthorized data transfer.

---

## Chapter 9: Getting Started

EPAC is available as a Python package. The core library supports LangGraph for orchestration and LiteLLM for multi-provider LLM routing.

### Installation

```bash
# Minimal install
pip install epac

# With LangGraph (recommended: full HITL support)
pip install "epac[langgraph,litellm]"

# Full install (GitHub integration + SAST tools)
pip install "epac[all]"
```

**Prerequisites:**
- Python 3.11 or later.
- API credentials for at least one LLM provider (OpenAI, Anthropic, Google, or any LiteLLM-compatible endpoint).
- For production deployments: a container runtime (Docker or Podman) for Actor sandboxing.

### Your First Pipeline

The following example implements the canonical rate limiting use case. It demonstrates the full EPAC configuration including dual-provider Critic setup.

```python
import asyncio
from epac import EPACPipeline, EPACConfig
from epac.artifacts import EPACSpec
from epac.artifacts.spec import AcceptanceCriterion, Constraint, RiskLevel

async def main():
    spec = EPACSpec(
        created_by="you@example.com",
        title="Add rate limiting to the API",
        goal="Implement per-client rate limiting using Redis.",
        acceptance_criteria=[
            AcceptanceCriterion(
                given="a client sends 101 requests in 60 seconds",
                when="using the same API key",
                then="the 101st request returns HTTP 429",
            )
        ],
        constraints=[
            Constraint(category="security", description="API keys must not appear in logs"),
            Constraint(category="performance", description="< 5ms p99 added latency"),
        ],
        risk_level=RiskLevel.MEDIUM,
    )
    config = EPACConfig(
        llm_model="openai/gpt-4o",
        critic_llm_model="anthropic/claude-3-7-sonnet-20250219",
        critic_tools=["bandit", "semgrep"],
    )
    pipeline = EPACPipeline(config=config)
    result = await pipeline.run(spec=spec)
    print(f"Completed: {result.completed}")
    print(f"Actor-Critic iterations: {result.actor_critic_iterations}")
    for change in result.final_implementation.file_changes:
        print(f"  [{change.action}] {change.path}")

asyncio.run(main())
```

### Human-in-the-Loop Workflow

For real enterprise use, pause the pipeline at approval gates so the Expert can review artifacts:

```python
from epac import EPACPipeline, EPACConfig
from epac.state import AutonomyLevel

pipeline = EPACPipeline(config=EPACConfig(autonomy_level=AutonomyLevel.DUAL_CONTROL))

# Start pipeline; it pauses at the plan approval gate
thread_id = await pipeline.start(spec=spec)

# Expert reviews the plan (show in your UI, Slack, etc.)
state = pipeline._get_state(thread_id)
plan = state.plan  # EPACPlan with tasks, design decisions, etc.

# Expert approves or rejects
await pipeline.approve_plan(thread_id, notes="Looks good, proceed")
# or: await pipeline.reject_plan(thread_id, notes="Revise the auth approach")

# Pipeline runs Actor-Critic loop, then pauses for final approval
await pipeline.approve_implementation(thread_id)

result = pipeline.get_result(thread_id)
```

### Configuration Reference

The `EPACConfig` object controls the pipeline's behavior:

| Parameter | Type | Default | Description |
|---|---|---|---|
| `llm_model` | str | Required | LiteLLM model string for Planner and Actor |
| `critic_llm_model` | str | Required | LiteLLM model string for Critic (different provider recommended) |
| `autonomy_level` | int | 3 | Pipeline autonomy level (1-5) |
| `critic_tools` | list[str] | [] | Static analysis tools for the Critic |
| `max_revision_cycles` | int | 3 | Maximum Actor-Critic revision iterations before escalation |
| `max_file_content_chars` | int | 20000 | Max characters of file content passed to Actor (0 = unlimited) |
| `max_actor_feedback_chars` | int | 10000 | Max characters of Critic feedback passed back to Actor (0 = unlimited) |
| `max_actor_feedback_findings` | int | 50 | Max findings passed back to Actor per revision cycle (0 = unlimited) |
| `max_tokens` | int | 16000 | Max tokens for LLM responses |
| `sandbox_ttl_seconds` | int | 900 | Actor credential TTL in seconds |
| `audit_backend` | str | "local" | Audit trail storage: "local", "s3", or "azure_blob" |

### GitHub Integration (optional)

```bash
pip install "epac[github]"
```

Generate a GitHub Actions workflow:

```bash
epac workflow --trigger issue_comment > .github/workflows/epac.yml
```

The Expert triggers the pipeline by commenting `/epac run` on a GitHub Issue. The Actor opens a pull request; the Critic uploads SARIF to GitHub Code Scanning; the Expert approves by merging.

### CLI Reference

```bash
# Run a pipeline from the command line
epac run --goal "Migrate SQLite to PostgreSQL" --expert-id bob@corp.com

# Run from a spec file
epac run --spec feature.json --output-file result.json

# Generate a GitHub Actions workflow
epac workflow --trigger workflow_dispatch > .github/workflows/epac.yml

# Report on a result
epac report --result-file result.json
```

### Writing Good Acceptance Criteria

The quality of the `EPACSpec` determines the quality of everything that follows. These practices consistently produce better results:

- Write exactly one GIVEN/WHEN/THEN per testable behavior. Do not bundle multiple behaviors into one criterion.
- Make the THEN clause a concrete, measurable outcome (HTTP status code, database record state, function return value) rather than a subjective description.
- Specify what should NOT happen as a separate criterion if it matters (for example, a criterion asserting that rate-limited requests are not charged).
- Set `risk_level` conservatively. Upgrading autonomy is easy; recovering from an under-constrained high-autonomy run is not.

### Next Steps

After running your first pipeline:

1. Review the `EPACPlan` your Planner produced and compare it to what you expected. The gap reveals where your `EPACSpec` was ambiguous.
2. Examine the `EPACReview` SARIF report even when the Critic recommends APPROVE. Understanding what the Critic evaluated (and what it did not flag) builds confidence in the framework's coverage.
3. Enable the audit backend for your environment. Local audit trails are useful for development; production deployments require durable, tamper-evident storage.
4. Tune the `autonomy_level` to match your team's risk tolerance. Most teams start at Level 3 and move to Level 2 for routine, low-complexity tasks after building confidence in the pipeline's outputs.

---

## Architecture

```
epac/
├── artifacts/           # Typed contracts between stages
│   ├── spec.py          # EPACSpec (Expert --> Planner)
│   ├── plan.py          # EPACPlan (Planner --> Actor)
│   ├── implementation.py  # EPACImplementation (Actor --> Critic)
│   └── review.py        # EPACReview + SARIF 2.1.0 (Critic --> Expert)
├── roles/               # Agent implementations
│   ├── expert.py        # Human-in-the-loop node with approval gates
│   ├── planner.py       # Task decomposition agent
│   ├── actor.py         # Code generation agent
│   └── critic.py        # Security + quality review agent
├── integrations/        # Tool and platform integrations
│   ├── _builtin.py      # bandit, semgrep, sarif
│   └── github/          # GitHub PR, Code Scanning, Actions workflow
├── pipeline.py          # LangGraph orchestrator
├── state.py             # Shared EPACState (persisted across HITL pauses)
└── cli.py               # Command-line interface
```

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). All contributions are welcome: new role implementations, tool integrations, documentation, and examples.

The project follows the [EPAC design pattern document](docs/pattern.md) as its specification. PRs that modify core behavior should reference the relevant pattern section.

---

## License

Apache 2.0 — see [LICENSE](LICENSE).

---

## Citation

If you use EPAC in research or write about it, please cite:

```bibtex
@software{thomason2026epac,
  author  = {Thomason, James},
  title   = {EPAC: Expert-Planner-Actor-Critic Framework for Enterprise Agentic AI},
  year    = {2026},
  url     = {https://github.com/thomason-io/epac},
  license = {Apache-2.0}
}
```

---

## Related Work

EPAC builds on and acknowledges:
- Andrew Ng's [four agentic design patterns](https://www.deeplearning.ai/the-batch/agentic-design-patterns-part-1-reflection/) (Reflection, Tool Use, Planning, Multi-Agent)
- [LangGraph](https://github.com/langchain-ai/langgraph) for stateful, HITL-capable orchestration
- [AGENTS.md](https://github.com/agentdotmd/agents.md) for Expert specification conventions
- [OWASP Top 10 for Agentic Applications](https://owasp.org/www-project-top-10-for-large-language-model-applications/) for Critic security requirements
- Hohpe & Woolf's *Enterprise Integration Patterns* as the methodological predecessor

---

> "The goal is not AI that does what we say. The goal is AI that does what we mean, within boundaries we set, for purposes we control, with evidence we can inspect. EPAC is an attempt to build that structure before the absence of it becomes a crisis."
>
> **James Thomason**

---

*EPAC Framework Documentation. All typed artifacts are Pydantic models. All pipeline runs produce immutable, hash-chained audit trails. AI systems are Responsible. Humans are Accountable.*
