# EPAC Pattern Specification

**Name:** EPAC (Expert-Planner-Actor-Critic)  
**Also known as:** Expert-supervised Planner-Actor-Critic workflow  
**Version:** 1.0  
**Author:** James Thomason  
**Date:** March 2026  
**Classification:** Enterprise macro-pattern (composition of recognized agentic primitives)

---

## Context

- Operating in an **enterprise environment** where AI agents generate code or other high-impact artifacts (configuration, infrastructure, workflows) that must meet correctness, security, and compliance requirements
- Work follows an SDLC-like lifecycle: requirements → design → implementation → review, with existing human roles mapped to these phases
- Organizations seek to adopt agentic AI to accelerate this lifecycle without losing human ownership of intent and accountability

---

## Problem

How do you structure a multi-agent AI workflow for code (or artifact) generation such that:

- Domain **intent** and non-functional requirements are captured reliably from experts
- Decomposition and **planning** are explicit and inspectable, not hidden in a single prompt
- Code generation is fast but **bounded** by clear specifications and tool constraints
- Outputs are rigorously **reviewed** for bugs, security, and policy compliance before production deployment
- The entire process remains **auditable**, with well-defined human approval points

---

## Forces

| Force | Tension |
|-------|---------|
| Safety vs. speed | Rapid iteration is needed, but failures are expensive |
| Explainability vs. autonomy | Stakeholders demand traceability; agents need autonomy to be useful |
| Role clarity vs. complexity | Specialized agents improve quality but add orchestration overhead |
| Human oversight vs. fatigue | Experts must be in the loop, but not reviewing every token |
| Standardization vs. flexibility | A repeatable pattern must still adapt to different domains |

---

## Solution

Instantiate four roles — three agentic, one human (or human-supervised) — in a looped workflow:

### Role 1: Expert (human or human-assisted)

- **Inputs:** Business context, domain knowledge, risk tolerance
- **Outputs:** EPACSpec (title, goal, acceptance criteria, constraints, risk level)
- **Responsibilities:** Define intent, set acceptance criteria, approve plans, sign off on implementations
- **Boundaries:** Cannot generate implementation code; cannot bypass Critic reviews
- **Governance:** Only the Expert can approve deployment to protected environments

### Role 2: Planner (AI agent)

- **Inputs:** EPACSpec from Expert
- **Outputs:** EPACPlan (tasks, dependencies, design decisions, API contracts, guidance)
- **Responsibilities:** Task decomposition, dependency ordering, design decision documentation
- **Boundaries:** Cannot write implementation code; cannot commit to repositories; cannot change Expert acceptance criteria
- **LLM temperature:** Low (0.1) — deterministic planning favored

### Role 3: Actor (AI agent)

- **Inputs:** EPACPlan, Critic feedback (on re-iteration)
- **Outputs:** EPACImplementation (file changes, test results, build results)
- **Responsibilities:** Code generation, test writing, build verification
- **Boundaries:** Cannot modify EPACPlan; cannot bypass Critic for high-risk tasks; cannot deploy to production; executes in sandboxed/isolated environment
- **LLM temperature:** Very low (0.05) — deterministic code generation favored

### Role 4: Critic (AI agent)

- **Inputs:** EPACImplementation, EPACSpec (for goal alignment), tool scan results
- **Outputs:** EPACReview (pass/fail, findings, SARIF 2.1.0, actor feedback)
- **Responsibilities:** Security review (OWASP Agentic Top 10), goal alignment verification, correctness checking, policy compliance, prompt injection detection
- **Boundaries:** Cannot change code directly; cannot modify requirements; cannot approve with unaddressed ERROR findings
- **LLM temperature:** Low (0.1); should use a different model provider than Actor to reduce correlated failures

### Loop Closure

```
Expert ──spec──► Planner ──plan──► [Expert approval gate]
                                            │
                                            ▼
                    Actor ◄──feedback── Critic
                      │                    ▲
                      └──implementation────┘
                              (loop until Critic passes)
                                            │
                                            ▼
                              [Expert final approval gate]
                                            │
                                            ▼
                                       Deploy / Merge
```

---

## Artifact Contracts

| Artifact | Producer | Consumer | Schema |
|----------|----------|----------|--------|
| EPACSpec | Expert | Planner | `epac.artifacts.EPACSpec` |
| EPACPlan | Planner | Actor (+ Expert at approval gate) | `epac.artifacts.EPACPlan` |
| EPACImplementation | Actor | Critic | `epac.artifacts.EPACImplementation` |
| EPACReview | Critic | Actor (on fail) / Expert (on pass) | `epac.artifacts.EPACReview` |

All artifacts are versioned, immutable once produced, and logged in the audit trail.

---

## Autonomy Levels

EPAC defines five autonomy levels that determine where approval gates fire:

| Level | Name | When to use | Expert gates |
|-------|------|-------------|--------------|
| 1 | Autonomous | Read-only operations | None |
| 2 | Monitored | Low-impact changes | Final output (post-hoc) |
| 3 | Gated (default) | Medium-impact changes | Plan approval + final output |
| 4 | Dual Control | High-impact / regulated | Plan approval (mandatory) + final output (mandatory) |
| 5 | Human-Only | Critical / irreversible | AI recommends only; Expert executes |

Risk level in EPACSpec drives autonomy level selection unless explicitly overridden.

---

## RACI Matrix

| Activity | Expert | Planner | Actor | Critic | AI Owner |
|----------|--------|---------|-------|--------|----------|
| Define requirements | **R, A** | I | I | I | C |
| Create execution plan | C | **R** | I | I | **A** |
| Approve plan | **R, A** | C | I | I | I |
| Execute implementation | I | C | **R** | I | **A** |
| Quality/security review | C | I | I | **R** | **A** |
| Accept deliverable | **R, A** | I | I | C | I |

**Key principle:** AI systems can be Responsible (R) but never Accountable (A). Accountability always remains human.

---

## Security Requirements

Each EPAC component must satisfy:

### Actor
- Executes in a MicroVM or gVisor-isolated environment
- Deny-all network egress by default; allow only explicitly listed endpoints
- Short-lived, task-scoped credentials injected via credential broker
- No persistent access to production systems

### Critic
- Runs in a separate isolation boundary from the Actor
- Should use a different LLM provider than the Actor (reduces correlated failures)
- System prompt is immutable; Critic must ignore instructions embedded in Actor code/comments
- Must detect and flag prompt injection in any LLM-facing strings produced by Actor

### All components
- Unique verified identity per component
- Delegated short-lived tokens for inter-component communication
- Cryptographic provenance at every artifact handoff
- Audit log entries at every stage boundary

---

## Audit Trail Schema

Each audit entry captures:

```json
{
  "timestamp": "2026-04-01T12:34:56Z",
  "stage": "critic",
  "event": "review_completed",
  "actor_id": "critic-agent",
  "artifact_id": "review-uuid",
  "risk_level": "medium",
  "metadata": {
    "passed": false,
    "blocking_findings": 2,
    "iteration": 1
  }
}
```

Audit log storage tiers:
- Hot (0–30 days): searchable, for active debugging
- Warm (1–12 months): compliance queries
- Cold (1–7+ years): archival / regulatory

---

## Regulatory Mapping

| Regulation | EPAC Mapping |
|------------|-------------|
| EU AI Act Art. 14 | Expert approval gates operationalize "meaningful human oversight" |
| NIST AI RMF GOVERN | Expert role defines policies and risk tolerance |
| NIST AI RMF MAP | Planner identifies context and stakeholders per task |
| NIST AI RMF MEASURE | Critic evaluates outcomes against acceptance criteria |
| NIST AI RMF MANAGE | Expert-Critic feedback loop enables continuous risk response |
| ISO 42001 | Audit trail at every boundary satisfies AI management system auditability |

---

## Known Uses

- Planner-Actor-Critic pipelines in 3D modeling agents (conceptually similar; no Expert role)
- Reflection and CriticGPT-style code review patterns (Actor-Critic loop without formal Expert)
- Amazon Kiro IDE Spec-Driven Development (closest analog; lacks AI Critic as persistent role)
- Microsoft Copilot Studio structured approvals (no cyclical Expert loop)

---

## Related Patterns

| EPAC Role | Pattern Precedent |
|-----------|-------------------|
| Expert | Human-in-the-loop supervisor; Coactive Design (IHMC) |
| Planner | Planning / task decomposition pattern (Ng, Arsanjani) |
| Actor | Tool-use / executor pattern (Ng, Dibia) |
| Critic | Reflection / critique / evaluator pattern (Ng's Reflection; CriticGPT) |
| EPAC loop | Multi-agent coordinator / pipeline pattern |

EPAC is a **macro-pattern**: a named composition that enterprises can adopt as a reference architecture, rather than a primitive building block.

---

## Liabilities

- Orchestration complexity (state management, routing, retries) increases vs. single-agent approaches
- Latency and cost are higher than a monolithic agent flow
- Poorly specified boundaries (e.g. Planner doing implementation; Critic changing code) blur responsibilities
- Over-reliance on Critic quality can create false sense of security if its checks are weak
- Actor-Critic loops can stall if the Critic raises issues the Actor cannot resolve without Planner-level changes

---

## Implementation Notes

See the EPAC Python SDK for a reference implementation:

```
github.com/thomason-io/epac
```

The SDK uses LangGraph for HITL-capable orchestration, Pydantic for typed artifacts, and LiteLLM for model-agnostic LLM calls.
