# Changelog

All notable changes to EPAC are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

## [0.1.4] — 2026-05-01

### Changed (quality-first reframe)
- **Planner system prompt**: rewritten to position the Planner as a quality multiplier. The Planner's job is to set the Actor up for high-quality work by eliminating ambiguity before execution begins. Ambiguity in the plan becomes defects in the implementation.
- **Actor system prompt**: rewritten to frame the Actor-Critic loop as iterative quality improvement, not compliance enforcement. Critic feedback is described as collaborator notes. Actor is instructed to consider whether feedback reveals a deeper issue, not just patch flagged lines.
- **Critic system prompt**: rewritten to center output quality as the Critic's purpose. Critic is told that a thorough, actionable review produces better final outputs than one that simply blocks. Goal alignment check elevated as the most important check. "Use sparingly and precisely" guidance added to ERROR severity.
- **`EPACConfig`**: `max_actor_critic_iterations` field comment changed from `# Governance` to `# Quality loop`. Description updated to frame iterations as quality improvements.
- **README**: subtitle changed from "governed and auditable AI agent pipelines" to "structure your agentic workflows to produce better outputs through iteration and human judgment". Chapter 1 rewritten: lead problem is output quality, not accountability gap. Three failure modes reframed as quality failures (no shared definition of success, no second perspective, no structured iteration). "What Enterprises Actually Need" replaced with "What Actually Improves Output Quality". Chapter 2 intro rewritten: EPAC framed as quality framework first, governance second. "Why This Structure Works" expanded to explain the quality mechanism: Actor knows success criteria upfront, Critic has no investment in Actor's approach, loop compounds quality across revisions. Governance and audit properties explicitly positioned as downstream consequences.
- **Demo (`epac-demo/index.html`)**: added full Actor-Critic second iteration to the terminal animation. Critic iteration 1 now returns the Actor with specific feedback (two incomplete sub-clauses). Actor iteration 2 addresses both and finds a new HIGH finding (§7.4 force majeure carve-out that voids the SLA). Critic iteration 2 passes. Summary shows "Quality gain: §7.4 force majeure void found on iter 2 — missed on first pass". Demo subtitle and "With EPAC" bullet updated to describe iteration.
- **Demo**: version badge updated to v0.1.4.

## [0.1.3] — 2026-04-18

### Added
- `epac spec init` — interactive prompt-to-EPACSpec wizard (`epac/spec_init.py`)
  - Paste any free-form prompt; Haiku extracts goal, title, constraints, acceptance criteria, risk level, and scope in one LLM call
  - CLI asks only for what the LLM could not confidently infer — never redundant questions
  - GIVEN/WHEN/THEN acceptance criteria entry with plain-language input
  - Constraint category auto-inference (security, compliance, legal-policy, red-line, etc.)
  - Red-line detection from natural language ("must not", "never", "do not")
  - Human approval gate configuration (plan gate, final gate) with risk-aware defaults
  - Outputs a clean, commented YAML file ready for `epac run --spec`
  - `--prompt`, `--prompt-file`, `--output`, `--model`, `--expert-id`, `--no-llm` flags
  - Reads `EPAC_EXPERT_ID` and `EPAC_SPEC_MODEL` environment variables
  - 20 new tests (43 total, all passing)

## [0.1.2] – 2026-04-17

This release incorporates findings from an independent Gemini review of the full codebase.

### Fixed
- **[Critical] Concurrent pipeline state corruption** (`pipeline.py`): `_full_state` now resolves `pipeline_id` from the LangGraph `RunnableConfig` passed to each node, not from a fallback over all stored pipelines. Node functions accept `(state, config)`. The dangerous `next(reversed(...))` fallback is replaced with a hard `RuntimeError` if `pipeline_id` is unresolvable.
- **[Critical] Actor can commit malicious GitHub Actions workflows before Critic review** (`integrations/github/actor.py`): `GitHubActorTool._push_sync` now blocks any file change targeting `.github/workflows` or `.github/actions` before making any GitHub API call.
- **[Critical] Actor hallucinates test and build results** (`roles/actor.py`): `ACTOR_SYSTEM_PROMPT` now explicitly prohibits the LLM from fabricating `test_results` or `build_results`. These fields should be empty lists unless a real sandbox ran. A `# TODO: sandbox execution` marker documents where real execution should be wired in.
- **[High] SAST scanners silently skip files with duplicate names** (`integrations/_builtin.py`): `run_bandit` and `run_semgrep` now preserve the full relative path under `tmpdir` (e.g. `backend/utils.py` stays `backend/utils.py`, not `utils.py`), with `mkdir(parents=True)` to create intermediate directories.
- **[High] Critic silently drops all findings when LLM returns a dict instead of a list** (`roles/critic.py`): Findings are now normalized from `dict` to `list` before the per-finding loop, so `{"vuln1": {...}}` style LLM output is handled correctly.
- **[High] `_run_langgraph` auto-approve loop exits prematurely** (`pipeline.py`): Replaced the unreliable `expert_status == AWAITING_APPROVAL` check (which is never True when `interrupt_before=["expert"]` is active) with `graph_state.next` to detect when the graph is actually parked at an Expert interrupt.
- **[Medium] Unhandled `JSONDecodeError` in markdown block fallback** (`_llm.py`): If the extracted code block from a markdown-wrapped LLM response is still invalid JSON, a `ValueError` is now raised cleanly instead of re-throwing an unhandled `JSONDecodeError`.

### Added
- `tests/test_pipeline.py` (10 new tests): covers `EPACPipeline` compilation, `_run_simple` end-to-end with mocked LLM calls, HITL `approve_plan` / `reject_plan` / `approve_implementation` decision injection, and Actor-Critic loop exhaustion. Test suite grows from 13 to 23 tests.

## [0.1.1] – 2026-04-17

### Fixed
- LangGraph node wrappers now merge partial state deltas against a stored full-state baseline, preventing `pipeline_id` missing validation errors on every node after the first
- System prompts (`PLANNER_SYSTEM_PROMPT`, `ACTOR_SYSTEM_PROMPT`, `CRITIC_SYSTEM_PROMPT`) are now correctly applied when `EPACConfig` overrides `llm_model`; previously the pipeline passed blank prompts to all three LLM roles
- Added envelope unwrapping in `PlannerRole`, `ActorRole`, and `CriticRole` parsers to handle LLM responses wrapped in keys like `{"epac_plan": {...}}` or `{"reviews": [...]}`
- `PlannerRole` now normalizes `files_to_change` from LLM output into `PlanTask.artifacts` (Pydantic was silently discarding unknown fields, leaving tasks with no file guidance)
- `CriticRole` now sets `completed=True` when `requires_expert_final_approval=False` and the review passes, fixing pipelines that ran to completion but never exited
- `max_tokens` from `RoleConfig` is now passed through to all `call_llm_json` invocations; default raised from 8 192 to 16 000 to prevent truncation on larger implementations
- Prompt size limits are now configurable via `EPACConfig`: `max_file_content_chars` (default 20 000), `max_actor_feedback_chars` (default 10 000), `max_actor_feedback_findings` (default 50), and `max_tokens` (default 16 000). Set any to `0` for unlimited. Previously these were hardcoded at 3 000 / 2 000 / 10 respectively

## [0.1.0] – 2026-04-16

### Added
- Initial release of the EPAC Python SDK
- Four typed artifact classes: `EPACSpec`, `EPACPlan`, `EPACImplementation`, `EPACReview`
- Four role agent implementations: `ExpertRole`, `PlannerRole`, `ActorRole`, `CriticRole`
- `EPACPipeline` orchestrator with LangGraph backend and simple sequential fallback
- Five-level autonomy model (`AutonomyLevel`) with configurable approval gates
- Structured audit trail (`AuditEntry`) at every stage boundary
- SARIF 2.1.0 output from `CriticRole` (compatible with GitHub Code Scanning)
- Built-in Critic tool integrations: `bandit`, `semgrep`, `sarif` file loader
- `epac.integrations.github` extras: `GitHubActorTool`, `GitHubCriticTool`, SARIF upload
- GitHub Actions workflow YAML generator (`generate_workflow_yaml`)
- CLI (`epac run`, `epac report`, `epac workflow`)
- Pattern specification document (`docs/pattern.md`)
