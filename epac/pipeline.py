"""
EPACPipeline – the LangGraph-powered orchestrator.

Wires Expert → Planner → Actor ⇌ Critic → Expert into a stateful, resumable
graph with human-in-the-loop approval gates.

LangGraph concepts used:
  - StateGraph with typed EPACState nodes
  - interrupt_before for Expert approval gates
  - Checkpointer for durable state (Postgres or SQLite)
  - Command-based human input injection
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field

from epac.artifacts import EPACSpec, EPACPlan, EPACImplementation, EPACReview
from epac.roles import ExpertRole, PlannerRole, ActorRole, CriticRole
from epac.roles.base import RoleConfig
from epac.state import EPACState, AutonomyLevel, StageStatus

logger = logging.getLogger(__name__)

try:
    from langgraph.graph import StateGraph, END  # type: ignore[import]
    from langgraph.checkpoint.memory import MemorySaver  # type: ignore[import]
    _LANGGRAPH_AVAILABLE = True
except ImportError:
    _LANGGRAPH_AVAILABLE = False


class EPACConfig(BaseModel):
    """Top-level configuration for an EPACPipeline instance."""

    # LLM settings
    llm_model: str = Field(
        default="openai/gpt-4o",
        description="Default LLM for Planner and Actor (LiteLLM model string)",
    )
    critic_llm_model: str = Field(
        default="anthropic/claude-3-7-sonnet-20250219",
        description="Critic uses a different provider to reduce correlated failures",
    )

    # Quality loop
    autonomy_level: AutonomyLevel = Field(
        default=AutonomyLevel.GATED,
        description="Default autonomy level; may be overridden per-spec via risk_level",
    )
    max_actor_critic_iterations: int = Field(
        default=5,
        description="Maximum Actor-Critic iterations. Each iteration improves output quality; 3-5 is typical.",
    )

    # Critic tooling
    critic_tools: list[str] = Field(
        default_factory=list,
        description="Critic tool integrations to run (e.g. ['bandit', 'semgrep'])",
    )

    # Role system prompt overrides
    expert_system_prompt: str = ""
    planner_system_prompt: str = ""
    actor_system_prompt: str = ""
    critic_system_prompt: str = ""

    # Prompt size limits (set to 0 to disable truncation)
    max_file_content_chars: int = Field(
        default=20000,
        description="Max chars of file content included per file in the Critic review prompt (0 = unlimited)",
    )
    max_actor_feedback_chars: int = Field(
        default=10000,
        description="Max chars of actor_feedback included in Actor re-iteration prompt (0 = unlimited)",
    )
    max_actor_feedback_findings: int = Field(
        default=50,
        description="Max number of blocking findings passed back to Actor per iteration (0 = unlimited)",
    )

    # LLM token budget
    max_tokens: int = Field(
        default=16000,
        description="Max tokens for all LLM calls (applies to Planner, Actor, Critic)",
    )

    # Checkpointing
    use_memory_checkpointer: bool = Field(
        default=True,
        description="Use in-memory checkpointer (dev). Set False to provide your own.",
    )

    # Metadata
    pipeline_name: str = "epac-pipeline"


class EPACResult(BaseModel):
    """Final result returned by EPACPipeline.run()."""

    pipeline_id: str
    completed: bool
    failed: bool
    failure_reason: str = ""
    spec: EPACSpec | None = None
    plan: EPACPlan | None = None
    implementations: list[EPACImplementation] = Field(default_factory=list)
    reviews: list[EPACReview] = Field(default_factory=list)
    audit_log: list[dict[str, Any]] = Field(default_factory=list)
    actor_critic_iterations: int = 0

    @property
    def final_implementation(self) -> EPACImplementation | None:
        return self.implementations[-1] if self.implementations else None

    @property
    def final_review(self) -> EPACReview | None:
        return self.reviews[-1] if self.reviews else None


class EPACPipeline:
    """
    The EPAC orchestration pipeline.

    Usage (simple — no persistent checkpointing)::

        pipeline = EPACPipeline(config=EPACConfig(llm_model="openai/gpt-4o"))
        result = await pipeline.run(goal="Add rate limiting to the API gateway")

    Usage (with human-in-the-loop)::

        pipeline = EPACPipeline(config=EPACConfig())
        thread_id = await pipeline.start(spec=my_spec)
        # … pipeline pauses at Expert plan approval gate …
        await pipeline.approve_plan(thread_id, notes="Looks good")
        # … pipeline runs Actor + Critic loop …
        await pipeline.approve_implementation(thread_id)
        result = pipeline.get_result(thread_id)
    """

    def __init__(self, config: EPACConfig | None = None) -> None:
        self.config = config or EPACConfig()
        self._expert = self._build_expert()
        self._planner = self._build_planner()
        self._actor = self._build_actor()
        self._critic = self._build_critic()
        self._graph = self._build_graph() if _LANGGRAPH_AVAILABLE else None
        self._thread_states: dict[str, EPACState] = {}  # Fallback for non-LangGraph use

    # ── Public API ───────────────────────────────────────────────────────────

    async def run(
        self,
        goal: str,
        expert_id: str = "expert",
        spec: EPACSpec | None = None,
        **spec_kwargs: Any,
    ) -> EPACResult:
        """
        Run a complete EPAC pipeline for a given goal.

        In non-interactive mode (no approval gates), the pipeline runs end-to-end.
        In HITL mode, this method raises `EPACApprovalRequired` at each gate —
        use `start()` / `approve_plan()` / `approve_implementation()` instead.

        Parameters
        ----------
        goal:
            High-level description of what the Expert wants built.
        expert_id:
            Identifier for the Expert (e.g. email or username).
        spec:
            Pre-built EPACSpec; if None, one is constructed from goal and spec_kwargs.
        **spec_kwargs:
            Extra EPACSpec fields (constraints, acceptance_criteria, risk_level, etc.)
        """
        if spec is None:
            spec = EPACSpec(created_by=expert_id, title=goal, goal=goal, **spec_kwargs)

        pipeline_id = str(uuid.uuid4())

        if _LANGGRAPH_AVAILABLE:
            return await self._run_langgraph(pipeline_id, spec)
        else:
            return await self._run_simple(pipeline_id, spec)

    async def start(self, spec: EPACSpec) -> str:
        """
        Start a HITL pipeline and return a thread_id for later resumption.

        The pipeline will run until the first approval gate and then pause.
        """
        pipeline_id = str(uuid.uuid4())
        if _LANGGRAPH_AVAILABLE:
            thread_id = await self._start_langgraph(pipeline_id, spec)
        else:
            thread_id = await self._start_simple(pipeline_id, spec)
        return thread_id

    async def approve_plan(self, thread_id: str, notes: str = "") -> None:
        """Inject an Expert plan approval into a paused pipeline."""
        await self._inject_decision(thread_id, "approve", notes)

    async def reject_plan(self, thread_id: str, notes: str = "") -> None:
        """Inject an Expert plan rejection (triggers re-planning)."""
        await self._inject_decision(thread_id, "reject", notes)

    async def approve_implementation(self, thread_id: str, notes: str = "") -> None:
        """Inject Expert final approval into a paused pipeline."""
        await self._inject_decision(thread_id, "approve", notes)

    async def reject_implementation(self, thread_id: str, notes: str = "") -> None:
        """Inject Expert rejection of the final implementation."""
        await self._inject_decision(thread_id, "reject", notes)

    def get_result(self, thread_id: str) -> EPACResult:
        """Retrieve the final result for a completed or failed pipeline."""
        state = self._get_state(thread_id)
        return self._state_to_result(state)

    # ── LangGraph implementation ─────────────────────────────────────────────

    def _build_graph(self) -> Any:
        """Build the LangGraph StateGraph for the EPAC pipeline."""
        if not _LANGGRAPH_AVAILABLE:
            return None

        from langgraph.graph import StateGraph, END

        # We use a plain dict as the graph state (LangGraph requirement)
        graph = StateGraph(dict)

        # Add nodes
        graph.add_node("expert", self._expert_node)
        graph.add_node("planner", self._planner_node)
        graph.add_node("actor", self._actor_node)
        graph.add_node("critic", self._critic_node)

        # Entry point
        graph.set_entry_point("expert")

        # Edges
        graph.add_conditional_edges(
            "expert",
            self._route_from_expert,
            {
                "planner": "planner",
                "actor": "actor",
                "complete": END,
                "wait": "expert",  # Interrupt — graph pauses here
            },
        )
        graph.add_conditional_edges(
            "planner",
            self._route_from_planner,
            {
                "expert": "expert",   # Plan approval gate
                "actor": "actor",     # Skip approval (low-risk)
                "failed": END,
            },
        )
        graph.add_conditional_edges(
            "actor",
            self._route_from_actor,
            {
                "critic": "critic",
                "failed": END,
            },
        )
        graph.add_conditional_edges(
            "critic",
            self._route_from_critic,
            {
                "actor": "actor",     # Actor-Critic loop
                "expert": "expert",   # Final approval gate
                "complete": END,      # Skip final approval
                "failed": END,
            },
        )

        checkpointer = MemorySaver() if self.config.use_memory_checkpointer else None
        return graph.compile(
            checkpointer=checkpointer,
            interrupt_before=["expert"],  # Pause before Expert node for HITL
        )

    # ── Node wrappers ─────────────────────────────────────────────────────
    # LangGraph passes only the delta (changed keys) to nodes, not the full
    # accumulated state.  We merge the incoming partial dict with the stored
    # full state (keyed by pipeline_id) so every node sees a complete picture.

    def _full_state(self, state: dict, config: dict | None = None) -> dict:
        """Merge partial LangGraph state with the stored baseline.

        LangGraph only passes changed keys to each node, so we merge the
        incoming delta against the full baseline we keep in _thread_states.
        We resolve pipeline_id from (in priority order):
          1. config["configurable"]["thread_id"] — always correct in concurrent use
          2. state.get("pipeline_id") — present when LangGraph includes it in the delta
        If pipeline_id still cannot be resolved, we raise to prevent silent corruption.
        """
        pid = ""
        if config is not None:
            pid = config.get("configurable", {}).get("thread_id", "")
        if not pid:
            pid = state.get("pipeline_id", "")
        if not pid:
            raise RuntimeError(
                "_full_state: cannot resolve pipeline_id — no config thread_id and "
                "pipeline_id is absent from the state delta. This is a bug."
            )
        base = self._thread_states.get(pid)
        if base is None:
            base = {}
        if hasattr(base, "model_dump"):
            base = base.model_dump()
        merged = {**base, **{k: v for k, v in state.items() if v is not None}}
        return merged

    def _save_state(self, pipeline_id: str, updates: dict) -> None:
        """Persist the updated state dict for this pipeline."""
        existing = self._thread_states.get(pipeline_id, {})
        if hasattr(existing, "model_dump"):
            existing = existing.model_dump()
        self._thread_states[pipeline_id] = {**existing, **updates}

    async def _expert_node(self, state: dict, config: dict) -> dict:
        full = self._full_state(state, config)
        updates = await self._expert.run(full)
        self._save_state(full.get("pipeline_id", ""), updates)
        return updates

    async def _planner_node(self, state: dict, config: dict) -> dict:
        full = self._full_state(state, config)
        updates = await self._planner.run(full)
        self._save_state(full.get("pipeline_id", ""), updates)
        return updates

    async def _actor_node(self, state: dict, config: dict) -> dict:
        full = self._full_state(state, config)
        updates = await self._actor.run(full)
        self._save_state(full.get("pipeline_id", ""), updates)
        return updates

    async def _critic_node(self, state: dict, config: dict) -> dict:
        full = self._full_state(state, config)
        updates = await self._critic.run(full)
        self._save_state(full.get("pipeline_id", ""), updates)
        return updates

    # ── Routing functions ────────────────────────────────────────────────────

    def _route_from_expert(self, state: dict) -> str:
        full = self._full_state(state)
        current = full.get("current_stage", "expert")
        if full.get("completed"):
            return "complete"
        if current == "planner":
            return "planner"
        if current == "actor":
            return "actor"
        return "wait"

    def _route_from_planner(self, state: dict) -> str:
        full = self._full_state(state)
        if full.get("failed"):
            return "failed"
        current = full.get("current_stage", "expert")
        if current == "actor":
            return "actor"
        return "expert"  # Approval gate

    def _route_from_actor(self, state: dict) -> str:
        full = self._full_state(state)
        if full.get("failed"):
            return "failed"
        return "critic"

    def _route_from_critic(self, state: dict) -> str:
        full = self._full_state(state)
        if full.get("failed"):
            return "failed"
        if full.get("completed"):
            return "complete"
        current = full.get("current_stage", "actor")
        if current == "actor":
            return "actor"
        if current == "expert":
            return "expert"
        return "complete"

    # ── Simple (non-LangGraph) fallback ─────────────────────────────────────

    async def _run_simple(self, pipeline_id: str, spec: EPACSpec) -> EPACResult:
        """
        Minimal sequential runner when LangGraph is not installed.

        Runs without HITL interrupts — all approval gates auto-approve.
        Install langgraph for full HITL support.
        """
        state = EPACState(
            pipeline_id=pipeline_id,
            max_actor_critic_iterations=self.config.max_actor_critic_iterations,
            autonomy_level=self.config.autonomy_level,
        )

        # Expert: submit spec
        updates = self._expert.submit_spec(state.model_dump(), spec)
        state = EPACState(**{**state.model_dump(), **updates})

        # Planner
        updates = await self._planner.run(state.model_dump())
        state = EPACState(**{**state.model_dump(), **updates})

        # Auto-approve plan (simple mode)
        if state.plan and state.plan.expert_approved is None:
            plan = state.plan.model_copy(update={"expert_approved": True})
            state = state.model_copy(update={"plan": plan, "current_stage": "actor"})

        # Actor-Critic loop
        while not state.completed and not state.failed:
            if state.actor_critic_loop_exhausted:
                state = state.model_copy(
                    update={
                        "failed": True,
                        "failure_reason": "Actor-Critic loop exhausted",
                    }
                )
                break

            # Actor
            updates = await self._actor.run(state.model_dump())
            state = EPACState(**{**state.model_dump(), **updates})
            if state.failed:
                break

            # Critic
            updates = await self._critic.run(state.model_dump())
            state = EPACState(**{**state.model_dump(), **updates})
            if state.failed:
                break

            if state.critic_passed:
                # Auto-approve final implementation (simple mode)
                state = state.model_copy(update={"completed": True})

        self._thread_states[pipeline_id] = state
        return self._state_to_result(state)

    async def _run_langgraph(self, pipeline_id: str, spec: EPACSpec) -> EPACResult:
        """Run via LangGraph graph."""
        thread_id = await self._start_langgraph(pipeline_id, spec)
        config = {"configurable": {"thread_id": thread_id}}
        # For non-HITL usage, auto-approve all Expert interrupt gates.
        # Use graph_state.next to detect when the graph is parked at an interrupt
        # rather than checking expert_status (which is never set before the Expert
        # node runs, causing the old loop to exit prematurely).
        for _ in range(20):  # Safety iteration cap
            s = self._get_state(thread_id)
            if s.completed or s.failed:
                break
            graph_state = self._graph.get_state(config)
            if graph_state.next and "expert" in graph_state.next:
                await self._inject_decision(thread_id, "approve", "")
            else:
                break
        return self.get_result(thread_id)

    async def _start_simple(self, pipeline_id: str, spec: EPACSpec) -> str:
        state = EPACState(pipeline_id=pipeline_id)
        updates = self._expert.submit_spec(state.model_dump(), spec)
        state = EPACState(**{**state.model_dump(), **updates})
        self._thread_states[pipeline_id] = state
        return pipeline_id

    async def _start_langgraph(self, pipeline_id: str, spec: EPACSpec) -> str:
        init_state = EPACState(
            pipeline_id=pipeline_id,
            max_actor_critic_iterations=self.config.max_actor_critic_iterations,
            autonomy_level=self.config.autonomy_level,
        ).model_dump()
        # Store full baseline so nodes can merge against it
        self._thread_states[pipeline_id] = dict(init_state)
        config = {"configurable": {"thread_id": pipeline_id}}
        # Seed graph with full state (first pass hits expert node which just
        # sets AWAITING_APPROVAL since spec is None)
        async for _ in self._graph.astream(init_state, config=config):
            pass
        # Inject spec via update_state
        spec_updates = self._expert.submit_spec(init_state, spec)
        self._graph.update_state(config, spec_updates)
        self._save_state(pipeline_id, spec_updates)
        # Resume — now spec is set, expert routes to planner
        async for _ in self._graph.astream(None, config=config):
            pass
        return pipeline_id

    async def _inject_decision(
        self, thread_id: str, decision: str, notes: str
    ) -> None:
        updates = {"pending_decision": decision, "pending_decision_notes": notes}
        if _LANGGRAPH_AVAILABLE and self._graph:
            config = {"configurable": {"thread_id": thread_id}}
            self._graph.update_state(config, updates)
            async for _ in self._graph.astream(None, config=config):
                pass
        else:
            if thread_id in self._thread_states:
                state = self._thread_states[thread_id]
                state_dict = state.model_dump() if hasattr(state, "model_dump") else dict(state)
                state_dict.setdefault("pipeline_id", thread_id)
                self._thread_states[thread_id] = EPACState(
                    **{**state_dict, **updates}
                )

    def _get_state(self, thread_id: str) -> EPACState:
        if _LANGGRAPH_AVAILABLE and self._graph:
            config = {"configurable": {"thread_id": thread_id}}
            lg_values = dict(self._graph.get_state(config).values)
            # Merge with our saved full state to ensure all fields are present
            base = self._thread_states.get(thread_id, {})
            if hasattr(base, "model_dump"):
                base = base.model_dump()
            merged = {**base, **{k: v for k, v in lg_values.items() if v is not None}}
            merged.setdefault("pipeline_id", thread_id)
            return EPACState(**merged)
        stored = self._thread_states.get(thread_id, {})
        if hasattr(stored, "model_dump"):
            return stored
        if isinstance(stored, dict):
            stored.setdefault("pipeline_id", thread_id)
            return EPACState(**stored)
        return EPACState(pipeline_id=thread_id)

    def _state_to_result(self, state: EPACState) -> EPACResult:
        from epac.artifacts import EPACSpec, EPACPlan, EPACImplementation, EPACReview

        def _hydrate(model_class: Any, data: Any) -> Any:
            if data is None:
                return None
            if isinstance(data, dict):
                return model_class(**data)
            return data

        return EPACResult(
            pipeline_id=state.pipeline_id,
            completed=state.completed,
            failed=state.failed,
            failure_reason=state.failure_reason,
            spec=_hydrate(EPACSpec, state.spec),
            plan=_hydrate(EPACPlan, state.plan),
            implementations=[
                _hydrate(EPACImplementation, i) for i in state.implementations
            ],
            reviews=[_hydrate(EPACReview, r) for r in state.reviews],
            audit_log=[
                e.model_dump() if hasattr(e, "model_dump") else e
                for e in state.audit_log
            ],
            actor_critic_iterations=state.actor_critic_iteration,
        )

    # ── Role builders ────────────────────────────────────────────────────────

    def _build_expert(self) -> ExpertRole:
        cfg = RoleConfig(role_name="expert")
        if self.config.expert_system_prompt:
            cfg.system_prompt = self.config.expert_system_prompt
        return ExpertRole(cfg)

    def _build_planner(self) -> PlannerRole:
        from epac.roles.planner import PLANNER_SYSTEM_PROMPT
        cfg = RoleConfig(
            role_name="planner",
            llm_model=self.config.llm_model,
            system_prompt=self.config.planner_system_prompt or PLANNER_SYSTEM_PROMPT,
            max_tokens=self.config.max_tokens,
        )
        return PlannerRole(cfg)

    def _build_actor(self) -> ActorRole:
        from epac.roles.actor import ACTOR_SYSTEM_PROMPT
        cfg = RoleConfig(
            role_name="actor",
            llm_model=self.config.llm_model,
            system_prompt=self.config.actor_system_prompt or ACTOR_SYSTEM_PROMPT,
            max_tokens=self.config.max_tokens,
            max_actor_feedback_chars=self.config.max_actor_feedback_chars,
            max_actor_feedback_findings=self.config.max_actor_feedback_findings,
        )
        return ActorRole(cfg)

    def _build_critic(self) -> CriticRole:
        from epac.roles.critic import CRITIC_SYSTEM_PROMPT
        cfg = RoleConfig(
            role_name="critic",
            llm_model=self.config.critic_llm_model,
            tools=self.config.critic_tools,
            system_prompt=self.config.critic_system_prompt or CRITIC_SYSTEM_PROMPT,
            max_tokens=self.config.max_tokens,
            max_file_content_chars=self.config.max_file_content_chars,
        )
        return CriticRole(cfg)
