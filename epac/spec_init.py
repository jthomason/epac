"""
epac spec init — interactive prompt-to-EPACSpec converter.

Flow:
  1. User pastes a free-form prompt (or provides --prompt / --prompt-file).
  2. An LLM extracts everything it can: goal, title, constraints, acceptance
     criteria, scope hints, risk signals.
  3. The CLI asks targeted follow-up questions only for fields that are missing
     or ambiguous — never asks for something the LLM already found.
  4. The completed EPACSpec is written to a YAML file (default: epac-spec.yaml).

The output is a valid EPACSpec that can be passed directly to `epac run --spec`.
"""

from __future__ import annotations

import json
import os
import sys
import textwrap
from pathlib import Path
from typing import Any

import yaml

# ── ANSI colour helpers ────────────────────────────────────────────────────────

_USE_COLOR = sys.stdout.isatty()


def _c(code: str, text: str) -> str:
    if not _USE_COLOR:
        return text
    return f"\033[{code}m{text}\033[0m"


def teal(t: str) -> str:   return _c("36", t)
def bold(t: str) -> str:   return _c("1", t)
def dim(t: str) -> str:    return _c("2", t)
def green(t: str) -> str:  return _c("32", t)
def yellow(t: str) -> str: return _c("33", t)
def red(t: str) -> str:    return _c("31", t)


# ── Prompt reading ─────────────────────────────────────────────────────────────

MULTILINE_SENTINEL = "EOF"


def read_prompt_interactively() -> str:
    """Read a free-form prompt from stdin, supporting multi-line input."""
    print()
    print(bold("Paste your prompt below."))
    print(dim(f'  Type or paste your text. Enter a blank line or "{MULTILINE_SENTINEL}" on its own line when done.'))
    print()

    lines: list[str] = []
    try:
        while True:
            line = input()
            if line.strip() == MULTILINE_SENTINEL or (not line.strip() and lines):
                break
            lines.append(line)
    except EOFError:
        pass

    return "\n".join(lines).strip()


# ── LLM extraction ─────────────────────────────────────────────────────────────

EXTRACTION_SYSTEM = """\
You are an expert at converting informal human prompts into structured
EPACSpec fields. The EPACSpec is a typed artifact that defines the goal,
constraints, acceptance criteria, scope, and risk level for an AI-driven
task pipeline.

Extract as much as you can from the user's prompt. For fields you cannot
confidently infer, return null. Do not invent information that is not
present or strongly implied.

Return ONLY a JSON object with these keys (all optional except goal and title):

{
  "title": "Short imperative title (5-8 words)",
  "goal": "Full goal statement, clarified and expanded from the prompt",
  "background": "Domain context inferred from the prompt, or null",
  "in_scope": ["list of things explicitly or clearly in scope"],
  "out_of_scope": ["list of things explicitly excluded, or []"],
  "acceptance_criteria": [
    {"given": "...", "when": "...", "then": "..."}
  ],
  "constraints": [
    {"category": "security|compliance|legal-policy|performance|style|output-format",
     "description": "...",
     "mandatory": true}
  ],
  "risk_level": "low|medium|high|critical or null if unclear",
  "tags": ["inferred domain tags"],
  "confidence": {
    "goal": "high|medium|low",
    "constraints": "high|medium|low",
    "acceptance_criteria": "high|medium|low",
    "risk_level": "high|medium|low"
  }
}

Guidelines:
- title: imperative verb phrase, e.g. "Review CloudVault vendor MSA"
- goal: rewrite the prompt goal clearly and completely — do not truncate
- acceptance_criteria: only include if the prompt strongly implies testable outcomes.
  Use GIVEN/WHEN/THEN. 1-3 criteria max from extraction alone.
- constraints: only include if explicitly stated or strongly implied (e.g. "don't
  change existing routes" -> out-of-scope constraint, "must be GDPR compliant" ->
  compliance constraint)
- risk_level: infer from language like "production", "legal", "sign", "deploy",
  "critical", "sensitive data". Default to null if ambiguous.
- Return valid JSON only. No markdown fences, no explanation.
"""


def _extract_with_llm(prompt: str, model: str) -> dict[str, Any]:
    """Call the LLM to extract structured fields from the raw prompt."""
    try:
        import litellm  # type: ignore
    except ImportError:
        print(red("ERROR: litellm is not installed. Run: pip install litellm"), file=sys.stderr)
        sys.exit(1)

    messages = [
        {"role": "system", "content": EXTRACTION_SYSTEM},
        {"role": "user", "content": f"User prompt:\n\n{prompt}"},
    ]

    try:
        response = litellm.completion(
            model=model,
            messages=messages,
            max_tokens=2000,
            temperature=0.1,
        )
        raw = response.choices[0].message.content.strip()
        # Strip markdown fences if the model added them anyway
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
        return json.loads(raw)
    except json.JSONDecodeError as e:
        print(yellow(f"Warning: LLM returned malformed JSON ({e}). Proceeding with empty extraction."))
        return {}
    except Exception as e:
        print(yellow(f"Warning: LLM extraction failed ({e}). Proceeding with manual entry."))
        return {}


# ── Interactive elicitation ────────────────────────────────────────────────────

def _ask(prompt: str, default: str | None = None, required: bool = True) -> str:
    """Ask a single question and return the answer. Loops until non-empty if required."""
    if default:
        display = f"{prompt} {dim(f'[{default}]')} "
    else:
        display = f"{prompt} "

    while True:
        try:
            answer = input(display).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            sys.exit(0)

        if answer:
            return answer
        if default is not None:
            return default
        if not required:
            return ""
        print(dim("  (required — please enter a value)"))


def _ask_list(prompt: str, hint: str = "", existing: list[str] | None = None) -> list[str]:
    """Ask for a comma or newline separated list. Returns [] if blank."""
    if existing:
        print(dim(f"  Already have: {', '.join(existing[:3])}{'...' if len(existing) > 3 else ''}"))

    print(f"{prompt}")
    if hint:
        print(dim(f"  {hint}"))
    if existing:
        print(dim("  Press Enter to keep existing, or type new values to replace."))

    try:
        raw = input("  > ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        sys.exit(0)

    if not raw:
        return existing or []

    # Split on commas or newlines
    items = [item.strip() for item in raw.replace(";", ",").split(",") if item.strip()]
    return items


def _ask_risk(extracted: str | None) -> str:
    levels = {
        "1": "low",
        "2": "medium",
        "3": "high",
        "4": "critical",
        "l": "low",
        "m": "medium",
        "h": "high",
        "c": "critical",
    }
    descriptions = {
        "low":      "Informational or exploratory. Fully reversible. No production impact.",
        "medium":   "Limited blast radius. Reviewed before deploy. Some irreversibility.",
        "high":     "Production system, sensitive data, legal/financial exposure, or hard to reverse.",
        "critical": "Catastrophic if wrong. Requires multiple human approvals. Regulatory risk.",
    }

    default_key = None
    if extracted:
        default_key = extracted
        print(dim(f"  (inferred from your prompt: {extracted})"))

    print()
    for key, label in [("low", "low"), ("medium", "medium"), ("high", "high"), ("critical", "critical")]:
        marker = teal("*") if key == default_key else " "
        print(f"  {marker} {bold(key):20s} {dim(descriptions[key])}")
    print()

    while True:
        raw = _ask(
            f"Risk level {dim('(low/medium/high/critical)')}:",
            default=default_key,
            required=True,
        ).lower()
        result = levels.get(raw, raw)
        if result in ("low", "medium", "high", "critical"):
            return result
        print(dim("  Enter one of: low, medium, high, critical (or l/m/h/c)"))


def _build_acceptance_criteria_interactively(existing: list[dict] | None) -> list[dict]:
    """Walk the user through adding GIVEN/WHEN/THEN criteria."""
    criteria = list(existing or [])

    if criteria:
        print()
        print(dim(f"  Extracted {len(criteria)} acceptance criteria from your prompt:"))
        for i, c in enumerate(criteria, 1):
            print(dim(f"    {i}. GIVEN {c['given']} WHEN {c['when']} THEN {c['then']}"))

    print()
    add = _ask(
        f"Add acceptance criteria? {dim('(y/n)')}",
        default="n" if criteria else "y",
        required=False,
    ).lower()

    if add not in ("y", "yes"):
        return criteria

    print(dim("  Describe when the task is 'done'. Use plain language — EPAC will structure it."))
    print(dim("  Example: 'when the user submits the form, the contract is saved and a confirmation email is sent'"))
    print(dim("  Enter one criterion per line. Blank line to finish."))
    print()

    while True:
        try:
            raw = input("  Criterion: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not raw:
            break

        # Simple heuristic decomposition: look for given/when/then keywords
        # Otherwise treat the whole thing as the "when" clause
        lower = raw.lower()
        if "given" in lower and "when" in lower and "then" in lower:
            # User wrote it in GIVEN/WHEN/THEN format already
            # Parse naively
            try:
                g = raw.lower().split("given")[1].split("when")[0].strip()
                w = raw.lower().split("when")[1].split("then")[0].strip()
                t = raw.lower().split("then")[1].strip()
                criteria.append({"given": g, "when": w, "then": t})
                print(green("    Added."))
                continue
            except IndexError:
                pass

        # Treat as a plain outcome statement — LLM will have structured the
        # proper ones; for additional ones entered here, build a simple structure
        criteria.append({
            "given": "the task has been executed",
            "when": raw,
            "then": "the outcome meets the stated requirement",
        })
        print(green("    Added."))

    return criteria


def _build_constraints_interactively(existing: list[dict] | None) -> list[dict]:
    """Ask about common constraint categories, pre-filling from extraction."""
    constraints = list(existing or [])

    if constraints:
        print()
        print(dim(f"  Extracted {len(constraints)} constraints from your prompt:"))
        for c in constraints:
            marker = teal("*") if c.get("mandatory") else dim("-")
            print(dim(f"    {marker} [{c['category']}] {c['description']}"))

    print()
    add = _ask(
        f"Add constraints or hard limits? {dim('(y/n)')}",
        default="n" if constraints else "y",
        required=False,
    ).lower()

    if add not in ("y", "yes"):
        return constraints

    print(dim("  Describe things the AI must not do, policies it must follow, or hard limits."))
    print(dim("  Examples: 'must not modify existing database schema', 'must comply with GDPR',"))
    print(dim("            'liability cap must be at least 12 months of fees'"))
    print(dim("  Enter one constraint per line. Blank line to finish."))
    print()

    categories = {
        "security": ["secret", "token", "password", "encrypt", "auth", "credential"],
        "compliance": ["gdpr", "hipaa", "sox", "pci", "regulatory", "legal", "comply", "regulation"],
        "legal-policy": ["liability", "indemnif", "contract", "clause", "sign", "ip ", "ownership"],
        "performance": ["latency", "throughput", "speed", "fast", "slow", "ms ", "second"],
        "style": ["format", "style", "naming", "convention", "lint"],
    }

    def _infer_category(text: str) -> str:
        lower = text.lower()
        for cat, keywords in categories.items():
            if any(kw in lower for kw in keywords):
                return cat
        return "policy"

    while True:
        try:
            raw = input("  Constraint: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not raw:
            break

        cat = _infer_category(raw)
        is_red_line = any(w in raw.lower() for w in [
            "must not", "never", "do not", "cannot", "red line", "red-line", "block", "prohibit"
        ])

        constraints.append({
            "category": "red-line" if is_red_line else cat,
            "description": raw,
            "mandatory": True,
        })
        print(green(f"    Added [{('red-line' if is_red_line else cat)}]."))

    return constraints


# ── YAML serialisation ─────────────────────────────────────────────────────────

def _spec_to_yaml(spec: dict[str, Any]) -> str:
    """Serialise the spec dict to clean, human-readable YAML."""

    # Custom representer for multiline strings
    def _str_representer(dumper: yaml.Dumper, data: str):  # type: ignore
        if "\n" in data or len(data) > 80:
            return dumper.represent_scalar("tag:yaml.org,2002:str", data, style=">")
        return dumper.represent_scalar("tag:yaml.org,2002:str", data)

    yaml.add_representer(str, _str_representer)

    header = textwrap.dedent(f"""\
        # EPACSpec — generated by `epac spec init`
        # Edit this file, then run: epac run --spec <this-file>
        # Docs: https://github.com/thomason-io/epac
        #
    """)
    return header + yaml.dump(spec, sort_keys=False, allow_unicode=True, width=88)


# ── Main entry point ───────────────────────────────────────────────────────────

def run_spec_init(
    prompt: str | None = None,
    prompt_file: str | None = None,
    output: str = "epac-spec.yaml",
    model: str = "anthropic/claude-haiku-4-5",
    expert_id: str | None = None,
    no_llm: bool = False,
) -> Path:
    """
    Interactive prompt-to-EPACSpec wizard.

    Parameters
    ----------
    prompt:       Raw prompt string (if provided via --prompt flag)
    prompt_file:  Path to a file containing the prompt (--prompt-file)
    output:       Output YAML file path
    model:        LiteLLM model string for extraction
    expert_id:    Pre-fill created_by (from env or flag)
    no_llm:       Skip LLM extraction, go straight to interactive mode

    Returns the Path of the written spec file.
    """

    # ── Banner ────────────────────────────────────────────────────────────────
    print()
    print(bold(teal("epac spec init")))
    print(dim("Convert a prompt into a structured EPACSpec. Takes ~60 seconds."))
    print(dim("─" * 60))

    # ── Get the raw prompt ────────────────────────────────────────────────────
    if prompt_file:
        raw_prompt = Path(prompt_file).read_text().strip()
        print(f"\n{dim('Prompt loaded from:')} {prompt_file}")
    elif prompt:
        raw_prompt = prompt.strip()
        print(f"\n{dim('Prompt:')} {raw_prompt[:120]}{'...' if len(raw_prompt) > 120 else ''}")
    else:
        raw_prompt = read_prompt_interactively()

    if not raw_prompt:
        print(red("ERROR: No prompt provided."), file=sys.stderr)
        sys.exit(1)

    # ── LLM extraction ────────────────────────────────────────────────────────
    extracted: dict[str, Any] = {}
    if not no_llm:
        print()
        print(dim("Extracting structure from your prompt..."), end=" ", flush=True)
        extracted = _extract_with_llm(raw_prompt, model)
        confidence = extracted.pop("confidence", {})
        print(green("done"))

        # Show what was found
        found = [k for k in ("title", "goal", "constraints", "acceptance_criteria", "risk_level") if extracted.get(k)]
        if found:
            print(dim(f"  Found: {', '.join(found)}"))
        missing = [k for k in ("title", "goal", "constraints", "acceptance_criteria", "risk_level") if not extracted.get(k)]
        if missing:
            print(dim(f"  Will ask about: {', '.join(missing)}"))
    else:
        confidence = {}

    print()
    print(dim("─" * 60))
    print(bold("A few questions to complete the spec."))
    print(dim("Press Enter to accept a suggested value shown in [brackets]."))
    print()

    # ── Expert identity ───────────────────────────────────────────────────────
    default_expert = (
        expert_id
        or os.environ.get("EPAC_EXPERT_ID")
        or os.environ.get("GIT_AUTHOR_EMAIL")
        or os.environ.get("USER", "")
    )
    created_by = _ask(
        f"{teal('1.')} Your email or identifier {dim('(created_by)')}:",
        default=default_expert or None,
        required=True,
    )

    # ── Title ─────────────────────────────────────────────────────────────────
    print()
    title = _ask(
        f"{teal('2.')} Task title {dim('(short imperative phrase)')}:",
        default=extracted.get("title"),
        required=True,
    )

    # ── Goal ──────────────────────────────────────────────────────────────────
    print()
    extracted_goal = extracted.get("goal", "")
    if extracted_goal and confidence.get("goal") == "high":
        print(dim(f"  Goal (from prompt): {extracted_goal[:160]}{'...' if len(extracted_goal) > 160 else ''}"))
        keep = _ask(
            f"{teal('3.')} Keep this goal? {dim('(y/n or type a replacement)')}",
            default="y",
            required=False,
        )
        if keep.lower() in ("y", "yes", ""):
            goal = extracted_goal
        else:
            goal = keep if len(keep) > 4 else _ask("  Enter the goal:", required=True)
    else:
        print(dim("  Describe what the AI should accomplish. Be specific."))
        goal = _ask(
            f"{teal('3.')} Goal:",
            default=extracted_goal or None,
            required=True,
        )

    # ── Background ────────────────────────────────────────────────────────────
    print()
    background_default = extracted.get("background", "")
    print(dim("  Optional: context that helps the AI understand your domain (system names,"))
    print(dim("  prior decisions, relevant constraints from outside this task)."))
    background = _ask(
        f"{teal('4.')} Background context {dim('(Enter to skip)')}:",
        default=background_default or None,
        required=False,
    )

    # ── Scope ─────────────────────────────────────────────────────────────────
    print()
    print(f"{teal('5.')} {bold('Scope')}")
    in_scope = _ask_list(
        "What is explicitly in scope? (comma-separated, or Enter to skip)",
        existing=extracted.get("in_scope") or [],
    )
    print()
    out_of_scope = _ask_list(
        "What is explicitly out of scope? (comma-separated, or Enter to skip)",
        existing=extracted.get("out_of_scope") or [],
    )

    # ── Risk level ────────────────────────────────────────────────────────────
    print()
    print(f"{teal('6.')} {bold('Risk level')}")
    print(dim("  How bad is it if the AI gets this wrong?"))
    risk_level = _ask_risk(extracted.get("risk_level"))

    # ── Acceptance criteria ───────────────────────────────────────────────────
    print()
    print(f"{teal('7.')} {bold('Acceptance criteria')}")
    print(dim("  How will you know the task is done correctly?"))
    acceptance_criteria = _build_acceptance_criteria_interactively(
        extracted.get("acceptance_criteria") or []
    )

    # ── Constraints ───────────────────────────────────────────────────────────
    print()
    print(f"{teal('8.')} {bold('Constraints and hard limits')}")
    constraints = _build_constraints_interactively(
        extracted.get("constraints") or []
    )

    # ── Human approval gates ──────────────────────────────────────────────────
    print()
    print(f"{teal('9.')} {bold('Human approval gates')}")
    print(dim("  EPAC can pause for human review at two points:"))
    print(dim("  (a) after the Planner produces a task plan, before the Actor runs"))
    print(dim("  (b) after the Critic produces its review, before results are acted on"))
    print()
    gate_plan = _ask(
        f"  Require human approval of the plan before execution? {dim('(y/n)')}:",
        default="y" if risk_level in ("high", "critical") else "n",
        required=False,
    ).lower() in ("y", "yes")

    gate_final = _ask(
        f"  Require human approval of the final review? {dim('(y/n)')}:",
        default="y",
        required=False,
    ).lower() in ("y", "yes")

    # ── Tags ──────────────────────────────────────────────────────────────────
    tags = extracted.get("tags", [])

    # ── Assemble spec dict ────────────────────────────────────────────────────
    import uuid
    from datetime import datetime, timezone

    spec: dict[str, Any] = {
        "id": str(uuid.uuid4()),
        "version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "created_by": created_by,
        "title": title,
        "goal": goal,
    }
    if background:
        spec["background"] = background
    if in_scope:
        spec["in_scope"] = in_scope
    if out_of_scope:
        spec["out_of_scope"] = out_of_scope
    if acceptance_criteria:
        spec["acceptance_criteria"] = acceptance_criteria
    if constraints:
        spec["constraints"] = constraints

    spec["risk_level"] = risk_level
    spec["requires_expert_plan_approval"] = gate_plan
    spec["requires_expert_final_approval"] = gate_final

    if tags:
        spec["tags"] = tags

    # ── Write file ────────────────────────────────────────────────────────────
    out_path = Path(output)
    out_path.write_text(_spec_to_yaml(spec))

    print()
    print(dim("─" * 60))
    print(bold(green("Spec written:")) + f" {out_path}")
    print()
    print(dim("Next steps:"))
    print(f"  {teal('epac run --spec')} {out_path}        {dim('# run the full pipeline')}")
    print(f"  {teal('epac run --spec')} {out_path} {dim('--dry-run')}  {dim('# validate without executing')}")
    print()

    return out_path
