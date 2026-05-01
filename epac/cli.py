"""
EPAC command-line interface.

Usage:
  epac spec init                                         # prompt-to-spec wizard
  epac spec init --prompt "review this contract"         # with inline prompt
  epac spec init --prompt-file brief.txt --output s.yaml # from file
  epac run --goal "Add rate limiting to the API" --expert-id alice@corp.com
  epac run --spec spec.json --output-file result.json
  epac report --result-file result.json
  epac workflow --trigger issue_comment > .github/workflows/epac.yml
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("epac.cli")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="epac",
        description="EPAC – Expert-Planner-Actor-Critic Framework CLI",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ── epac spec ─────────────────────────────────────────────────────────────
    spec_parser = subparsers.add_parser("spec", help="EPACSpec utilities")
    spec_sub = spec_parser.add_subparsers(dest="spec_command", required=True)

    init_parser = spec_sub.add_parser(
        "init",
        help="Convert a free-form prompt into a structured EPACSpec (interactive wizard)",
    )
    init_parser.add_argument(
        "--prompt", "-p",
        help="Prompt string to convert (if omitted, prompts interactively)",
    )
    init_parser.add_argument(
        "--prompt-file", "-f",
        help="Path to a text file containing the prompt",
    )
    init_parser.add_argument(
        "--output", "-o",
        default="epac-spec.yaml",
        help="Output YAML file path (default: epac-spec.yaml)",
    )
    init_parser.add_argument(
        "--model",
        default=os.environ.get("EPAC_SPEC_MODEL", "anthropic/claude-haiku-4-5"),
        help="LiteLLM model for extraction (default: anthropic/claude-haiku-4-5)",
    )
    init_parser.add_argument(
        "--expert-id",
        default=os.environ.get("EPAC_EXPERT_ID", ""),
        help="Pre-fill created_by field (also reads EPAC_EXPERT_ID env var)",
    )
    init_parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Skip LLM extraction and use fully manual entry",
    )

    # ── epac run ──────────────────────────────────────────────────────────────
    run_parser = subparsers.add_parser("run", help="Run an EPAC pipeline")
    run_parser.add_argument("--goal", help="High-level goal for the pipeline")
    run_parser.add_argument("--spec", help="Path to EPACSpec JSON file")
    run_parser.add_argument(
        "--expert-id", default="cli-user", help="Expert identifier"
    )
    run_parser.add_argument(
        "--llm-model", default="openai/gpt-4o", help="LiteLLM model string"
    )
    run_parser.add_argument(
        "--critic-model",
        default="anthropic/claude-3-7-sonnet-20250219",
        help="Critic LLM model (should differ from main model)",
    )
    run_parser.add_argument(
        "--autonomy-level",
        type=int,
        default=int(os.environ.get("EPAC_AUTONOMY_LEVEL", "3")),
        help="Autonomy level 1-5 (default 3 = Gated)",
    )
    run_parser.add_argument(
        "--critic-tools",
        default=os.environ.get("EPAC_CRITIC_TOOLS", ""),
        help="Comma-separated Critic tool integrations",
    )
    run_parser.add_argument(
        "--max-iterations", type=int, default=5, help="Max Actor-Critic iterations"
    )
    run_parser.add_argument("--output-file", help="Write EPACResult JSON to this file")

    # ── epac report ───────────────────────────────────────────────────────────
    report_parser = subparsers.add_parser("report", help="Report on a pipeline result")
    report_parser.add_argument("--result-file", required=True)
    report_parser.add_argument("--post-github-comment", action="store_true")

    # ── epac workflow ─────────────────────────────────────────────────────────
    workflow_parser = subparsers.add_parser(
        "workflow", help="Print a GitHub Actions workflow YAML"
    )
    workflow_parser.add_argument(
        "--trigger",
        choices=["issue_comment", "workflow_dispatch", "schedule"],
        default="issue_comment",
    )

    args = parser.parse_args()

    if args.command == "spec":
        _spec(args)
    elif args.command == "run":
        asyncio.run(_run(args))
    elif args.command == "report":
        _report(args)
    elif args.command == "workflow":
        _workflow(args)


def _spec(args: argparse.Namespace) -> None:
    if args.spec_command == "init":
        from epac.spec_init import run_spec_init
        run_spec_init(
            prompt=args.prompt or None,
            prompt_file=args.prompt_file or None,
            output=args.output,
            model=args.model,
            expert_id=args.expert_id or None,
            no_llm=args.no_llm,
        )


async def _run(args: argparse.Namespace) -> None:
    from epac.pipeline import EPACPipeline, EPACConfig
    from epac.state import AutonomyLevel

    if not args.goal and not args.spec:
        print("ERROR: Provide --goal or --spec", file=sys.stderr)
        sys.exit(1)

    critic_tools = [t.strip() for t in args.critic_tools.split(",") if t.strip()]

    config = EPACConfig(
        llm_model=args.llm_model,
        critic_llm_model=args.critic_model,
        autonomy_level=AutonomyLevel(args.autonomy_level),
        max_actor_critic_iterations=args.max_iterations,
        critic_tools=critic_tools,
    )
    pipeline = EPACPipeline(config=config)

    spec = None
    if args.spec:
        spec_data = json.loads(Path(args.spec).read_text())
        from epac.artifacts import EPACSpec
        spec = EPACSpec(**spec_data)

    result = await pipeline.run(
        goal=args.goal or (spec.goal if spec else ""),
        expert_id=args.expert_id,
        spec=spec,
    )

    result_dict = result.model_dump(mode="json")

    if args.output_file:
        Path(args.output_file).write_text(json.dumps(result_dict, indent=2))
        logger.info("Result written to %s", args.output_file)

        # Write separate audit log
        audit_path = Path(args.output_file).with_name("epac-audit.json")
        audit_path.write_text(json.dumps(result.audit_log, indent=2))
        logger.info("Audit log written to %s", audit_path)

    if result.completed:
        print(f"\n✓ EPAC pipeline completed after {result.actor_critic_iterations} Actor-Critic iteration(s)")
        if result.final_implementation and result.final_implementation.pr_url:
            print(f"  PR: {result.final_implementation.pr_url}")
    else:
        print(f"\n✗ EPAC pipeline failed: {result.failure_reason}", file=sys.stderr)
        sys.exit(1)


def _report(args: argparse.Namespace) -> None:
    result_data = json.loads(Path(args.result_file).read_text())

    print("\n── EPAC Pipeline Report ──────────────────────────────────────────")
    print(f"Pipeline ID : {result_data.get('pipeline_id', 'unknown')}")
    print(f"Status      : {'COMPLETED' if result_data.get('completed') else 'FAILED'}")
    print(f"Iterations  : {result_data.get('actor_critic_iterations', 0)}")

    if result_data.get("spec"):
        print(f"Spec        : {result_data['spec'].get('title', '')}")

    if result_data.get("reviews"):
        review = result_data["reviews"][-1]
        findings = review.get("findings", [])
        errors = sum(1 for f in findings if f.get("severity") == "error")
        warnings = sum(1 for f in findings if f.get("severity") == "warning")
        print(f"Findings    : {errors} errors, {warnings} warnings")

    if result_data.get("failure_reason"):
        print(f"Failure     : {result_data['failure_reason']}")

    print("──────────────────────────────────────────────────────────────────\n")


def _workflow(args: argparse.Namespace) -> None:
    from epac.integrations.github.workflow import generate_workflow_yaml
    print(generate_workflow_yaml(trigger=args.trigger))


if __name__ == "__main__":
    main()
