# Contributing to EPAC

Thank you for your interest in contributing. EPAC is an open framework — contributions of all kinds are welcome.

## What to contribute

- **New Critic tool integrations** (ESLint, Pylint, Snyk, Trivy, CodeQL, etc.)
- **New orchestration backends** (CrewAI adapter, OpenAI Agents SDK adapter, Temporal)
- **New Expert interfaces** (Slack approval bot, web UI, Jira integration)
- **New Actor integrations** (GitLab MR, Bitbucket PR, local filesystem)
- **Documentation and examples** (new use cases, domain-specific configurations)
- **Bug fixes and test coverage improvements**
- **Governance and compliance additions** (GDPR mapping, SOC 2 controls)

## Development setup

```bash
git clone https://github.com/thomason-io/epac.git
cd epac
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
```

## Running tests

```bash
pytest
```

## Code style

EPAC uses `ruff` for linting and formatting:

```bash
ruff check epac/ tests/
ruff format epac/ tests/
```

## Pull request guidelines

1. **Reference the pattern**: PRs that change core behavior (roles, artifact schemas, orchestration) should reference the relevant section of [`docs/pattern.md`](docs/pattern.md)
2. **Add tests**: new features need test coverage; bug fixes need regression tests
3. **Document**: update docstrings and README sections as needed
4. **Keep roles separate**: the Expert, Planner, Actor, and Critic must remain cleanly separated — resist the urge to shortcut by having one role call another directly
5. **Audit trail**: any new state transitions must produce an `AuditEntry`

## Adding a Critic tool integration

```python
# epac/integrations/my_tool.py

from epac.integrations import register_tool
from epac.artifacts.review import ReviewFinding, FindingSeverity, FindingCategory

@register_tool("my_tool")
async def run_my_tool(implementation, state) -> list[ReviewFinding]:
    """
    Run My Tool scanner over Actor-produced files.

    Requires: pip install my-tool-package
    """
    findings = []
    # ... run your tool, convert results to ReviewFinding objects
    return findings
```

Then add `"my_tool"` to `epac/integrations/__init__.py`'s built-in import and update the README.

## Design principles

- **Expert is always in control**: never auto-approve an approval gate without explicit configuration
- **Typed artifacts only**: no untyped string passing between roles
- **Audit everything**: every stage transition must produce an `AuditEntry`
- **Fail safe**: prefer failing loudly to silently degrading
- **Separation of concerns**: the Critic must never modify code; the Actor must never change the plan

## License

By contributing, you agree that your contributions will be licensed under the Apache 2.0 license.
