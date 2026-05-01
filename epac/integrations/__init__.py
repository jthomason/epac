"""
EPAC Integration Registry.

Tools registered here can be referenced by name in CriticRole.config.tools.
Each tool receives an EPACImplementation and EPACState and returns a list of ReviewFindings.

Built-in integrations:
  - "sarif"    : Parse SARIF files from a previous scan run
  - "bandit"   : Python security linter (requires bandit)
  - "semgrep"  : Semgrep SAST (requires semgrep CLI)
  - "eslint"   : JavaScript/TypeScript linter (requires eslint)

GitHub extras (requires epac[github]):
  - "github_code_scanning" : Read existing GitHub Code Scanning alerts via API
  - "github_pr_checks"     : Read GitHub Actions check results from a PR

Register custom tools::

    from epac.integrations import register_tool

    @register_tool("my_custom_scanner")
    async def run_my_scanner(implementation, state):
        ...
        return findings  # list[ReviewFinding]
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Awaitable

from epac.artifacts.review import ReviewFinding

logger = logging.getLogger(__name__)

# Tool registry: name → async callable
_TOOL_REGISTRY: dict[str, Callable[..., Awaitable[list[ReviewFinding]]]] = {}


def register_tool(name: str) -> Callable:
    """Decorator to register a Critic tool integration."""

    def decorator(fn: Callable) -> Callable:
        _TOOL_REGISTRY[name] = fn
        return fn

    return decorator


async def run_tool(
    tool_name: str, implementation: Any, state: Any
) -> list[ReviewFinding]:
    """Dispatch to a registered Critic tool integration."""
    if tool_name not in _TOOL_REGISTRY:
        logger.warning("[Integrations] Unknown tool '%s' — skipping", tool_name)
        return []
    return await _TOOL_REGISTRY[tool_name](implementation, state)


# Register built-in integrations
from epac.integrations import _builtin  # noqa: E402, F401 (side-effect import)
