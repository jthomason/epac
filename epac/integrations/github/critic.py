"""
GitHubCriticTool – Critic integration for GitHub Code Scanning and Actions.

Reads existing Code Scanning alerts (from CodeQL, Dependabot, Snyk, etc.)
for the PR branch and merges them into the Critic review.

Also provides upload_sarif() to publish the Critic's SARIF output back to
GitHub Code Scanning for the PR.
"""

from __future__ import annotations

import asyncio
import base64
import gzip
import json
import logging
import os
from typing import Any

from epac.artifacts.implementation import EPACImplementation
from epac.artifacts.review import FindingCategory, FindingSeverity, ReviewFinding
from epac.integrations import register_tool

logger = logging.getLogger(__name__)


@register_tool("github_code_scanning")
async def run_github_code_scanning(
    implementation: EPACImplementation, state: Any
) -> list[ReviewFinding]:
    """
    Read GitHub Code Scanning alerts for the implementation's PR branch.

    Registered as a built-in Critic tool: add "github_code_scanning" to
    EPACConfig.critic_tools to activate.
    """
    token = os.environ.get("GITHUB_TOKEN", "")
    repository = os.environ.get("GITHUB_REPOSITORY", "")

    if not token or not repository or not implementation.branch_name:
        return []

    return await asyncio.to_thread(
        _fetch_code_scanning_alerts, token, repository, implementation.branch_name
    )


def _fetch_code_scanning_alerts(
    token: str, repository: str, branch: str
) -> list[ReviewFinding]:
    try:
        from github import Github  # type: ignore[import]
    except ImportError:
        logger.warning("[GitHubCritic] PyGithub not installed — skipping")
        return []

    g = Github(token)
    repo = g.get_repo(repository)
    findings: list[ReviewFinding] = []

    try:
        alerts = repo.get_codescan_alerts()
        for alert in alerts:
            if alert.state != "open":
                continue
            # Filter to alerts on this branch if possible
            rule = alert.rule
            location = alert.most_recent_instance
            severity_str = getattr(rule, "severity", "warning").lower()
            sev_map = {
                "critical": FindingSeverity.ERROR,
                "high": FindingSeverity.ERROR,
                "medium": FindingSeverity.WARNING,
                "low": FindingSeverity.NOTE,
                "warning": FindingSeverity.WARNING,
                "error": FindingSeverity.ERROR,
                "note": FindingSeverity.NOTE,
            }
            findings.append(
                ReviewFinding(
                    rule_id=f"github/{getattr(rule, 'id', 'unknown')}",
                    severity=sev_map.get(severity_str, FindingSeverity.WARNING),
                    category=FindingCategory.SECURITY,
                    message=getattr(rule, "description", ""),
                    file_path=getattr(location, "path", ""),
                    line_start=getattr(location, "start_line", None),
                    line_end=getattr(location, "end_line", None),
                    references=[alert.html_url],
                )
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning("[GitHubCritic] Could not fetch Code Scanning alerts: %s", exc)

    return findings


async def upload_sarif(
    sarif_data: dict[str, Any],
    ref: str,
    commit_sha: str,
    token: str | None = None,
    repository: str | None = None,
) -> str:
    """
    Upload SARIF output to GitHub Code Scanning.

    Parameters
    ----------
    sarif_data:
        SARIF dict (from EPACReview.sarif.to_sarif_dict())
    ref:
        Git ref (e.g. "refs/heads/epac/actor/my-branch")
    commit_sha:
        SHA of the commit being scanned
    token:
        GitHub token (defaults to GITHUB_TOKEN env var)
    repository:
        owner/repo (defaults to GITHUB_REPOSITORY env var)

    Returns
    -------
    str
        URL of the uploaded SARIF analysis
    """
    import aiohttp  # type: ignore[import]

    _token = token or os.environ.get("GITHUB_TOKEN", "")
    _repo = repository or os.environ.get("GITHUB_REPOSITORY", "")

    if not _token or not _repo:
        raise ValueError("GITHUB_TOKEN and GITHUB_REPOSITORY must be set for SARIF upload.")

    # Compress and encode
    sarif_json = json.dumps(sarif_data).encode()
    compressed = gzip.compress(sarif_json)
    sarif_b64 = base64.b64encode(compressed).decode()

    payload = {
        "commit_sha": commit_sha,
        "ref": ref,
        "sarif": sarif_b64,
        "tool_name": "epac-critic",
    }

    url = f"https://api.github.com/repos/{_repo}/code-scanning/sarifs"
    headers = {
        "Authorization": f"Bearer {_token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload, headers=headers) as resp:
            resp.raise_for_status()
            data = await resp.json()
            analysis_url = data.get("url", "")
            logger.info("[GitHubCritic] Uploaded SARIF: %s", analysis_url)
            return analysis_url


class GitHubCriticTool:
    """
    Helper class that bundles GitHub Critic integrations.

    Usage::

        from epac.integrations.github import GitHubCriticTool
        tool = GitHubCriticTool()
        findings = await tool.get_alerts(implementation)
        await tool.upload_review_sarif(review, implementation)
    """

    async def get_alerts(self, implementation: EPACImplementation) -> list[ReviewFinding]:
        return await run_github_code_scanning(implementation, None)

    async def upload_review_sarif(
        self, review: Any, implementation: EPACImplementation
    ) -> str:
        """Upload the Critic's SARIF to GitHub Code Scanning for the PR."""
        sarif_data = review.sarif.to_sarif_dict()
        ref = f"refs/heads/{implementation.branch_name}"
        return await upload_sarif(sarif_data, ref=ref, commit_sha=implementation.commit_sha)
