"""
EPAC GitHub Integration (optional extras).

Install with: pip install epac[github]

Provides:
  - GitHubActorTool   : Actor that works via GitHub PRs (branch + commit + PR)
  - GitHubCriticTool  : Critic integration that reads GitHub Code Scanning alerts
  - upload_sarif      : Upload Critic SARIF output to GitHub Code Scanning
  - EPACWorkflow      : Pre-built GitHub Actions workflow YAML generator

Environment variables:
  GITHUB_TOKEN          GitHub personal access token or Actions token
  GITHUB_REPOSITORY     owner/repo (e.g. acme/my-service)
  EPAC_BASE_BRANCH      Base branch for PRs (default: main)
"""

from epac.integrations.github.actor import GitHubActorTool
from epac.integrations.github.critic import GitHubCriticTool, upload_sarif
from epac.integrations.github.workflow import generate_workflow_yaml

__all__ = [
    "GitHubActorTool",
    "GitHubCriticTool",
    "upload_sarif",
    "generate_workflow_yaml",
]
