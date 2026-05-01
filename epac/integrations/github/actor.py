"""
GitHubActorTool – Actor integration that creates branches and pull requests.

Wraps the Actor's file changes into a real GitHub pull request, enabling
the Critic to use GitHub Code Scanning results and the Expert to review
a proper PR diff before approval.

Requires:
  pip install epac[github]  (PyGithub)

Environment:
  GITHUB_TOKEN        Required – PAT or Actions GITHUB_TOKEN
  GITHUB_REPOSITORY   Required – owner/repo
  EPAC_BASE_BRANCH    Optional – default 'main'
"""

from __future__ import annotations

import base64
import logging
import os
from typing import Any

from epac.artifacts.implementation import EPACImplementation

logger = logging.getLogger(__name__)


class GitHubActorTool:
    """
    Creates GitHub pull requests from EPACImplementation file changes.

    Usage in ActorRole subclass::

        class MyActorRole(ActorRole):
            def __init__(self, ...):
                super().__init__(...)
                self.github = GitHubActorTool()

            async def run(self, state):
                updates = await super().run(state)
                impl_data = updates.get("implementations", [])[-1]
                impl = EPACImplementation(**impl_data)
                pr_url, branch, sha = await self.github.push_implementation(impl, state)
                impl_data.update(pr_url=pr_url, branch_name=branch, commit_sha=sha)
                return updates
    """

    def __init__(
        self,
        token: str | None = None,
        repository: str | None = None,
        base_branch: str | None = None,
    ) -> None:
        self.token = token or os.environ.get("GITHUB_TOKEN", "")
        self.repository = repository or os.environ.get("GITHUB_REPOSITORY", "")
        self.base_branch = base_branch or os.environ.get("EPAC_BASE_BRANCH", "main")

        if not self.token or not self.repository:
            raise ValueError(
                "GitHubActorTool requires GITHUB_TOKEN and GITHUB_REPOSITORY env vars."
            )

    def _get_repo(self) -> Any:
        try:
            from github import Github  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "PyGithub is required for GitHub integration. "
                "Install with: pip install epac[github]"
            ) from exc
        return Github(self.token).get_repo(self.repository)

    async def push_implementation(
        self, implementation: EPACImplementation, state: Any
    ) -> tuple[str, str, str]:
        """
        Push implementation file changes to a new branch and open a PR.

        Returns
        -------
        tuple
            (pr_url, branch_name, commit_sha)
        """
        import asyncio

        return await asyncio.to_thread(self._push_sync, implementation, state)

    def _push_sync(
        self, implementation: EPACImplementation, state: Any
    ) -> tuple[str, str, str]:
        BLOCKED_PATHS = [".github/workflows", ".github/actions"]
        file_changes = implementation.file_changes
        for change in file_changes:
            for blocked in BLOCKED_PATHS:
                if change.path.startswith(blocked) or blocked in change.path:
                    raise ValueError(
                        f"Actor is prohibited from modifying '{change.path}'. "
                        f"CI workflow files require manual Expert review."
                    )

        repo = self._get_repo()

        spec_title = ""
        if hasattr(state, "spec") and state.spec:
            spec_title = getattr(state.spec, "title", "")
        elif isinstance(state, dict) and state.get("spec"):
            spec_title = state["spec"].get("title", "")

        branch_name = (
            f"epac/actor/{implementation.id[:8]}"
        )
        safe_title = spec_title[:40].lower().replace(" ", "-") if spec_title else "task"
        branch_name = f"epac/actor/{safe_title}-{implementation.id[:8]}"

        # Create branch from base
        base_sha = repo.get_branch(self.base_branch).commit.sha
        repo.create_git_ref(ref=f"refs/heads/{branch_name}", sha=base_sha)
        logger.info("[GitHubActor] Created branch %s", branch_name)

        # Push file changes
        last_sha = base_sha
        for change in implementation.file_changes:
            if change.action == "deleted":
                try:
                    contents = repo.get_contents(change.path, ref=branch_name)
                    repo.delete_file(
                        change.path,
                        f"epac: delete {change.path}",
                        contents.sha,  # type: ignore[attr-defined]
                        branch=branch_name,
                    )
                except Exception:  # noqa: BLE001
                    pass
            else:
                content = change.content or self._apply_diff(
                    repo, change.path, change.diff, branch_name
                )
                encoded = base64.b64encode(content.encode()).decode()
                try:
                    existing = repo.get_contents(change.path, ref=branch_name)
                    result = repo.update_file(
                        change.path,
                        f"epac: {change.action} {change.path}",
                        content,
                        existing.sha,  # type: ignore[attr-defined]
                        branch=branch_name,
                    )
                except Exception:
                    result = repo.create_file(
                        change.path,
                        f"epac: create {change.path}",
                        content,
                        branch=branch_name,
                    )
                last_sha = result["commit"].sha

        # Build PR body
        pr_body = self._build_pr_body(implementation, state)

        # Open PR
        pr = repo.create_pull(
            title=f"[EPAC] {spec_title or 'Implementation'} (iter {implementation.iteration})",
            body=pr_body,
            head=branch_name,
            base=self.base_branch,
        )
        logger.info("[GitHubActor] Opened PR #%d: %s", pr.number, pr.html_url)
        return pr.html_url, branch_name, last_sha

    def _apply_diff(self, repo: Any, path: str, diff: str, branch: str) -> str:
        """Naively apply added lines from a diff. For production, use patch(1)."""
        try:
            existing = repo.get_contents(path, ref=branch)
            original = existing.decoded_content.decode()  # type: ignore[attr-defined]
        except Exception:
            original = ""

        lines = original.splitlines()
        added: list[str] = []
        for line in diff.splitlines():
            if line.startswith("+") and not line.startswith("+++"):
                added.append(line[1:])
        return "\n".join(added) if added else original

    def _build_pr_body(self, implementation: EPACImplementation, state: Any) -> str:
        lines = [
            "## EPAC-Generated Pull Request",
            "",
            f"**Pipeline ID:** `{getattr(state, 'pipeline_id', 'unknown')}`",
            f"**Implementation ID:** `{implementation.id}`",
            f"**Actor-Critic Iteration:** {implementation.iteration}",
            "",
        ]
        if implementation.actor_notes:
            lines += ["### Actor Notes", implementation.actor_notes, ""]
        if implementation.known_issues:
            lines += ["### Known Issues"]
            for issue in implementation.known_issues:
                lines.append(f"- {issue}")
            lines.append("")
        if implementation.test_results:
            lines += ["### Test Results"]
            for tr in implementation.test_results:
                status = "✅" if tr.all_passed else "❌"
                lines.append(f"- {status} {tr.suite}: {tr.passed} passed, {tr.failed} failed")
        lines += [
            "",
            "---",
            "_This PR was generated by the [EPAC Framework](https://github.com/thomason-io/epac). "
            "Review the Critic findings before merging._",
        ]
        return "\n".join(lines)
