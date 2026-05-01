"""Built-in Critic tool integrations."""

from __future__ import annotations

import asyncio
import json
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from epac.artifacts.review import FindingCategory, FindingSeverity, ReviewFinding
from epac.integrations import register_tool

logger = logging.getLogger(__name__)


@register_tool("bandit")
async def run_bandit(implementation: Any, state: Any) -> list[ReviewFinding]:
    """
    Run Bandit (Python security linter) over Actor-produced Python files.

    Requires: pip install bandit
    """
    findings: list[ReviewFinding] = []
    python_changes = [f for f in implementation.file_changes if f.path.endswith(".py")]
    if not python_changes:
        return findings

    with tempfile.TemporaryDirectory() as tmpdir:
        for change in python_changes:
            content = change.content or _diff_to_content(change.diff)
            if not content:
                continue
            fpath = Path(tmpdir) / change.path
            fpath.parent.mkdir(parents=True, exist_ok=True)
            fpath.write_text(content)

        try:
            result = await asyncio.to_thread(
                subprocess.run,
                ["bandit", "-r", "-f", "json", tmpdir],
                capture_output=True,
                text=True,
                timeout=60,
            )
            if result.stdout:
                data = json.loads(result.stdout)
                for issue in data.get("results", []):
                    severity_map = {"HIGH": FindingSeverity.ERROR, "MEDIUM": FindingSeverity.WARNING, "LOW": FindingSeverity.NOTE}
                    findings.append(
                        ReviewFinding(
                            rule_id=f"bandit/{issue.get('test_id', 'unknown')}",
                            severity=severity_map.get(issue.get("issue_severity", "LOW"), FindingSeverity.NOTE),
                            category=FindingCategory.SECURITY,
                            message=issue.get("issue_text", ""),
                            file_path=issue.get("filename", ""),
                            line_start=issue.get("line_number"),
                            references=[issue.get("more_info", "")],
                        )
                    )
        except FileNotFoundError:
            logger.info("[bandit] bandit not installed — skipping")
        except Exception as exc:  # noqa: BLE001
            logger.warning("[bandit] Error: %s", exc)

    return findings


@register_tool("semgrep")
async def run_semgrep(implementation: Any, state: Any) -> list[ReviewFinding]:
    """
    Run Semgrep SAST over Actor-produced code.

    Requires: pip install semgrep or brew install semgrep
    Uses the auto-selected ruleset by default; override with EPAC_SEMGREP_CONFIG env.
    """
    import os
    findings: list[ReviewFinding] = []

    with tempfile.TemporaryDirectory() as tmpdir:
        for change in implementation.file_changes:
            content = change.content or _diff_to_content(change.diff)
            if not content:
                continue
            fpath = Path(tmpdir) / change.path
            fpath.parent.mkdir(parents=True, exist_ok=True)
            fpath.write_text(content)

        config = os.environ.get("EPAC_SEMGREP_CONFIG", "auto")
        try:
            result = await asyncio.to_thread(
                subprocess.run,
                ["semgrep", "--config", config, "--json", str(tmpdir)],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.stdout:
                data = json.loads(result.stdout)
                for match in data.get("results", []):
                    extra = match.get("extra", {})
                    severity_str = extra.get("severity", "WARNING").upper()
                    sev_map = {"ERROR": FindingSeverity.ERROR, "WARNING": FindingSeverity.WARNING, "INFO": FindingSeverity.NOTE}
                    findings.append(
                        ReviewFinding(
                            rule_id=f"semgrep/{match.get('check_id', 'unknown')}",
                            severity=sev_map.get(severity_str, FindingSeverity.WARNING),
                            category=FindingCategory.SECURITY,
                            message=extra.get("message", ""),
                            file_path=match.get("path", ""),
                            line_start=match.get("start", {}).get("line"),
                            line_end=match.get("end", {}).get("line"),
                        )
                    )
        except FileNotFoundError:
            logger.info("[semgrep] semgrep not installed — skipping")
        except Exception as exc:  # noqa: BLE001
            logger.warning("[semgrep] Error: %s", exc)

    return findings


@register_tool("sarif")
async def load_sarif_file(implementation: Any, state: Any) -> list[ReviewFinding]:
    """
    Load pre-existing SARIF output from a scan tool (e.g. CodeQL, Snyk).

    Expects the path in state.metadata["sarif_path"] or implementation.metadata["sarif_path"].
    """
    sarif_path = (
        getattr(state, "metadata", {}).get("sarif_path")
        or getattr(implementation, "metadata", {}).get("sarif_path")
    )
    if not sarif_path:
        return []

    findings: list[ReviewFinding] = []
    try:
        data = json.loads(Path(sarif_path).read_text())
        for run in data.get("runs", []):
            for result in run.get("results", []):
                level = result.get("level", "warning")
                sev_map = {"error": FindingSeverity.ERROR, "warning": FindingSeverity.WARNING, "note": FindingSeverity.NOTE, "none": FindingSeverity.NONE}
                loc = result.get("locations", [{}])[0]
                phys = loc.get("physicalLocation", {})
                findings.append(
                    ReviewFinding(
                        rule_id=result.get("ruleId", "unknown"),
                        severity=sev_map.get(level, FindingSeverity.WARNING),
                        category=FindingCategory.SECURITY,
                        message=result.get("message", {}).get("text", ""),
                        file_path=phys.get("artifactLocation", {}).get("uri", ""),
                        line_start=phys.get("region", {}).get("startLine"),
                    )
                )
    except Exception as exc:  # noqa: BLE001
        logger.warning("[sarif] Could not load SARIF file %s: %s", sarif_path, exc)

    return findings


def _diff_to_content(diff: str) -> str:
    """Extract added lines from a unified diff."""
    lines = []
    for line in diff.splitlines():
        if line.startswith("+") and not line.startswith("+++"):
            lines.append(line[1:])
    return "\n".join(lines)
