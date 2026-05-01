"""
Tests for epac spec init (spec_init.py).

Covers:
- LLM extraction output parsing and field mapping
- Interactive elicitation helpers (no actual LLM calls)
- Constraint category inference
- Red-line detection
- YAML serialisation round-trip
- run_spec_init end-to-end with --no-llm and mocked stdin
- CLI argument wiring
"""

from __future__ import annotations

import json
import sys
import uuid
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from epac.spec_init import (
    _extract_with_llm,
    _spec_to_yaml,
    run_spec_init,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_extraction(**overrides) -> dict:
    base = {
        "title": "Review CloudVault MSA",
        "goal": "Analyze the vendor MSA and flag risky clauses.",
        "background": "Annual contract value $600K.",
        "in_scope": ["§8 Liability", "§11 Data Processing"],
        "out_of_scope": ["Order forms"],
        "acceptance_criteria": [
            {
                "given": "the MSA contains a liability clause",
                "when": "analyzed",
                "then": "finding states cap and severity",
            }
        ],
        "constraints": [
            {
                "category": "legal-policy",
                "description": "Liability cap must be at least 12 months of fees.",
                "mandatory": True,
            }
        ],
        "risk_level": "high",
        "tags": ["vendor-review", "saas-msa"],
        "confidence": {
            "goal": "high",
            "constraints": "high",
            "acceptance_criteria": "medium",
            "risk_level": "high",
        },
    }
    base.update(overrides)
    return base


# ── YAML round-trip ────────────────────────────────────────────────────────────

class TestSpecToYaml:
    def test_produces_valid_yaml(self):
        spec = {
            "id": str(uuid.uuid4()),
            "title": "Test spec",
            "goal": "Do a thing",
            "risk_level": "medium",
            "created_by": "user@example.com",
        }
        raw = _spec_to_yaml(spec)
        parsed = yaml.safe_load(raw)
        assert parsed["title"] == "Test spec"
        assert parsed["risk_level"] == "medium"

    def test_header_comment_present(self):
        raw = _spec_to_yaml({"title": "x", "goal": "y", "created_by": "z"})
        assert "epac spec init" in raw
        assert "epac run --spec" in raw

    def test_long_strings_use_block_scalar(self):
        long_goal = "A" * 100
        raw = _spec_to_yaml({"goal": long_goal, "created_by": "x"})
        # Block scalar uses > style — yaml will re-join on parse
        parsed = yaml.safe_load(raw)
        assert parsed["goal"].replace("\n", "").replace(" ", "") == long_goal.replace(" ", "")

    def test_round_trip_preserves_lists(self):
        spec = {
            "in_scope": ["§2 Definitions", "§8 Liability"],
            "constraints": [
                {"category": "compliance", "description": "Must comply with GDPR", "mandatory": True}
            ],
            "created_by": "x",
        }
        parsed = yaml.safe_load(_spec_to_yaml(spec))
        assert parsed["in_scope"] == spec["in_scope"]
        assert parsed["constraints"][0]["category"] == "compliance"


# ── LLM extraction ─────────────────────────────────────────────────────────────

class TestExtractWithLlm:
    def test_returns_dict_on_success(self):
        mock_response = MagicMock()
        mock_response.choices[0].message.content = json.dumps(_make_extraction())

        with patch("litellm.completion", return_value=mock_response):
            result = _extract_with_llm("review this vendor contract", "anthropic/claude-haiku-4-5")

        assert result["title"] == "Review CloudVault MSA"
        assert result["risk_level"] == "high"
        # confidence is popped inside run_spec_init, not _extract_with_llm
        assert "confidence" in result

    def test_strips_markdown_fences(self):
        raw_with_fences = "```json\n" + json.dumps(_make_extraction()) + "\n```"
        mock_response = MagicMock()
        mock_response.choices[0].message.content = raw_with_fences

        with patch("litellm.completion", return_value=mock_response):
            result = _extract_with_llm("some prompt", "model")

        assert result["title"] == "Review CloudVault MSA"

    def test_returns_empty_dict_on_malformed_json(self, capsys):
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "not json at all"

        with patch("litellm.completion", return_value=mock_response):
            result = _extract_with_llm("prompt", "model")

        assert result == {}
        captured = capsys.readouterr()
        assert "malformed" in captured.out.lower() or "warning" in captured.out.lower()

    def test_returns_empty_dict_on_llm_exception(self, capsys):
        with patch("litellm.completion", side_effect=Exception("API error")):
            result = _extract_with_llm("prompt", "model")

        assert result == {}

    def test_missing_litellm_exits(self):
        with patch.dict("sys.modules", {"litellm": None}):
            with pytest.raises(SystemExit):
                _extract_with_llm("prompt", "model")


# ── run_spec_init end-to-end ───────────────────────────────────────────────────

class TestRunSpecInit:
    """End-to-end tests using --no-llm and mocked stdin."""

    def _make_inputs(self, *answers: str) -> str:
        """Join answers with newlines to simulate stdin."""
        return "\n".join(answers) + "\n"

    def test_creates_yaml_file(self, tmp_path):
        out = tmp_path / "spec.yaml"
        inputs = self._make_inputs(
            # Q1 expert id
            "alice@example.com",
            # Q2 title
            "Review vendor MSA",
            # Q3 goal
            "Analyze the vendor MSA for risky clauses",
            # Q4 background
            "",
            # Q5 in_scope
            "",
            # Q5 out_of_scope
            "",
            # Q6 risk level
            "high",
            # Q7 add acceptance criteria?
            "n",
            # Q8 add constraints?
            "n",
            # Q9 gate plan?
            "y",
            # Q9 gate final?
            "y",
        )
        with patch("builtins.input", side_effect=inputs.split("\n")):
            result_path = run_spec_init(
                prompt="Review this vendor MSA and flag risky clauses",
                output=str(out),
                no_llm=True,
            )

        assert result_path == out
        assert out.exists()
        parsed = yaml.safe_load(out.read_text())
        assert parsed["created_by"] == "alice@example.com"
        assert parsed["risk_level"] == "high"
        assert parsed["requires_expert_plan_approval"] is True
        assert parsed["requires_expert_final_approval"] is True

    def test_goal_is_written_to_file(self, tmp_path):
        out = tmp_path / "spec.yaml"
        inputs = self._make_inputs(
            "bob@example.com",
            "My task title",
            "Perform a detailed analysis of contract terms",
            "", "", "", "medium", "n", "n", "y", "y",
        )
        with patch("builtins.input", side_effect=inputs.split("\n")):
            run_spec_init(
                prompt="Analyze contract terms carefully",
                output=str(out),
                no_llm=True,
            )

        parsed = yaml.safe_load(out.read_text())
        assert "analysis" in parsed["goal"].lower() or "analyze" in parsed["goal"].lower()

    def test_low_risk_defaults_plan_gate_to_false(self, tmp_path):
        out = tmp_path / "spec.yaml"
        inputs = self._make_inputs(
            "carol@example.com",
            "Low risk task",
            "Just summarize this document",
            "", "", "", "low",
            "n", "n",
            # defaults: plan gate=n, final gate=y
            "", "",
        )
        with patch("builtins.input", side_effect=inputs.split("\n")):
            run_spec_init(
                prompt="Summarize this document",
                output=str(out),
                no_llm=True,
            )

        parsed = yaml.safe_load(out.read_text())
        assert parsed["risk_level"] == "low"
        # default for low risk: plan gate = n
        assert parsed["requires_expert_plan_approval"] is False

    def test_spec_id_is_valid_uuid(self, tmp_path):
        out = tmp_path / "spec.yaml"
        inputs = self._make_inputs(
            "dan@example.com", "Title", "Goal", "", "", "", "medium", "n", "n", "n", "y",
        )
        with patch("builtins.input", side_effect=inputs.split("\n")):
            run_spec_init(prompt="Do something", output=str(out), no_llm=True)

        parsed = yaml.safe_load(out.read_text())
        # Should not raise
        uuid.UUID(parsed["id"])

    def test_accepts_prompt_from_file(self, tmp_path):
        prompt_file = tmp_path / "prompt.txt"
        prompt_file.write_text("Review this contract for GDPR compliance issues")
        out = tmp_path / "spec.yaml"

        inputs = self._make_inputs(
            "eve@example.com", "GDPR Review", "Review contract for GDPR", "",
            "", "", "high", "n", "n", "y", "y",
        )
        with patch("builtins.input", side_effect=inputs.split("\n")):
            run_spec_init(
                prompt_file=str(prompt_file),
                output=str(out),
                no_llm=True,
            )

        assert out.exists()
        parsed = yaml.safe_load(out.read_text())
        assert parsed["created_by"] == "eve@example.com"

    def test_empty_prompt_exits(self):
        # When prompt is empty string AND interactive read returns nothing, should exit
        with patch("builtins.input", return_value=""):
            with pytest.raises(SystemExit):
                run_spec_init(prompt="", no_llm=True)

    def test_expert_id_from_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("EPAC_EXPERT_ID", "frank@example.com")
        out = tmp_path / "spec.yaml"
        # First input is expert id — accept the default (press enter)
        inputs = self._make_inputs(
            "",  # accept env default
            "Title", "Goal", "", "", "", "medium", "n", "n", "n", "y",
        )
        with patch("builtins.input", side_effect=inputs.split("\n")):
            run_spec_init(
                prompt="Do something",
                output=str(out),
                no_llm=True,
                expert_id="frank@example.com",
            )

        parsed = yaml.safe_load(out.read_text())
        assert parsed["created_by"] == "frank@example.com"

    def test_constraints_added_interactively(self, tmp_path):
        out = tmp_path / "spec.yaml"
        inputs = self._make_inputs(
            "grace@example.com",
            "Constrained task",
            "Do something with strict rules",
            "", "", "",
            "medium",
            "n",   # no acceptance criteria
            "y",   # yes add constraints
            "Must not modify the production database",
            "Must comply with GDPR",
            "",    # blank to finish constraints
            "n", "y",
        )
        with patch("builtins.input", side_effect=inputs.split("\n")):
            run_spec_init(
                prompt="Do something carefully",
                output=str(out),
                no_llm=True,
            )

        parsed = yaml.safe_load(out.read_text())
        assert len(parsed["constraints"]) == 2
        descs = [c["description"] for c in parsed["constraints"]]
        assert any("database" in d for d in descs)
        assert any("GDPR" in d for d in descs)

    def test_red_line_constraint_category(self, tmp_path):
        out = tmp_path / "spec.yaml"
        inputs = self._make_inputs(
            "henry@example.com",
            "High stakes task",
            "Do something important",
            "", "", "",
            "critical",
            "n",
            "y",
            "Must never delete customer data",
            "",
            "y", "y",
        )
        with patch("builtins.input", side_effect=inputs.split("\n")):
            run_spec_init(
                prompt="Never delete customer data",
                output=str(out),
                no_llm=True,
            )

        parsed = yaml.safe_load(out.read_text())
        categories = [c["category"] for c in parsed["constraints"]]
        assert "red-line" in categories


# ── CLI wiring ─────────────────────────────────────────────────────────────────

class TestCliWiring:
    def test_spec_init_subcommand_exists(self):
        """epac spec init --help should not raise SystemExit with error code."""
        with pytest.raises(SystemExit) as exc_info:
            sys.argv = ["epac", "spec", "init", "--help"]
            from epac.cli import main
            main()
        # --help exits with code 0
        assert exc_info.value.code == 0

    def test_spec_init_flags_parsed(self, tmp_path):
        """Flags --prompt, --output, --no-llm should be passed through correctly."""
        out = tmp_path / "wired.yaml"
        inputs = "\n".join([
            "wired@example.com",
            "Wired title",
            "Wired goal",
            "", "", "",
            "low",
            "n", "n",
            "n", "y",
        ]) + "\n"

        with patch("builtins.input", side_effect=inputs.split("\n")):
            sys.argv = [
                "epac", "spec", "init",
                "--prompt", "Review this contract",
                "--output", str(out),
                "--no-llm",
                "--expert-id", "wired@example.com",
            ]
            from epac.cli import main
            main()

        assert out.exists()
        parsed = yaml.safe_load(out.read_text())
        assert parsed["created_by"] == "wired@example.com"
