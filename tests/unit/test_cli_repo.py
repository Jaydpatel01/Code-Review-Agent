"""Unit tests for repository review CLI command helpers and commands in cli.py."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import typer
from typer.testing import CliRunner

from code_reviewer.cli import (
    _determine_repo_llm_targets,
    _parse_and_validate_repo_inputs,
    _run_repo_ast_pass,
    _run_repo_llm_pass_async,
    _run_repo_review,
    app,
)
from code_reviewer.config import Settings
from code_reviewer.core.models import Finding, ReviewResult
from code_reviewer.indexer.file_walker import FileWalker


# ---------------------------------------------------------------------------
# Test Data Helpers
# ---------------------------------------------------------------------------

def _make_finding(severity: str = "MEDIUM", file_path: str = "src/app.py") -> Finding:
    return Finding(
        file_path=file_path,
        line_number=5,
        severity=severity,
        category="complexity",
        message="Too nested",
        suggestion="Simplify",
        source="ast",
    )


# ---------------------------------------------------------------------------
# Test Cases
# ---------------------------------------------------------------------------

class TestDetermineRepoLLMTargets:
    def test_static_only_returns_empty_list(self):
        targets = _determine_repo_llm_targets(
            mode="static-only",
            risk_tiers={"a.py": "HIGH", "b.py": "LOW"},
            all_files=["a.py", "b.py"],
        )
        assert targets == []

    def test_smart_returns_only_high_and_medium(self):
        targets = _determine_repo_llm_targets(
            mode="smart",
            risk_tiers={"a.py": "HIGH", "b.py": "MEDIUM", "c.py": "LOW"},
            all_files=["a.py", "b.py", "c.py"],
        )
        assert set(targets) == {"a.py", "b.py"}

    def test_thorough_returns_all_files(self):
        targets = _determine_repo_llm_targets(
            mode="thorough",
            risk_tiers={"a.py": "HIGH", "b.py": "LOW"},
            all_files=["a.py", "b.py"],
        )
        assert set(targets) == {"a.py", "b.py"}


class TestParseAndValidateRepoInputs:
    def test_invalid_mode_raises_exit(self):
        with pytest.raises(typer.Exit) as exc:
            _parse_and_validate_repo_inputs(
                path=".",
                mode="invalid-mode",
                include="*.py",
                exclude="",
                severity=None,
                output=None,
            )
        assert exc.value.exit_code == 1

    def test_nonexistent_directory_raises_exit(self, tmp_path):
        nonexistent = tmp_path / "does_not_exist"
        with pytest.raises(typer.Exit) as exc:
            _parse_and_validate_repo_inputs(
                path=str(nonexistent),
                mode="smart",
                include="*.py",
                exclude="",
                severity=None,
                output=None,
            )
        assert exc.value.exit_code == 1

    def test_valid_inputs_are_parsed(self, tmp_path):
        root, mode, include, exclude, settings = _parse_and_validate_repo_inputs(
            path=str(tmp_path),
            mode="smart",
            include="*.py,*.js",
            exclude="node_modules,build",
            severity="HIGH",
            output="json",
        )
        assert root == tmp_path.resolve()
        assert mode == "smart"
        assert include == ["*.py", "*.js"]
        assert exclude == ["node_modules", "build"]
        assert settings.severity_threshold == "HIGH"
        assert settings.output.format == "json"


class TestRunRepoASTPass:
    @patch("code_reviewer.indexer.risk_scorer.score_file_risk")
    def test_ast_pass_calls_risk_scorer_for_each_file(self, mock_scorer, tmp_path):
        f1 = tmp_path / "f1.py"
        f1.touch()
        f2 = tmp_path / "f2.py"
        f2.touch()

        mock_scorer.side_effect = [
            ("HIGH", [_make_finding("HIGH", str(f1))]),
            ("LOW", []),
        ]

        walker = FileWalker(tmp_path, max_files=10)
        settings = Settings()

        findings, risk_tiers = _run_repo_ast_pass(walker, settings)

        assert str(f1) in findings
        assert str(f2) in findings
        assert risk_tiers[str(f1)] == "HIGH"
        assert risk_tiers[str(f2)] == "LOW"
        assert len(findings[str(f1)]) == 1
        assert len(findings[str(f2)]) == 0


class TestRunRepoLLMPassAsync:
    @pytest.mark.anyio
    @patch("code_reviewer.core.reviewer.FileReviewer.review_file")
    async def test_llm_pass_runs_concurrently(self, mock_review_file):
        # Setup mock return values
        mock_review_file.side_effect = lambda fp, **kwargs: ReviewResult(
            file_path=fp,
            findings=[_make_finding("MEDIUM", fp)],
            summary="Checked",
            model_used="gemini-mock",
            lines_reviewed=10,
        )

        settings = Settings()
        target_files = ["f1.py", "f2.py"]
        root = Path(".")

        results = await _run_repo_llm_pass_async(target_files, settings, root)

        assert len(results) == 2
        assert "f1.py" in results
        assert "f2.py" in results
        assert len(results["f1.py"].findings) == 1
        assert results["f1.py"].findings[0].file_path == "f1.py"

class TestRunRepoReview:
    @patch("code_reviewer.cli._run_repo_ast_pass")
    @patch("code_reviewer.cli._run_repo_llm_pass_async")
    def test_run_repo_review_static_only(self, mock_llm_pass, mock_ast_pass, tmp_path):
        f1 = str(tmp_path / "f1.py")
        mock_ast_pass.return_value = ({f1: [_make_finding("HIGH", f1)]}, {f1: "HIGH"})

        settings = Settings()
        all_findings = _run_repo_review(
            root=tmp_path,
            mode="static-only",
            include_patterns=["*.py"],
            exclude_patterns=[],
            settings=settings,
            max_files=10,
        )

        assert f1 in all_findings
        assert len(all_findings[f1]) == 1
        mock_llm_pass.assert_not_called()

    @patch("code_reviewer.cli._run_repo_ast_pass")
    @patch("code_reviewer.cli._run_repo_llm_pass_async")
    def test_run_repo_review_smart_mode(self, mock_llm_pass, mock_ast_pass, tmp_path):
        f1 = str(tmp_path / "f1.py")  # HIGH
        f2 = str(tmp_path / "f2.py")  # LOW
        mock_ast_pass.return_value = ({f1: [], f2: []}, {f1: "HIGH", f2: "LOW"})

        # Setup LLM mock response
        mock_llm_pass.return_value = {
            f1: ReviewResult(
                file_path=f1,
                findings=[_make_finding("MEDIUM", f1)],
                summary="Checked",
                model_used="gemini-mock",
                lines_reviewed=10,
            )
        }

        settings = Settings()
        all_findings = _run_repo_review(
            root=tmp_path,
            mode="smart",
            include_patterns=["*.py"],
            exclude_patterns=[],
            settings=settings,
            max_files=10,
        )

        # Smart mode must only request LLM review on f1.py (HIGH)
        mock_llm_pass.assert_called_once_with([f1], settings, tmp_path)
        assert len(all_findings[f1]) == 1
        assert len(all_findings[f2]) == 0

class TestRepoCliCommand:
    @patch("code_reviewer.cli._run_repo_review")
    def test_cli_runner_invokes_repo_command(self, mock_repo_review, tmp_path):
        f1 = str(tmp_path / "f1.py")
        mock_repo_review.return_value = {f1: [_make_finding("HIGH", f1)]}

        runner = CliRunner()
        result = runner.invoke(app, ["review", "repo", str(tmp_path), "--mode", "static-only", "--output", "json"])

        assert result.exit_code == 0
        assert "f1.py" in result.output
        mock_repo_review.assert_called_once()
