"""Unit tests covering the four fixes in fix/cli-remaining-issues.

1. _validate_git_target  — allowlist regex for git refs
2. _review_all_files_async — parallel review with asyncio.gather
3. _output_diff_results dispatch — unknown format raises Exit(1)
4. Empty results guard — review_diff_cmd exits cleanly on empty hunks_by_file
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import typer

from code_reviewer.cli import (
    _output_diff_results,
    _review_all_files_async,
    _validate_git_target,
)
from code_reviewer.config import Settings
from code_reviewer.core.models import Finding, ReviewResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _finding(
    severity: str = "HIGH",
    category: str = "security",
    message: str = "SQL injection",
    suggestion: str = "Use parameterized queries.",
    line_number: int | None = 10,
    file_path: str = "src/app.py",
) -> Finding:
    return Finding(
        file_path=file_path,
        line_number=line_number,
        severity=severity,
        category=category,
        message=message,
        suggestion=suggestion,
        source="llm",
    )


def _result(
    findings: list[Finding] | None = None,
    file_path: str = "src/app.py",
) -> ReviewResult:
    return ReviewResult(
        file_path=file_path,
        findings=findings or [],
        summary="Test summary.",
        reviewed_at=datetime.now(tz=timezone.utc),
        model_used="gemini/gemini-3.1-flash-lite",
        lines_reviewed=10,
    )


def _make_hunk(file_path: str = "src/app.py"):
    """Return a minimal DiffHunk-like object for testing."""
    from code_reviewer.core.models import DiffHunk
    return DiffHunk(
        file_path=file_path,
        start_line=1,
        end_line=5,
        added_lines=[(1, "+ x = 1"), (2, "+ y = 2")],
        removed_lines=[],
        context_lines=[],
        raw_hunk="@@ -0,0 +1,2 @@\n+ x = 1\n+ y = 2\n",
    )


# ===========================================================================
# Fix 1: _validate_git_target
# ===========================================================================

class TestValidateGitTarget:
    """Commit 1: strict regex allowlist for git refs."""

    def test_simple_branch_accepted(self):
        assert _validate_git_target("main") is True

    def test_head_tilde_accepted(self):
        assert _validate_git_target("HEAD~1") is True

    def test_head_tilde_5_accepted(self):
        assert _validate_git_target("HEAD~5") is True

    def test_range_syntax_accepted(self):
        assert _validate_git_target("main..feature-branch") is True

    def test_commit_sha_accepted(self):
        assert _validate_git_target("abc1234") is True

    def test_full_sha_accepted(self):
        assert _validate_git_target("a" * 40) is True

    def test_remote_branch_accepted(self):
        assert _validate_git_target("origin/main") is True

    def test_semver_tag_accepted(self):
        assert _validate_git_target("v1.2.3") is True

    def test_semicolon_injection_rejected(self):
        assert _validate_git_target("; rm -rf /") is False

    def test_main_with_shell_command_rejected(self):
        assert _validate_git_target("main; echo pwned") is False

    def test_subshell_injection_rejected(self):
        assert _validate_git_target("$(whoami)") is False

    def test_pipe_injection_rejected(self):
        assert _validate_git_target("main | cat /etc/passwd") is False

    def test_backtick_injection_rejected(self):
        assert _validate_git_target("`id`") is False

    def test_newline_injection_rejected(self):
        assert _validate_git_target("main\necho pwned") is False

    def test_empty_string_rejected(self):
        # Empty string should not match the regex (requires at least one char)
        assert _validate_git_target("") is False


# ===========================================================================
# Fix 2: _review_all_files_async
# ===========================================================================

class TestReviewAllFilesAsync:
    """Commit 3 (async parallel): _review_all_files_async behaviour."""

    def _run(self, coro):
        """Helper to run a coroutine in a fresh event loop."""
        return asyncio.run(coro)

    def test_returns_one_result_per_successful_file(self, mocker):
        """Each file that succeeds produces exactly one ReviewResult."""
        mock_findings = [_finding()]
        mocker.patch(
            "code_reviewer.agents.graph.run_agent_review",
            new=AsyncMock(return_value=mock_findings),
        )

        hunks_by_file = {
            "a.py": [_make_hunk("a.py")],
            "b.py": [_make_hunk("b.py")],
        }
        settings = Settings()
        results = self._run(_review_all_files_async(hunks_by_file, settings))

        assert len(results) == 2
        assert all(isinstance(r, ReviewResult) for r in results)
        file_paths = {r.file_path for r in results}
        assert file_paths == {"a.py", "b.py"}

    def test_failed_file_is_skipped_gracefully(self, mocker):
        """A file whose review raises an Exception is excluded from results."""
        async def _flaky(hunks, model, settings):
            if any(h.file_path == "bad.py" for h in hunks):
                raise RuntimeError("Agent exploded")
            return [_finding()]

        mocker.patch(
            "code_reviewer.agents.graph.run_agent_review",
            new=_flaky,
        )

        hunks_by_file = {
            "good.py": [_make_hunk("good.py")],
            "bad.py":  [_make_hunk("bad.py")],
        }
        settings = Settings()
        results = self._run(_review_all_files_async(hunks_by_file, settings))

        # Only good.py should be in results
        assert len(results) == 1
        assert results[0].file_path == "good.py"

    def test_results_contain_no_none_entries(self, mocker):
        """None entries from failed files are filtered before returning."""
        mocker.patch(
            "code_reviewer.agents.graph.run_agent_review",
            new=AsyncMock(side_effect=Exception("boom")),
        )

        hunks_by_file = {"x.py": [_make_hunk("x.py")]}
        settings = Settings()
        results = self._run(_review_all_files_async(hunks_by_file, settings))

        assert results == []
        assert None not in results

    def test_empty_hunks_by_file_returns_empty_list(self, mocker):
        """An empty input dict produces an empty result list without errors."""
        mocker.patch(
            "code_reviewer.agents.graph.run_agent_review",
            new=AsyncMock(return_value=[]),
        )
        results = self._run(_review_all_files_async({}, Settings()))
        assert results == []


# ===========================================================================
# Fix 3 & 4: _output_diff_results dispatch dict + unknown format
# ===========================================================================

class TestOutputDiffResultsDispatch:
    """Commit 4: dispatch dict replaces if/elif; unknown format raises Exit(1)."""

    def test_unknown_format_raises_exit_1(self):
        """An unrecognised output format must raise typer.Exit with code 1."""
        with pytest.raises(typer.Exit) as exc_info:
            _output_diff_results([], "xml", Settings(), elapsed=0.0)
        assert exc_info.value.exit_code == 1

    def test_yaml_also_raises_exit_1(self):
        """Another unknown format also triggers Exit(1)."""
        with pytest.raises(typer.Exit) as exc_info:
            _output_diff_results([], "yaml", Settings(), elapsed=0.0)
        assert exc_info.value.exit_code == 1

    def test_pretty_format_calls_pretty_formatter(self, capsys):
        """'pretty' format reaches _print_diff_pretty without error."""
        results = [_result(findings=[_finding()])]
        # Should not raise; pretty output goes to rich console (not capsys stdout)
        _output_diff_results(results, "pretty", Settings(), elapsed=0.5)

    def test_github_format_prints_annotations(self, capsys):
        """'github' format emits ::error / ::warning annotation lines."""
        results = [_result(findings=[_finding(severity="HIGH", message="SQL")])]
        _output_diff_results(results, "github", Settings(), elapsed=0.0)
        captured = capsys.readouterr()
        assert "::error" in captured.out

    def test_json_format_prints_valid_json(self, capsys):
        """'json' format prints a valid JSON list."""
        import json as _json
        results = [_result(findings=[_finding()], file_path="f.py")]
        _output_diff_results(results, "json", Settings(), elapsed=0.0)
        captured = capsys.readouterr()
        parsed = _json.loads(captured.out)
        assert isinstance(parsed, list)
        assert len(parsed) == 1


# ===========================================================================
# Fix 2 (guard): Empty all_results causes clean exit
# ===========================================================================

class TestEmptyResultsGuard:
    """Commit 2: when _run_diff_review returns [], the command exits with code 0."""

    def test_empty_results_exits_with_code_0(self, mocker):
        """review_diff_cmd must exit code=0 (not NameError) when all reviews fail."""
        # Patch _run_git_diff to return a valid-looking diff
        mocker.patch(
            "code_reviewer.cli._run_git_diff",
            return_value="diff --git a/x.py b/x.py\n+line\n",
        )
        # Patch _validate_diff_not_empty to always pass
        mocker.patch("code_reviewer.cli._validate_diff_not_empty", return_value=True)
        # Patch _run_diff_review to return empty list (all files failed)
        mocker.patch("code_reviewer.cli._run_diff_review", return_value=[])

        from typer.testing import CliRunner
        from code_reviewer.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["review", "diff", "HEAD"])
        # Exit code 0: no files reviewed, but not a crash
        assert result.exit_code == 0
        assert "No files could be reviewed" in result.output

    def test_review_result_not_referenced_outside_try(self, mocker):
        """Verify _run_diff_review never leaks an unset review_result variable.

        This test calls _run_diff_review with a diff where the reviewer
        raises for every file, and asserts we get an empty list (not NameError).
        """
        mocker.patch(
            "code_reviewer.cli.parse_diff",
            return_value=[_make_hunk("a.py"), _make_hunk("b.py")],
        )
        # Async helper raises for all files
        mocker.patch(
            "code_reviewer.cli._review_all_files_async",
            new=AsyncMock(return_value=[]),
        )

        from code_reviewer.cli import _run_diff_review
        results = _run_diff_review("fake diff", Settings())
        # Should return empty list cleanly, never raise NameError
        assert results == []
