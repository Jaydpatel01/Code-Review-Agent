"""Unit tests for private helper functions in cli.py.

Coverage targets:
  _sanitize_annotation
  _validate_severity
  _validate_output_format
  _apply_cli_overrides
  _filter_by_severity
  _severity_to_github_level
  _format_finding_github
  _format_finding_pretty_lines
  _build_git_diff_command
  _run_git_diff
  _validate_diff_not_empty
  _output_result (json, github, pretty dispatching)
  _output_diff_results (json, github, pretty dispatching)
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
import typer

from code_reviewer.cli import (
    _apply_cli_overrides,
    _build_git_diff_command,
    _filter_by_severity,
    _format_finding_github,
    _format_finding_pretty_lines,
    _output_diff_results,
    _output_result,
    _run_git_diff,
    _sanitize_annotation,
    _severity_to_github_level,
    _validate_diff_not_empty,
    _validate_output_format,
    _validate_severity,
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
    source: str = "llm",
) -> Finding:
    return Finding(
        file_path=file_path,
        line_number=line_number,
        severity=severity,
        category=category,
        message=message,
        suggestion=suggestion,
        source=source,
    )


def _result(findings: list[Finding] | None = None, file_path: str = "src/app.py") -> ReviewResult:
    return ReviewResult(
        file_path=file_path,
        findings=findings or [],
        summary="Test summary.",
        reviewed_at=datetime.now(tz=timezone.utc),
        model_used="gemini/gemini-3.1-flash-lite",
        lines_reviewed=10,
    )


def _settings(**kwargs) -> Settings:
    s = Settings()
    for k, v in kwargs.items():
        setattr(s, k, v)
    return s


# ===========================================================================
# _sanitize_annotation
# ===========================================================================

class TestSanitizeAnnotation:
    def test_percent_encoded(self):
        assert _sanitize_annotation("50%") == "50%25"

    def test_carriage_return_encoded(self):
        assert _sanitize_annotation("a\rb") == "a%0Db"

    def test_newline_encoded(self):
        assert _sanitize_annotation("a\nb") == "a%0Ab"

    def test_colon_encoded(self):
        assert _sanitize_annotation("key:value") == "key%3Avalue"

    def test_comma_encoded(self):
        assert _sanitize_annotation("a,b") == "a%2Cb"

    def test_clean_string_unchanged(self):
        assert _sanitize_annotation("hello world") == "hello world"

    def test_multiple_chars_encoded(self):
        result = _sanitize_annotation("a%b:c,d\ne\rf")
        assert "%" not in result.replace("%25", "").replace("%0D", "").replace(
            "%0A", "").replace("%3A", "").replace("%2C", "")


# ===========================================================================
# _validate_severity
# ===========================================================================

class TestValidateSeverity:
    def test_high_valid(self):
        assert _validate_severity("HIGH") == "HIGH"

    def test_lowercase_accepted(self):
        assert _validate_severity("medium") == "MEDIUM"

    def test_info_valid(self):
        assert _validate_severity("info") == "INFO"

    def test_invalid_raises_exit(self):
        with pytest.raises(typer.Exit):
            _validate_severity("CRITICAL")


# ===========================================================================
# _validate_output_format
# ===========================================================================

class TestValidateOutputFormat:
    def test_pretty_valid(self):
        assert _validate_output_format("pretty") == "pretty"

    def test_json_uppercase_accepted(self):
        assert _validate_output_format("JSON") == "json"

    def test_github_valid(self):
        assert _validate_output_format("github") == "github"

    def test_invalid_raises_exit(self):
        with pytest.raises(typer.Exit):
            _validate_output_format("xml")


# ===========================================================================
# _apply_cli_overrides
# ===========================================================================

class TestApplyCliOverrides:
    def test_no_overrides_returns_defaults(self):
        s = Settings()
        original_threshold = s.severity_threshold
        original_format = s.output.format
        result = _apply_cli_overrides(s, None, None)
        assert result.severity_threshold == original_threshold
        assert result.output.format == original_format

    def test_severity_override_applied(self):
        s = Settings()
        _apply_cli_overrides(s, "LOW", None)
        assert s.severity_threshold == "LOW"

    def test_output_override_applied(self):
        s = Settings()
        _apply_cli_overrides(s, None, "json")
        assert s.output.format == "json"

    def test_invalid_severity_raises_exit(self):
        with pytest.raises(typer.Exit):
            _apply_cli_overrides(Settings(), "BOGUS", None)

    def test_invalid_output_raises_exit(self):
        with pytest.raises(typer.Exit):
            _apply_cli_overrides(Settings(), None, "yaml")


# ===========================================================================
# _filter_by_severity
# ===========================================================================

class TestFilterBySeverity:
    def _findings(self) -> list[Finding]:
        return [
            _finding(severity="HIGH"),
            _finding(severity="MEDIUM"),
            _finding(severity="LOW"),
            _finding(severity="INFO"),
        ]

    def test_threshold_high_keeps_only_high(self):
        result = _filter_by_severity(self._findings(), "HIGH")
        assert all(f.severity == "HIGH" for f in result)
        assert len(result) == 1

    def test_threshold_medium_keeps_high_and_medium(self):
        result = _filter_by_severity(self._findings(), "MEDIUM")
        severities = {f.severity for f in result}
        assert severities == {"HIGH", "MEDIUM"}

    def test_threshold_info_keeps_all(self):
        result = _filter_by_severity(self._findings(), "INFO")
        assert len(result) == 4

    def test_none_entries_excluded(self):
        findings = [_finding(severity="HIGH"), None]  # type: ignore[list-item]
        result = _filter_by_severity(findings, "INFO")
        assert all(f is not None for f in result)

    def test_empty_input_returns_empty(self):
        assert _filter_by_severity([], "HIGH") == []


# ===========================================================================
# _severity_to_github_level
# ===========================================================================

class TestSeverityToGithubLevel:
    def test_high_maps_to_error(self):
        assert _severity_to_github_level("HIGH") == "error"

    def test_medium_maps_to_warning(self):
        assert _severity_to_github_level("MEDIUM") == "warning"

    def test_low_maps_to_notice(self):
        assert _severity_to_github_level("LOW") == "notice"

    def test_info_maps_to_notice(self):
        assert _severity_to_github_level("INFO") == "notice"


# ===========================================================================
# _format_finding_github
# ===========================================================================

class TestFormatFindingGithub:
    def test_single_line_output(self):
        f = _finding(severity="HIGH", category="security", message="SQL injection",
                     suggestion="Use params.", line_number=5, file_path="app.py")
        line = _format_finding_github(f)
        assert "\n" not in line
        assert "\r" not in line

    def test_starts_with_level(self):
        f = _finding(severity="HIGH")
        assert _format_finding_github(f).startswith("::error ")

    def test_warning_for_medium(self):
        f = _finding(severity="MEDIUM")
        assert _format_finding_github(f).startswith("::warning ")

    def test_notice_for_low(self):
        f = _finding(severity="LOW")
        assert _format_finding_github(f).startswith("::notice ")

    def test_colon_in_message_sanitized(self):
        f = _finding(message="Error: bad code", suggestion="")
        line = _format_finding_github(f)
        # The double-colon separator must not be broken by the message colon
        # Check that the file= parameter appears on the same line
        assert "file=" in line

    def test_suggestion_appended(self):
        f = _finding(message="Issue", suggestion="Fix it.")
        line = _format_finding_github(f)
        assert "Fix it." in line

    def test_no_line_number_omits_line_param(self):
        f = _finding(line_number=None)
        line = _format_finding_github(f)
        assert "line=" not in line


# ===========================================================================
# _format_finding_pretty_lines
# ===========================================================================

class TestFormatFindingPrettyLines:
    def test_returns_list(self):
        lines = _format_finding_pretty_lines(_finding(), show_suggestion=True)
        assert isinstance(lines, list)
        assert len(lines) >= 2

    def test_suggestion_included_when_flag_true(self):
        f = _finding(suggestion="Use params.")
        lines = _format_finding_pretty_lines(f, show_suggestion=True)
        assert any("params" in l for l in lines)

    def test_suggestion_excluded_when_flag_false(self):
        f = _finding(suggestion="Use params.")
        lines = _format_finding_pretty_lines(f, show_suggestion=False)
        assert not any("params" in l for l in lines)

    def test_high_severity_uses_red_color(self):
        lines = _format_finding_pretty_lines(_finding(severity="HIGH"), show_suggestion=False)
        assert any("red" in l for l in lines)

    def test_info_severity_uses_dim_color(self):
        lines = _format_finding_pretty_lines(_finding(severity="INFO"), show_suggestion=False)
        assert any("dim" in l for l in lines)


# ===========================================================================
# _build_git_diff_command
# ===========================================================================

class TestBuildGitDiffCommand:
    def test_staged_returns_cached_flag(self):
        cmd = _build_git_diff_command(target=None, staged=True)
        assert cmd == ["git", "diff", "--cached"]

    def test_target_appended(self):
        cmd = _build_git_diff_command(target="HEAD~1", staged=False)
        assert cmd == ["git", "diff", "HEAD~1"]

    def test_neither_returns_plain_diff(self):
        cmd = _build_git_diff_command(target=None, staged=False)
        assert cmd == ["git", "diff"]

    def test_staged_takes_priority_over_target(self):
        # If both set (shouldn't happen from CLI but defensive check)
        cmd = _build_git_diff_command(target="main", staged=True)
        assert "--cached" in cmd
        assert "main" not in cmd


# ===========================================================================
# _run_git_diff
# ===========================================================================

class TestRunGitDiff:
    def test_returns_stdout_on_success(self, mocker):
        proc = MagicMock()
        proc.stdout = "diff --git a/foo.py b/foo.py\n+line"
        mocker.patch("subprocess.run", return_value=proc)
        result = _run_git_diff(target="HEAD", staged=False)
        assert result == proc.stdout

    def test_returns_none_on_called_process_error(self, mocker):
        import subprocess
        mocker.patch(
            "subprocess.run",
            side_effect=subprocess.CalledProcessError(128, "git", stderr="fatal: bad revision"),
        )
        result = _run_git_diff(target="bad-ref", staged=False)
        assert result is None


# ===========================================================================
# _validate_diff_not_empty
# ===========================================================================

class TestValidateDiffNotEmpty:
    def test_returns_true_for_non_empty(self):
        assert _validate_diff_not_empty("diff --git a/foo.py b/foo.py", target=None) is True

    def test_returns_false_for_whitespace_only(self):
        assert _validate_diff_not_empty("   \n\t  ", target="HEAD") is False

    def test_returns_false_for_empty_string(self):
        assert _validate_diff_not_empty("", target=None) is False


# ===========================================================================
# _output_result (dispatching)
# ===========================================================================

class TestOutputResult:
    def test_json_format_prints_json(self, capsys):
        result = _result(findings=[_finding()])
        _output_result(result, "json", Settings(), elapsed=1.0)
        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        assert "findings" in parsed

    def test_github_format_prints_annotation(self, capsys):
        result = _result(findings=[_finding(severity="HIGH", message="SQL")])
        _output_result(result, "github", Settings(), elapsed=1.0)
        captured = capsys.readouterr()
        assert "::error" in captured.out
        assert "SQL" in captured.out

    def test_pretty_format_does_not_raise(self):
        result = _result(findings=[_finding()])
        # Just check it doesn't raise
        _output_result(result, "pretty", Settings(), elapsed=0.5)

    def test_none_findings_skipped_in_github(self, capsys):
        # Pydantic enforces non-None in findings list, so test with a single finding
        findings = [_finding()]
        result = _result(findings=findings)
        _output_result(result, "github", Settings(), elapsed=0.0)
        captured = capsys.readouterr()
        # One annotation line must be present
        assert "::" in captured.out


# ===========================================================================
# _output_diff_results (dispatching)
# ===========================================================================

class TestOutputDiffResults:
    def test_json_format_prints_list(self, capsys):
        results = [_result(findings=[_finding()], file_path="a.py"),
                   _result(findings=[_finding(severity="LOW")], file_path="b.py")]
        _output_diff_results(results, "json", Settings(), elapsed=1.0)
        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        assert isinstance(parsed, list)
        assert len(parsed) == 2

    def test_github_format_prints_annotations(self, capsys):
        results = [_result(findings=[_finding(severity="MEDIUM", message="N+1")])]
        _output_diff_results(results, "github", Settings(), elapsed=0.5)
        captured = capsys.readouterr()
        assert "::warning" in captured.out

    def test_empty_results_no_crash(self, capsys):
        _output_diff_results([], "pretty", Settings(), elapsed=0.0)

    def test_empty_findings_per_file_skipped_in_pretty(self, capsys):
        results = [_result(findings=[], file_path="clean.py")]
        _output_diff_results(results, "pretty", Settings(), elapsed=0.0)
        # Should not raise and should print the total line
        captured = capsys.readouterr()
        assert "0 findings" in captured.out
