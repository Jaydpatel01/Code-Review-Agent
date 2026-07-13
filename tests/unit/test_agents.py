"""Unit tests for the five specialised review agents.

Tests per agent:
  1. Valid JSON response parses into correct Finding objects
     (source="llm", correct category).
  2. Malformed JSON returns [] and does not raise.
  3. A finding whose line_number is not in any hunk's added_lines is rejected.
  4. __call__ returns the correct AgentState key.

All litellm.completion() calls are mocked — no real API calls are made.
Hunk input comes from tests/fixtures/sample_diffs/simple_change.diff,
parsed with the project's own DiffParser.
"""

from __future__ import annotations

import json
import os
import pytest
from unittest.mock import MagicMock

from code_reviewer.analyzers.diff_parser import parse_diff
from code_reviewer.agents.state import AgentState
from code_reviewer.core.models import DiffHunk, Finding

# ---------------------------------------------------------------------------
# Load the real fixture diff once and parse into hunks
# ---------------------------------------------------------------------------

DIFFS_DIR = os.path.join(
    os.path.dirname(__file__), "..", "fixtures", "sample_diffs"
)


def _load_hunks(filename: str) -> list[DiffHunk]:
    """Parse a .diff fixture file into DiffHunk objects."""
    with open(os.path.join(DIFFS_DIR, filename), encoding="utf-8") as f:
        raw = f.read()
    return parse_diff(raw)


# Parse simple_change.diff once at module level.
SIMPLE_HUNKS = _load_hunks("simple_change.diff")
# simple_change.diff adds lines 11 and 12 in app.py
ADDED_LINE_NUMS = {ln for h in SIMPLE_HUNKS for ln, _ in h.added_lines}


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_state(hunks: list[DiffHunk] | None = None) -> AgentState:
    return AgentState(
        hunks=hunks if hunks is not None else SIMPLE_HUNKS,
        model="gemini/gemini-3.1-flash-lite",
        context_lines_count=5,
        security_findings=[],
        performance_findings=[],
        logic_findings=[],
        style_findings=[],
        docs_findings=[],
        final_findings=[],
        error=None,
    )


def _mock_llm(mocker, findings_payload: list[dict]) -> None:
    """Patch litellm.completion to return the given findings payload."""
    resp = MagicMock()
    resp.choices[0].message.content = json.dumps({"findings": findings_payload})
    mocker.patch("litellm.completion", return_value=resp)


def _finding_payload(line_number: int, severity: str, category: str) -> dict:
    return {
        "line_number": line_number,
        "severity": severity,
        "category": category,
        "message": f"Test {category} finding",
        "suggestion": "Fix it.",
    }


# First valid added line in simple_change.diff
_VALID_LINE = next(iter(ADDED_LINE_NUMS))


# ===========================================================================
# SecurityAgent
# ===========================================================================

class TestSecurityAgentParsing:
    @pytest.fixture
    def agent(self):
        from code_reviewer.agents.security import SecurityAgent
        return SecurityAgent()

    def test_valid_json_produces_correct_finding(self, agent, mocker):
        """Valid JSON with a line in added_lines → one Finding with source=llm."""
        _mock_llm(mocker, [_finding_payload(_VALID_LINE, "HIGH", "security")])
        result = agent(_make_state())
        findings = result["security_findings"]
        assert len(findings) == 1
        assert findings[0].source == "llm"
        assert findings[0].category == "security"
        assert findings[0].severity == "HIGH"
        assert findings[0].line_number == _VALID_LINE

    def test_malformed_json_returns_empty(self, agent, mocker):
        """Garbage LLM output must not raise — returns empty list."""
        resp = MagicMock()
        resp.choices[0].message.content = "I cannot review this."
        mocker.patch("litellm.completion", return_value=resp)
        result = agent(_make_state())
        assert result["security_findings"] == []

    def test_hallucinated_line_rejected(self, agent, mocker):
        """line_number not in any hunk's added_lines must be dropped."""
        _mock_llm(mocker, [_finding_payload(9999, "HIGH", "security")])
        result = agent(_make_state())
        assert result["security_findings"] == []

    def test_call_returns_correct_state_key(self, agent, mocker):
        _mock_llm(mocker, [])
        result = agent(_make_state())
        assert "security_findings" in result
        assert isinstance(result["security_findings"], list)


# ===========================================================================
# PerformanceAgent
# ===========================================================================

class TestPerformanceAgentParsing:
    @pytest.fixture
    def agent(self):
        from code_reviewer.agents.performance import PerformanceAgent
        return PerformanceAgent()

    def test_valid_json_produces_correct_finding(self, agent, mocker):
        _mock_llm(mocker, [_finding_payload(_VALID_LINE, "MEDIUM", "performance")])
        result = agent(_make_state())
        findings = result["performance_findings"]
        assert len(findings) == 1
        assert findings[0].source == "llm"
        assert findings[0].category == "performance"

    def test_malformed_json_returns_empty(self, agent, mocker):
        resp = MagicMock()
        resp.choices[0].message.content = "{broken json"
        mocker.patch("litellm.completion", return_value=resp)
        result = agent(_make_state())
        assert result["performance_findings"] == []

    def test_hallucinated_line_rejected(self, agent, mocker):
        _mock_llm(mocker, [_finding_payload(9999, "MEDIUM", "performance")])
        result = agent(_make_state())
        assert result["performance_findings"] == []

    def test_call_returns_correct_state_key(self, agent, mocker):
        _mock_llm(mocker, [])
        result = agent(_make_state())
        assert "performance_findings" in result


# ===========================================================================
# LogicAgent
# ===========================================================================

class TestLogicAgentParsing:
    @pytest.fixture
    def agent(self):
        from code_reviewer.agents.logic import LogicAgent
        return LogicAgent()

    def test_valid_json_produces_correct_finding(self, agent, mocker):
        _mock_llm(mocker, [_finding_payload(_VALID_LINE, "HIGH", "logic")])
        result = agent(_make_state())
        findings = result["logic_findings"]
        assert len(findings) == 1
        assert findings[0].source == "llm"
        assert findings[0].category == "logic"

    def test_malformed_json_returns_empty(self, agent, mocker):
        resp = MagicMock()
        resp.choices[0].message.content = "not json at all"
        mocker.patch("litellm.completion", return_value=resp)
        result = agent(_make_state())
        assert result["logic_findings"] == []

    def test_hallucinated_line_rejected(self, agent, mocker):
        _mock_llm(mocker, [_finding_payload(0, "MEDIUM", "logic")])
        result = agent(_make_state())
        assert result["logic_findings"] == []

    def test_call_returns_correct_state_key(self, agent, mocker):
        _mock_llm(mocker, [])
        result = agent(_make_state())
        assert "logic_findings" in result

    def test_category_override_applied(self, agent, mocker):
        """LLM returning the wrong category is overridden to 'logic'."""
        _mock_llm(mocker, [_finding_payload(_VALID_LINE, "MEDIUM", "style")])
        result = agent(_make_state())
        findings = result["logic_findings"]
        assert len(findings) == 1
        assert findings[0].category == "logic"


# ===========================================================================
# StyleAgent
# ===========================================================================

class TestStyleAgentParsing:
    @pytest.fixture
    def agent(self):
        from code_reviewer.agents.style import StyleAgent
        return StyleAgent()

    def test_valid_json_produces_correct_finding(self, agent, mocker):
        _mock_llm(mocker, [_finding_payload(_VALID_LINE, "LOW", "style")])
        result = agent(_make_state())
        findings = result["style_findings"]
        assert len(findings) == 1
        assert findings[0].source == "llm"
        assert findings[0].category == "style"

    def test_malformed_json_returns_empty(self, agent, mocker):
        resp = MagicMock()
        resp.choices[0].message.content = ""
        mocker.patch("litellm.completion", return_value=resp)
        result = agent(_make_state())
        assert result["style_findings"] == []

    def test_hallucinated_line_rejected(self, agent, mocker):
        _mock_llm(mocker, [_finding_payload(1, "LOW", "style")])
        result = agent(_make_state())
        # Line 1 is not in simple_change.diff added lines (11, 12)
        assert result["style_findings"] == []

    def test_call_returns_correct_state_key(self, agent, mocker):
        _mock_llm(mocker, [])
        result = agent(_make_state())
        assert "style_findings" in result


# ===========================================================================
# DocsAgent
# ===========================================================================

class TestDocsAgentParsing:
    @pytest.fixture
    def agent(self):
        from code_reviewer.agents.docs import DocsAgent
        return DocsAgent()

    def test_valid_json_produces_correct_finding(self, agent, mocker):
        _mock_llm(mocker, [_finding_payload(_VALID_LINE, "LOW", "docs")])
        result = agent(_make_state())
        findings = result["docs_findings"]
        assert len(findings) == 1
        assert findings[0].source == "llm"
        assert findings[0].category == "docs"

    def test_malformed_json_returns_empty(self, agent, mocker):
        resp = MagicMock()
        resp.choices[0].message.content = "```\nnot valid\n```"
        mocker.patch("litellm.completion", return_value=resp)
        result = agent(_make_state())
        assert result["docs_findings"] == []

    def test_hallucinated_line_rejected(self, agent, mocker):
        _mock_llm(mocker, [_finding_payload(500, "INFO", "docs")])
        result = agent(_make_state())
        assert result["docs_findings"] == []

    def test_call_returns_correct_state_key(self, agent, mocker):
        _mock_llm(mocker, [])
        result = agent(_make_state())
        assert "docs_findings" in result

    def test_code_fence_response_parses_correctly(self, agent, mocker):
        """LLM wrapping its JSON in ```json ... ``` must still parse."""
        payload = json.dumps({
            "findings": [_finding_payload(_VALID_LINE, "INFO", "docs")]
        })
        resp = MagicMock()
        resp.choices[0].message.content = f"```json\n{payload}\n```"
        mocker.patch("litellm.completion", return_value=resp)
        result = agent(_make_state())
        assert len(result["docs_findings"]) == 1

    def test_multiple_valid_findings_all_returned(self, agent, mocker):
        """Multiple valid findings must all be returned."""
        added_lines = list(ADDED_LINE_NUMS)
        if len(added_lines) < 2:
            pytest.skip("Need at least 2 added lines in fixture")
        payload = [
            _finding_payload(added_lines[0], "LOW", "docs"),
            _finding_payload(added_lines[1], "INFO", "docs"),
        ]
        _mock_llm(mocker, payload)
        result = agent(_make_state())
        assert len(result["docs_findings"]) == 2
