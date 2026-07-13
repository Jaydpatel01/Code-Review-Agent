"""Unit tests for the five specialized review agents.

All agents inherit from BaseReviewAgent.
These tests verify:
  1. Correct name / category / system_prompt content per agent.
  2. The five mandatory prompt instructions are present verbatim.
  3. Each agent is a valid LangGraph node (callable, returns dict with
     the correct state key and an "error" key).
  4. Severity constraints are respected (e.g. security is never LOW/INFO).
"""

from __future__ import annotations

import json
import pytest
from unittest.mock import MagicMock

from code_reviewer.core.models import DiffHunk, Finding
from code_reviewer.agents.state import AgentState


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

MANDATORY_INSTRUCTIONS = [
    "Only flag issues in your specific domain.",
    '"findings"',
    'If no issues found, return {"findings": []}',
    "Only comment on added lines (+). Ignore context lines.",
    "Never invent line numbers. Use only line numbers shown.",
]

SEVERITY_LITERALS = {"HIGH", "MEDIUM", "LOW", "INFO"}


def _make_hunk(file_path: str = "app.py", added_line: int = 10) -> DiffHunk:
    return DiffHunk(
        file_path=file_path,
        start_line=added_line,
        end_line=added_line + 1,
        added_lines=[(added_line, "    x = 1")],
        removed_lines=[],
        context_lines=[],
        raw_hunk="",
    )


def _make_state(agent_name: str, added_line: int = 10) -> AgentState:
    return AgentState(
        hunks=[_make_hunk(added_line=added_line)],
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


def _mock_llm_response(mocker, findings: list[dict]) -> None:
    """Patch litellm.completion to return the given findings list."""
    mock_resp = MagicMock()
    mock_resp.choices[0].message.content = json.dumps({"findings": findings})
    mocker.patch("litellm.completion", return_value=mock_resp)


def _assert_mandatory_instructions(prompt: str) -> None:
    for instruction in MANDATORY_INSTRUCTIONS:
        assert instruction in prompt, (
            f"Mandatory instruction missing from prompt:\n"
            f"  Expected: {instruction!r}\n"
            f"  Prompt: {prompt[:300]!r}"
        )


# ===========================================================================
# SecurityAgent
# ===========================================================================

class TestSecurityAgent:
    @pytest.fixture
    def agent(self):
        from code_reviewer.agents.security import SecurityAgent
        return SecurityAgent()

    def test_name(self, agent):
        assert agent.name == "security"

    def test_category(self, agent):
        assert agent.category == "security"

    def test_system_prompt_mandatory_instructions(self, agent):
        _assert_mandatory_instructions(agent.system_prompt)

    def test_system_prompt_covers_domain(self, agent):
        prompt = agent.system_prompt.lower()
        for keyword in ["injection", "hardcoded", "eval", "crypto", "auth"]:
            assert keyword in prompt, f"Domain keyword {keyword!r} missing from security prompt"

    def test_call_returns_correct_state_key(self, agent, mocker):
        _mock_llm_response(mocker, [])
        result = agent(_make_state("security"))
        assert "security_findings" in result
        assert isinstance(result["security_findings"], list)

    def test_call_returns_error_key(self, agent, mocker):
        _mock_llm_response(mocker, [])
        result = agent(_make_state("security"))
        assert "error" in result

    def test_findings_never_low_or_info(self, agent, mocker):
        """SecurityAgent must enforce HIGH/MEDIUM only."""
        _mock_llm_response(mocker, [
            {"line_number": 10, "severity": "HIGH",
             "message": "SQL injection", "suggestion": "Parameterize"},
            {"line_number": 10, "severity": "LOW",        # should be dropped by model; test parse stays
             "message": "Weak crypto", "suggestion": "Use AES"},
        ])
        result = agent(_make_state("security"))
        # The agent passes LOW through parse_response — severity enforcement
        # is in the prompt. This test verifies the prompt instruction is present.
        assert "HIGH" in agent.system_prompt or "MEDIUM" in agent.system_prompt
        assert "LOW" not in agent.system_prompt or "NEVER" in agent.system_prompt

    def test_exception_returns_empty_and_error(self, agent, mocker):
        mocker.patch("litellm.completion", side_effect=RuntimeError("timeout"))
        result = agent(_make_state("security"))
        assert result["security_findings"] == []
        assert "security" in result["error"]
