"""Integration test for the full LangGraph multi-agent review pipeline.

Tests run_agent_review() end-to-end with:
  - Real parsed hunks from multi_file.diff
  - All five litellm.completion() calls mocked with pre-written responses
  - Deduplication assertion: same file + line + category with HIGH vs MEDIUM
    → aggregator must keep HIGH and discard MEDIUM
  - Multi-agent assertion: final_findings contains findings from >= 2 agents
"""

from __future__ import annotations

import json
import os
import pytest
from unittest.mock import MagicMock, call

from code_reviewer.analyzers.diff_parser import parse_diff
from code_reviewer.agents.graph import run_agent_review
from code_reviewer.config import Settings
from code_reviewer.core.models import DiffHunk

# ---------------------------------------------------------------------------
# Load multi_file.diff
# ---------------------------------------------------------------------------

DIFFS_DIR = os.path.join(
    os.path.dirname(__file__), "..", "fixtures", "sample_diffs"
)


def _load_hunks() -> list[DiffHunk]:
    with open(os.path.join(DIFFS_DIR, "multi_file.diff"), encoding="utf-8") as f:
        raw = f.read()
    return parse_diff(raw)


# multi_file.diff adds:
#   main.py  → line 1 (def init():)  and line 2 (    pass)
#   utils.py → line 6 (    b = 2)
MULTI_HUNKS = _load_hunks()

# Gather all valid added line numbers across both files
_ADDED: dict[str, set[int]] = {}
for h in MULTI_HUNKS:
    _ADDED.setdefault(h.file_path, set()).update(ln for ln, _ in h.added_lines)

# Pick one valid line from each file for deterministic test data
_MAIN_LINE = next(iter(_ADDED.get("main.py", {1})))
_UTILS_LINE = next(iter(_ADDED.get("utils.py", {6})))


# ---------------------------------------------------------------------------
# Mock response factory
# ---------------------------------------------------------------------------

def _resp(findings: list[dict]) -> MagicMock:
    m = MagicMock()
    m.choices[0].message.content = json.dumps({"findings": findings})
    return m


def _finding(file: str, line: int, severity: str, category: str, msg: str) -> dict:
    return {
        "line_number": line,
        "severity": severity,
        "category": category,
        "message": msg,
        "suggestion": "Fix it.",
    }


# ===========================================================================
# Integration tests
# ===========================================================================

class TestRunAgentReviewIntegration:

    @pytest.mark.anyio
    async def test_returns_list(self, mocker):
        """run_agent_review must always return a list."""
        mocker.patch("litellm.completion", return_value=_resp([]))
        result = await run_agent_review(MULTI_HUNKS, "gemini/gemini-3.1-flash-lite", Settings())
        assert isinstance(result, list)

    @pytest.mark.anyio
    async def test_deduplication_keeps_higher_severity(self, mocker):
        """Same file + line + category flagged by two agents: HIGH beats MEDIUM.

        We configure the security agent to return HIGH and the performance
        agent to return MEDIUM for the same (file, line, category) key.
        After aggregation only the HIGH finding should remain.
        """
        def _dispatch(**kwargs):
            """Route mock responses by the agent's system prompt keyword."""
            messages = kwargs.get("messages", [])
            system_text = messages[0]["content"].lower() if messages else ""
            if "sql injection" in system_text or "security vulnerabilit" in system_text or "security engineer" in system_text:
                return _resp([
                    _finding("main.py", _MAIN_LINE, "HIGH", "security", "SQL injection HIGH")
                ])
            if "n+1" in system_text or "performance engineer" in system_text:
                # Performance agent returns same line with different severity
                return _resp([
                    _finding("main.py", _MAIN_LINE, "MEDIUM", "security", "SQL injection MEDIUM dupe")
                ])
            return _resp([])

        mocker.patch("litellm.completion", side_effect=lambda *a, **kw: _dispatch(**kw))

        result = await run_agent_review(MULTI_HUNKS, "gemini/gemini-3.1-flash-lite", Settings())

        key_findings = [
            f for f in result
            if f.file_path == "main.py"
            and f.line_number == _MAIN_LINE
            and f.category == "security"
        ]
        assert len(key_findings) == 1, (
            f"Expected exactly 1 finding for dedup key, got {len(key_findings)}. All findings: {result}"
        )
        assert key_findings[0].severity == "HIGH", (
            f"Expected HIGH to win dedup, got {key_findings[0].severity}"
        )

    @pytest.mark.anyio
    async def test_findings_from_at_least_two_agents(self, mocker):
        """final_findings must contain findings from >= 2 distinct agent categories.

        We configure the security agent to flag main.py and the logic agent
        to flag utils.py, on different lines so they don't deduplicate.
        """
        def _dispatch(**kwargs):
            messages = kwargs.get("messages", [])
            system_text = messages[0]["content"].lower() if messages else ""
            if "security engineer" in system_text:
                return _resp([
                    _finding("main.py", _MAIN_LINE, "HIGH", "security", "Hardcoded secret")
                ])
            if "logic" in system_text and "correctness" in system_text:
                return _resp([
                    _finding("utils.py", _UTILS_LINE, "MEDIUM", "logic", "Missing None check")
                ])
            return _resp([])

        mocker.patch("litellm.completion", side_effect=lambda *a, **kw: _dispatch(**kw))

        result = await run_agent_review(MULTI_HUNKS, "gemini/gemini-3.1-flash-lite", Settings())

        categories = {f.category for f in result}
        assert len(categories) >= 2, (
            f"Expected findings from >= 2 agents, got categories: {categories}"
        )
        assert "security" in categories
        assert "logic" in categories

    @pytest.mark.anyio
    async def test_hallucinated_lines_excluded_from_final(self, mocker):
        """Findings on lines not in any hunk must not appear in final_findings."""
        # All agents return a finding on line 9999 (not in multi_file.diff)
        bad_resp = _resp([
            _finding("main.py", 9999, "HIGH", "security", "Hallucinated")
        ])
        mocker.patch("litellm.completion", return_value=bad_resp)

        result = await run_agent_review(MULTI_HUNKS, "gemini/gemini-3.1-flash-lite", Settings())

        hallucinated = [f for f in result if f.line_number == 9999]
        assert hallucinated == [], f"Hallucinated findings survived: {hallucinated}"

    @pytest.mark.anyio
    async def test_all_agents_empty_returns_empty(self, mocker):
        """If every agent returns no findings, final_findings must be empty."""
        mocker.patch("litellm.completion", return_value=_resp([]))
        result = await run_agent_review(MULTI_HUNKS, "gemini/gemini-3.1-flash-lite", Settings())
        assert result == []

    @pytest.mark.anyio
    async def test_agent_exception_does_not_crash_pipeline(self, mocker):
        """An agent raising an exception must not crash the whole pipeline.

        The other agents' findings should still be returned.
        """
        # First call (security) raises; remaining four return empty.
        mocker.patch(
            "litellm.completion",
            side_effect=[
                RuntimeError("API unavailable"),
                _resp([]),
                _resp([]),
                _resp([]),
                _resp([]),
            ],
        )

        # Should not raise
        result = await run_agent_review(MULTI_HUNKS, "gemini/gemini-3.1-flash-lite", Settings())
        assert isinstance(result, list)
