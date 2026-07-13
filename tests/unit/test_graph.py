"""Unit tests for the LangGraph orchestrator graph (graph.py).

Tests cover:
  - aggregate_findings deduplication logic (pure function, no LLM)
  - review_graph structure (nodes, edges)
  - run_agent_review end-to-end with mocked agents
"""

from __future__ import annotations

import json
import pytest
from unittest.mock import MagicMock, patch

from code_reviewer.core.models import DiffHunk, Finding
from code_reviewer.agents.state import AgentState
from code_reviewer.agents.graph import aggregate_findings, review_graph, run_agent_review


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _finding(
    file_path: str = "app.py",
    line_number: int = 10,
    severity: str = "HIGH",
    category: str = "security",
    message: str = "Test finding",
) -> Finding:
    return Finding(
        file_path=file_path,
        line_number=line_number,
        severity=severity,
        category=category,
        message=message,
        suggestion="Fix it.",
        source="llm",
    )


def _make_hunk(added_line: int = 10) -> DiffHunk:
    return DiffHunk(
        file_path="app.py",
        start_line=added_line,
        end_line=added_line + 1,
        added_lines=[(added_line, "    x = secret_key")],
        removed_lines=[],
        context_lines=[],
        raw_hunk="",
    )


def _base_state(**overrides) -> AgentState:
    base = AgentState(
        hunks=[_make_hunk()],
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
    base.update(overrides)
    return base


# ===========================================================================
# aggregate_findings — pure function, no LLM needed
# ===========================================================================

class TestAggregateFindings:

    def test_empty_state_returns_empty_list(self):
        state = _base_state()
        result = aggregate_findings(state)
        assert result["final_findings"] == []

    def test_single_finding_passes_through(self):
        f = _finding()
        state = _base_state(security_findings=[f])
        result = aggregate_findings(state)
        assert len(result["final_findings"]) == 1
        assert result["final_findings"][0] == f

    def test_findings_from_all_agents_collected(self):
        state = _base_state(
            security_findings=[_finding(category="security", line_number=1)],
            performance_findings=[_finding(category="performance", line_number=2)],
            logic_findings=[_finding(category="logic", line_number=3)],
            style_findings=[_finding(category="style", line_number=4, severity="LOW")],
            docs_findings=[_finding(category="docs", line_number=5, severity="INFO")],
        )
        result = aggregate_findings(state)
        assert len(result["final_findings"]) == 5

    def test_duplicate_same_key_keeps_higher_severity(self):
        """security + logic both flag line 10 as 'security' → keep HIGH."""
        low = _finding(category="security", line_number=10, severity="LOW")
        high = _finding(category="security", line_number=10, severity="HIGH")
        state = _base_state(security_findings=[low], logic_findings=[high])
        result = aggregate_findings(state)
        assert len(result["final_findings"]) == 1
        assert result["final_findings"][0].severity == "HIGH"

    def test_duplicate_keeps_existing_if_already_higher(self):
        high = _finding(category="security", line_number=10, severity="HIGH")
        medium = _finding(category="security", line_number=10, severity="MEDIUM")
        state = _base_state(security_findings=[high], performance_findings=[medium])
        result = aggregate_findings(state)
        assert result["final_findings"][0].severity == "HIGH"

    def test_different_category_same_line_both_kept(self):
        """Same file+line but different categories are NOT duplicates."""
        sec = _finding(category="security", line_number=10, severity="HIGH")
        perf = _finding(category="performance", line_number=10, severity="MEDIUM")
        state = _base_state(security_findings=[sec], performance_findings=[perf])
        result = aggregate_findings(state)
        assert len(result["final_findings"]) == 2

    def test_different_line_same_category_both_kept(self):
        f1 = _finding(category="security", line_number=10)
        f2 = _finding(category="security", line_number=20)
        state = _base_state(security_findings=[f1, f2])
        result = aggregate_findings(state)
        assert len(result["final_findings"]) == 2

    def test_severity_order_info_lowest(self):
        """INFO should lose to any other severity."""
        info = _finding(severity="INFO", line_number=5, category="style")
        low = _finding(severity="LOW", line_number=5, category="style")
        state = _base_state(style_findings=[info], docs_findings=[low])
        result = aggregate_findings(state)
        assert result["final_findings"][0].severity == "LOW"

    def test_all_severity_levels_rank_correctly(self):
        """HIGH > MEDIUM > LOW > INFO for same key."""
        severities = ["MEDIUM", "INFO", "HIGH", "LOW"]
        findings = [_finding(severity=s, line_number=1, category="logic") for s in severities]
        state = _base_state(logic_findings=findings)
        result = aggregate_findings(state)
        assert len(result["final_findings"]) == 1
        assert result["final_findings"][0].severity == "HIGH"

    def test_returns_dict_with_final_findings_key(self):
        result = aggregate_findings(_base_state())
        assert "final_findings" in result
        assert isinstance(result["final_findings"], list)


# ===========================================================================
# review_graph — structural checks
# ===========================================================================

class TestReviewGraph:

    def test_graph_compiled(self):
        """review_graph must be importable and non-None."""
        assert review_graph is not None

    def test_graph_has_all_nodes(self):
        """All five agent nodes and the aggregate node must be registered."""
        nodes = set(review_graph.nodes.keys())
        for expected in ("security", "performance", "logic", "style", "docs", "aggregate"):
            assert expected in nodes, f"Node '{expected}' missing from review_graph"


# ===========================================================================
# run_agent_review — end-to-end with mocked LLM
# ===========================================================================

class TestRunAgentReview:

    @pytest.fixture
    def mock_llm_empty(self, mocker):
        """Patch litellm.completion to return an empty findings list."""
        mock_resp = MagicMock()
        mock_resp.choices[0].message.content = json.dumps({"findings": []})
        mocker.patch("litellm.completion", return_value=mock_resp)

    @pytest.fixture
    def mock_llm_security_finding(self, mocker):
        """Patch litellm.completion to return one security finding on line 10."""
        mock_resp = MagicMock()
        mock_resp.choices[0].message.content = json.dumps({
            "findings": [{
                "line_number": 10,
                "severity": "HIGH",
                "category": "security",
                "message": "Hardcoded secret key detected.",
                "suggestion": "Use an environment variable.",
            }]
        })
        mocker.patch("litellm.completion", return_value=mock_resp)

    @pytest.mark.anyio
    async def test_returns_list(self, mock_llm_empty):
        from code_reviewer.config import Settings
        result = await run_agent_review(
            hunks=[_make_hunk()],
            model="gemini/gemini-3.1-flash-lite",
            settings=Settings(),
        )
        assert isinstance(result, list)

    @pytest.mark.anyio
    async def test_empty_llm_returns_empty_findings(self, mock_llm_empty):
        from code_reviewer.config import Settings
        result = await run_agent_review(
            hunks=[_make_hunk()],
            model="gemini/gemini-3.1-flash-lite",
            settings=Settings(),
        )
        assert result == []

    @pytest.mark.anyio
    async def test_security_finding_returned(self, mock_llm_security_finding):
        """A valid security finding on an added line must survive the pipeline."""
        from code_reviewer.config import Settings
        result = await run_agent_review(
            hunks=[_make_hunk(added_line=10)],
            model="gemini/gemini-3.1-flash-lite",
            settings=Settings(),
        )
        security = [f for f in result if f.category == "security" and f.severity == "HIGH"]
        assert len(security) >= 1

    @pytest.mark.anyio
    async def test_deduplication_in_pipeline(self, mocker):
        """When all five agents return the same finding, only one survives."""
        mock_resp = MagicMock()
        mock_resp.choices[0].message.content = json.dumps({
            "findings": [{
                "line_number": 10,
                "severity": "MEDIUM",
                "category": "security",
                "message": "Same finding from every agent.",
                "suggestion": "Fix it.",
            }]
        })
        mocker.patch("litellm.completion", return_value=mock_resp)

        from code_reviewer.config import Settings
        result = await run_agent_review(
            hunks=[_make_hunk(added_line=10)],
            model="gemini/gemini-3.1-flash-lite",
            settings=Settings(),
        )
        # All 5 agents would produce "security" findings on line 10.
        # After dedup only 1 should remain.
        sec_line10 = [f for f in result if f.category == "security" and f.line_number == 10]
        assert len(sec_line10) == 1
