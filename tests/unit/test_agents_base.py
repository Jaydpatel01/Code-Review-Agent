"""Unit tests for AgentState and BaseReviewAgent."""

import json
import pytest
from unittest.mock import MagicMock, patch

from code_reviewer.agents.state import AgentState
from code_reviewer.agents.base_agent import BaseReviewAgent
from code_reviewer.core.models import DiffHunk, Finding


# ---------------------------------------------------------------------------
# Concrete stub — minimum implementation for testing
# ---------------------------------------------------------------------------

class SecurityStubAgent(BaseReviewAgent):
    """Minimal concrete subclass for unit testing the base class."""

    @property
    def name(self) -> str:
        return "security"

    @property
    def category(self) -> str:
        return "security"

    @property
    def system_prompt(self) -> str:
        return "You are a security expert. Review the code for vulnerabilities."


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def agent() -> SecurityStubAgent:
    """Return a fresh SecurityStubAgent."""
    return SecurityStubAgent()


@pytest.fixture
def simple_hunk() -> DiffHunk:
    """A single hunk with two added lines and one context line."""
    return DiffHunk(
        file_path="src/auth.py",
        start_line=10,
        end_line=13,
        added_lines=[(11, '    password = "secret123"'), (12, "    return password")],
        removed_lines=[(11, "    password = None")],
        context_lines=[(10, "def get_password():")],
        raw_hunk="",
    )


@pytest.fixture
def multi_hunk() -> list[DiffHunk]:
    """Two hunks across two files."""
    return [
        DiffHunk(
            file_path="api/views.py",
            start_line=1,
            end_line=3,
            added_lines=[(2, "    token = request.GET['token']")],
            removed_lines=[],
            context_lines=[(1, "def view(request):")],
            raw_hunk="",
        ),
        DiffHunk(
            file_path="utils/db.py",
            start_line=20,
            end_line=22,
            added_lines=[(21, "    cursor.execute(query)")],
            removed_lines=[],
            context_lines=[(20, "def run_query(query):")],
            raw_hunk="",
        ),
    ]


@pytest.fixture
def base_state(simple_hunk) -> AgentState:
    """A minimal AgentState ready to pass to __call__."""
    return AgentState(
        hunks=[simple_hunk],
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


# ===========================================================================
# AgentState
# ===========================================================================

class TestAgentState:
    """Verify AgentState can be constructed and accessed as a TypedDict."""

    def test_state_construction(self, simple_hunk):
        state = AgentState(
            hunks=[simple_hunk],
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
        assert state["model"] == "gemini/gemini-3.1-flash-lite"
        assert state["context_lines_count"] == 5
        assert state["hunks"][0].file_path == "src/auth.py"
        assert state["error"] is None

    def test_state_all_finding_lists_present(self, base_state):
        for key in ("security_findings", "performance_findings",
                    "logic_findings", "style_findings", "docs_findings",
                    "final_findings"):
            assert key in base_state
            assert isinstance(base_state[key], list)


# ===========================================================================
# BaseReviewAgent — abstract interface
# ===========================================================================

class TestBaseReviewAgentAbstract:
    def test_cannot_instantiate_abc_directly(self):
        """BaseReviewAgent must be abstract — instantiation should raise TypeError."""
        with pytest.raises(TypeError):
            BaseReviewAgent()  # type: ignore[abstract]

    def test_concrete_subclass_properties(self, agent):
        assert agent.name == "security"
        assert agent.category == "security"
        assert "security" in agent.system_prompt.lower()


# ===========================================================================
# build_user_message
# ===========================================================================

class TestBuildUserMessage:

    def test_file_header_present(self, agent, simple_hunk):
        msg = agent.build_user_message([simple_hunk])
        assert "File: src/auth.py" in msg

    def test_line_range_header_present(self, agent, simple_hunk):
        msg = agent.build_user_message([simple_hunk])
        assert "@@ Lines 10–13 @@" in msg

    def test_added_lines_have_plus_prefix(self, agent, simple_hunk):
        msg = agent.build_user_message([simple_hunk])
        assert '+    password = "secret123"' in msg
        assert "+    return password" in msg

    def test_context_lines_have_space_prefix(self, agent, simple_hunk):
        msg = agent.build_user_message([simple_hunk])
        assert " def get_password():" in msg

    def test_removed_lines_are_excluded(self, agent, simple_hunk):
        """Removed lines must never appear in the user message."""
        msg = agent.build_user_message([simple_hunk])
        assert "password = None" not in msg

    def test_lines_sorted_by_line_number(self, agent, simple_hunk):
        """Context line (10) must appear before added lines (11, 12)."""
        msg = agent.build_user_message([simple_hunk])
        ctx_pos = msg.index("def get_password():")
        added_pos = msg.index('password = "secret123"')
        assert ctx_pos < added_pos

    def test_multi_hunk_both_files_present(self, agent, multi_hunk):
        msg = agent.build_user_message(multi_hunk)
        assert "File: api/views.py" in msg
        assert "File: utils/db.py" in msg

    def test_empty_hunks_returns_empty_string(self, agent):
        assert agent.build_user_message([]) == ""


# ===========================================================================
# parse_response
# ===========================================================================

class TestParseResponse:

    def _make_json(self, findings: list[dict]) -> str:
        return json.dumps({"findings": findings})

    def test_happy_path_returns_finding(self, agent, simple_hunk):
        raw = self._make_json([{
            "line_number": 11,
            "severity": "HIGH",
            "category": "security",
            "message": "Hardcoded password",
            "suggestion": "Use environment variable",
        }])
        result = agent.parse_response(raw, [simple_hunk])
        assert len(result) == 1
        assert result[0].line_number == 11
        assert result[0].severity == "HIGH"
        assert result[0].source == "llm"
        assert result[0].file_path == "src/auth.py"

    def test_category_always_overridden_to_agent_category(self, agent, simple_hunk):
        """LLM hallucinating the wrong category must be corrected."""
        raw = self._make_json([{
            "line_number": 11,
            "severity": "MEDIUM",
            "category": "style",      # wrong — agent is security
            "message": "Bad style",
            "suggestion": "Fix it",
        }])
        result = agent.parse_response(raw, [simple_hunk])
        assert result[0].category == "security"

    def test_source_always_set_to_llm(self, agent, simple_hunk):
        raw = self._make_json([{
            "line_number": 11,
            "severity": "LOW",
            "category": "security",
            "message": "Minor issue",
            "suggestion": "Review",
        }])
        result = agent.parse_response(raw, [simple_hunk])
        assert result[0].source == "llm"

    def test_hallucinated_line_dropped(self, agent, simple_hunk):
        """line_number not in added_lines must be silently dropped."""
        raw = self._make_json([{
            "line_number": 999,    # not in added_lines (11, 12)
            "severity": "HIGH",
            "category": "security",
            "message": "Hallucinated",
            "suggestion": "N/A",
        }])
        result = agent.parse_response(raw, [simple_hunk])
        assert result == []

    def test_valid_and_hallucinated_mixed(self, agent, simple_hunk):
        """Valid finding kept, hallucinated one dropped, in the same response."""
        raw = self._make_json([
            {"line_number": 11, "severity": "HIGH", "category": "security",
             "message": "Real finding", "suggestion": "Fix"},
            {"line_number": 500, "severity": "HIGH", "category": "security",
             "message": "Fake finding", "suggestion": "Fix"},
        ])
        result = agent.parse_response(raw, [simple_hunk])
        assert len(result) == 1
        assert result[0].message == "Real finding"

    def test_invalid_json_returns_empty(self, agent, simple_hunk):
        result = agent.parse_response("this is not json at all", [simple_hunk])
        assert result == []

    def test_empty_findings_list_returns_empty(self, agent, simple_hunk):
        raw = json.dumps({"findings": []})
        result = agent.parse_response(raw, [simple_hunk])
        assert result == []

    def test_code_fence_stripped_before_parse(self, agent, simple_hunk):
        """LLM responses wrapped in ```json ... ``` must still parse correctly."""
        inner = self._make_json([{
            "line_number": 12,
            "severity": "MEDIUM",
            "category": "security",
            "message": "In a fence",
            "suggestion": "Fix",
        }])
        fenced = f"```json\n{inner}\n```"
        result = agent.parse_response(fenced, [simple_hunk])
        assert len(result) == 1
        assert result[0].line_number == 12

    def test_none_line_number_uses_first_hunk_file(self, agent, simple_hunk):
        """A finding with null line_number should not be dropped (no line to validate)."""
        raw = self._make_json([{
            "line_number": None,
            "severity": "INFO",
            "category": "security",
            "message": "General note",
            "suggestion": "Review",
        }])
        result = agent.parse_response(raw, [simple_hunk])
        # None is not in valid_lines, so it passes the guard
        assert len(result) == 1
        assert result[0].file_path == "src/auth.py"

    def test_multi_hunk_file_path_resolved_correctly(self, agent, multi_hunk):
        """line_number from the second hunk must resolve to its file_path."""
        raw = self._make_json([{
            "line_number": 21,    # in utils/db.py hunk
            "severity": "HIGH",
            "category": "security",
            "message": "SQL injection",
            "suggestion": "Use parameterized queries",
        }])
        result = agent.parse_response(raw, multi_hunk)
        assert len(result) == 1
        assert result[0].file_path == "utils/db.py"


# ===========================================================================
# __call__ — LangGraph node protocol
# ===========================================================================

class TestAgentCall:

    def test_successful_call_returns_findings(self, agent, base_state, mocker):
        """A well-formed LLM response must populate the correct state key."""
        mock_response = MagicMock()
        mock_response.choices[0].message.content = json.dumps({
            "findings": [{
                "line_number": 11,
                "severity": "HIGH",
                "category": "security",
                "message": "Hardcoded password",
                "suggestion": "Use env var",
            }]
        })
        mocker.patch("litellm.completion", return_value=mock_response)

        result = agent(base_state)

        assert "security_findings" in result
        assert len(result["security_findings"]) == 1
        assert result["security_findings"][0].severity == "HIGH"
        assert result["error"] is None

    def test_call_uses_model_from_state(self, agent, base_state, mocker):
        """litellm.completion must be called with state['model']."""
        mock_response = MagicMock()
        mock_response.choices[0].message.content = json.dumps({"findings": []})
        mock_completion = mocker.patch("litellm.completion", return_value=mock_response)

        agent(base_state)

        call_kwargs = mock_completion.call_args
        assert call_kwargs.kwargs["model"] == "gemini/gemini-3.1-flash-lite"

    def test_call_sends_system_and_user_messages(self, agent, base_state, mocker):
        """Messages list must start with the system prompt."""
        mock_response = MagicMock()
        mock_response.choices[0].message.content = json.dumps({"findings": []})
        mock_completion = mocker.patch("litellm.completion", return_value=mock_response)

        agent(base_state)

        messages = mock_completion.call_args.kwargs["messages"]
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == agent.system_prompt
        assert messages[1]["role"] == "user"

    def test_call_on_exception_returns_empty_and_sets_error(self, agent, base_state, mocker):
        """Any exception must be caught — node returns empty findings + error key."""
        mocker.patch("litellm.completion", side_effect=RuntimeError("API down"))

        result = agent(base_state)

        assert result["security_findings"] == []
        assert result["error"] is not None
        assert "security" in result["error"]
        assert "API down" in result["error"]

    def test_call_returns_empty_on_unparseable_response(self, agent, base_state, mocker):
        """Garbage LLM content must not raise — returns empty findings."""
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Sorry I cannot help with that."
        mocker.patch("litellm.completion", return_value=mock_response)

        result = agent(base_state)

        assert result["security_findings"] == []
        assert result["error"] is None   # parse failure is not an "error"
