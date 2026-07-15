"""Unit tests for FileReviewer logic."""

import pytest
from code_reviewer.core.reviewer import FileReviewer, DiffReviewer, combine_findings
from code_reviewer.core.models import LLMReviewResponse, LLMFinding, ReviewResult, DiffHunk, Finding
from code_reviewer.config import Settings


def test_reviewer_filtering(mocker, tmp_path):
    """Test FileReviewer filtering by severity threshold and rule categories."""
    # 1. Create a dummy file to review
    dummy_file = tmp_path / "app.py"
    dummy_file.write_text("print('hello')\n")

    # 2. Mock LLMClient
    mock_client = mocker.MagicMock()

    # Mock LLM return containing findings with different categories and severities
    mock_findings = [
        LLMFinding(
            line_number=1,
            severity="HIGH",
            category="security",
            message="Hardcoded API Key",
            suggestion="Use env var",
        ),
        LLMFinding(
            line_number=2,
            severity="LOW",  # Will be filtered out since threshold is MEDIUM
            category="style",
            message="Trailing whitespace",
            suggestion="Remove it",
        ),
        LLMFinding(
            line_number=3,
            severity="HIGH",
            category="docs",  # Will be filtered out since rules.docs.enabled is False
            message="Missing docstring",
            suggestion="Add docstring",
        ),
    ]
    mock_client.generate_completion.return_value = LLMReviewResponse(
        findings=mock_findings, summary="Review Summary"
    )
    mock_client.model = "gemini-3.1-flash-lite"

    # 3. Setup settings
    settings = Settings()
    settings.severity_threshold = "MEDIUM"
    settings.rules.docs.enabled = False
    settings.rules.security.enabled = True

    # 4. Run Reviewer
    reviewer = FileReviewer(llm_client=mock_client, settings=settings)
    result = reviewer.review_file(str(dummy_file))

    # 5. Verify results
    assert isinstance(result, ReviewResult)
    assert result.file_path == str(dummy_file)
    assert result.lines_reviewed == 1

    # Only the HIGH security finding should remain
    assert len(result.findings) == 1
    assert result.findings[0].category == "security"
    assert result.findings[0].severity == "HIGH"
    assert result.findings[0].message == "Hardcoded API Key"


def test_diff_reviewer_filtering(mocker):
    """Test DiffReviewer filtering by severity threshold using the agent pipeline.

    DiffReviewer now calls run_agent_review() (multi-agent graph) instead of
    the retired per-hunk llm_client.generate_completion() path.
    This test mocks run_agent_review at the reviewer module level.
    """
    mock_client = mocker.MagicMock()
    mock_client.model = "gemini-3.1-flash-lite"

    # Findings that run_agent_review would return (already validated by agents)
    agent_findings = [
        Finding(
            file_path="app.py",
            line_number=11,   # valid added line
            severity="HIGH",
            category="security",
            message="Hardcoded API Key",
            suggestion="Use env var",
            source="llm",
        ),
        Finding(
            file_path="app.py",
            line_number=12,   # valid added line
            severity="MEDIUM",
            category="logic",
            message="Edge case",
            suggestion="Fix it",
            source="llm",
        ),
    ]

    # Patch run_agent_review at the location it is used in reviewer.py
    async def _fake_run(hunks, model, settings):
        return agent_findings

    mocker.patch(
        "code_reviewer.core.reviewer.run_agent_review",
        side_effect=_fake_run,
    )

    settings = Settings()
    settings.severity_threshold = "MEDIUM"

    reviewer = DiffReviewer(llm_client=mock_client, settings=settings)

    hunk = DiffHunk(
        file_path="app.py",
        start_line=10,
        end_line=13,
        added_lines=[(11, 'print("World")'), (12, "return 1")],
        removed_lines=[],
        context_lines=[(10, 'print("Hello")')],
        raw_hunk="",
    )

    result = reviewer.review_hunks("app.py", [hunk])

    assert isinstance(result, ReviewResult)
    assert result.file_path == "app.py"
    assert result.lines_reviewed == 2

    # Both findings should survive — both are >= MEDIUM
    assert len(result.findings) == 2
    categories = {f.category for f in result.findings}
    assert "security" in categories
    assert "logic" in categories

def test_combine_findings_deduplication():
    """Test that combine_findings properly deduplicates based on line_number and category."""
    static = [
        Finding(
            file_path="app.py",
            line_number=10,
            severity="MEDIUM",
            category="complexity",
            message="Static complexity",
            suggestion="Fix it",
            source="ast"
        )
    ]
    
    llm = [
        Finding(
            file_path="app.py",
            line_number=10,
            severity="HIGH",
            category="complexity",
            message="LLM complexity",
            suggestion="Fix it",
            source="llm"
        ),
        Finding(
            file_path="app.py",
            line_number=10,
            severity="MEDIUM",
            category="style",
            message="LLM style",
            suggestion="Fix it",
            source="llm"
        )
    ]
    
    combined = combine_findings(static, llm)
    
    assert len(combined) == 2
    
    # Static finding should be preserved
    assert combined[0].source == "ast"
    assert combined[0].message == "Static complexity"
    
    # LLM style finding on the same line is kept (different category)
    assert combined[1].source == "llm"
    assert combined[1].category == "style"


def test_apply_filters_none_settings():
    """Test that _apply_filters gracefully defaults to True when rules config is None."""
    from code_reviewer.core.reviewer import _apply_filters
    
    settings = Settings()
    settings.rules = None  # Force None
    
    findings = [
        Finding(
            file_path="app.py",
            line_number=10,
            severity="HIGH",
            category="complexity",
            message="Complexity finding",
            suggestion="Fix it",
            source="llm",
        )
    ]
    
    res = _apply_filters(findings, settings)
    assert len(res) == 1
    assert res[0].category == "complexity"

