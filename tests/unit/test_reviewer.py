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
    """Test DiffReviewer filtering by severity threshold and mapping line numbers."""
    mock_client = mocker.MagicMock()

    mock_findings = [
        LLMFinding(
            line_number=2,  # hallucinated line number (not in added lines)
            severity="HIGH",
            category="security",
            message="Hardcoded API Key",
            suggestion="Use env var",
        ),
        LLMFinding(
            line_number=12,  # valid line number in added lines
            severity="MEDIUM",
            category="logic",
            message="Edge case",
            suggestion="Fix it",
        ),
    ]
    mock_client.generate_completion.return_value = LLMReviewResponse(
        findings=mock_findings, summary="Diff Summary"
    )
    mock_client.model = "gemini-3.1-flash-lite"

    settings = Settings()
    settings.severity_threshold = "MEDIUM"

    reviewer = DiffReviewer(llm_client=mock_client, settings=settings)

    # Create dummy hunk
    hunk = DiffHunk(
        file_path="app.py",
        start_line=10,
        end_line=13,
        added_lines=[(11, 'print("World")'), (12, 'return 1')],
        removed_lines=[],
        context_lines=[(10, 'print("Hello")')],
        raw_hunk=""
    )

    result = reviewer.review_hunks("app.py", [hunk])

    assert isinstance(result, ReviewResult)
    assert result.file_path == "app.py"
    assert result.lines_reviewed == 2

    # Both should remain since severities are >= MEDIUM
    assert len(result.findings) == 2

    finding_1 = result.findings[0]
    # The hallucinated line 2 should be mapped to 11 (first added line)
    assert finding_1.line_number == 11
    assert finding_1.category == "security"

    finding_2 = result.findings[1]
    # Valid line 12 stays 12
    assert finding_2.line_number == 12
    assert finding_2.category == "logic"

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
