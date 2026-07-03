"""Unit tests for FileReviewer logic."""

import pytest
from code_reviewer.core.reviewer import FileReviewer
from code_reviewer.core.models import LLMReviewResponse, LLMFinding, ReviewResult
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
