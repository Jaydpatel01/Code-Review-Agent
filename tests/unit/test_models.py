"""Unit tests for models.py."""

from datetime import datetime, timezone
from code_reviewer.core.models import Finding, ReviewResult


def test_finding_creation():
    """Test creating a Finding model with valid fields."""
    finding = Finding(
        file_path="app.py",
        line_number=10,
        severity="HIGH",
        category="security",
        message="Vulnerability found",
        suggestion="Fix it",
        source="llm"
    )
    assert finding.file_path == "app.py"
    assert finding.line_number == 10
    assert finding.severity == "HIGH"
    assert finding.category == "security"
    assert finding.source == "llm"


def test_review_result_creation():
    """Test creating a ReviewResult model and checking default timestamp."""
    result = ReviewResult(
        file_path="app.py",
        findings=[],
        summary="No issues",
        model_used="gemini-3.1-flash-lite",
        lines_reviewed=50
    )
    assert result.file_path == "app.py"
    assert len(result.findings) == 0
    assert isinstance(result.reviewed_at, datetime)
    assert result.reviewed_at.tzinfo == timezone.utc
