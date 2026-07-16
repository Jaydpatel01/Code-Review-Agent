"""Unit tests for CrossFileCorrelator."""

import pytest
from code_reviewer.core.correlator import CrossFileCorrelator, CorrelatedFinding
from code_reviewer.core.models import Finding


@pytest.fixture
def sample_findings():
    """Create sample findings for testing."""
    finding1 = Finding(
        file_path="file1.py",
        line_number=10,
        severity="HIGH",
        category="security",
        message="SQL injection vulnerability detected",
        suggestion="Use parameterized queries",
        source="llm",
    )
    
    finding2 = Finding(
        file_path="file2.py",
        line_number=20,
        severity="MEDIUM",
        category="security",
        message="SQL injection vulnerability detected",
        suggestion="Use parameterized queries",
        source="llm",
    )
    
    finding3 = Finding(
        file_path="file3.py",
        line_number=30,
        severity="LOW",
        category="style",
        message="Line too long",
        suggestion="Break into multiple lines",
        source="llm",
    )
    
    return {
        "file1.py": [finding1],
        "file2.py": [finding2],
        "file3.py": [finding3],
    }


def test_message_token_overlap_identical():
    """Test token overlap for identical messages."""
    correlator = CrossFileCorrelator()
    
    msg1 = "SQL injection vulnerability detected"
    msg2 = "SQL injection vulnerability detected"
    
    overlap = correlator._message_token_overlap(msg1, msg2)
    assert overlap == 1.0


def test_message_token_overlap_different():
    """Test token overlap for completely different messages."""
    correlator = CrossFileCorrelator()
    
    msg1 = "SQL injection vulnerability detected"
    msg2 = "Line too long needs fixing"
    
    overlap = correlator._message_token_overlap(msg1, msg2)
    assert overlap < 0.3


def test_message_token_overlap_partial():
    """Test token overlap for partially similar messages."""
    correlator = CrossFileCorrelator()
    
    msg1 = "SQL injection vulnerability detected in query"
    msg2 = "SQL injection vulnerability found"
    
    overlap = correlator._message_token_overlap(msg1, msg2)
    assert 0.4 < overlap < 0.9


def test_correlate_pattern_grouping(sample_findings):
    """Test correlation pass 1: pattern grouping by message similarity."""
    correlator = CrossFileCorrelator(graph=None)
    
    correlated = correlator.correlate(sample_findings)
    
    # Should find 1 cross-file pattern (SQL injection in file1 and file2)
    assert len(correlated) == 1
    
    finding = correlated[0]
    assert "SQL injection" in finding.pattern
    assert finding.severity == "HIGH"  # Max severity
    assert len(finding.affected_files) == 2
    assert "file1.py" in finding.affected_files
    assert "file2.py" in finding.affected_files


def test_correlate_same_file_not_grouped():
    """Test that findings in the same file are not correlated."""
    finding1 = Finding(
        file_path="file1.py",
        line_number=10,
        severity="HIGH",
        category="security",
        message="SQL injection vulnerability",
        suggestion="Fix it",
        source="llm",
    )
    
    finding2 = Finding(
        file_path="file1.py",
        line_number=20,
        severity="MEDIUM",
        category="security",
        message="SQL injection vulnerability",
        suggestion="Fix it",
        source="llm",
    )
    
    all_findings = {"file1.py": [finding1, finding2]}
    
    correlator = CrossFileCorrelator(graph=None)
    correlated = correlator.correlate(all_findings)
    
    # Should not create a correlated finding (same file)
    assert len(correlated) == 0


def test_correlate_low_overlap_not_grouped():
    """Test that findings with low message overlap are not grouped."""
    finding1 = Finding(
        file_path="file1.py",
        line_number=10,
        severity="HIGH",
        category="security",
        message="SQL injection vulnerability detected",
        suggestion="Fix it",
        source="llm",
    )
    
    finding2 = Finding(
        file_path="file2.py",
        line_number=20,
        severity="MEDIUM",
        category="style",
        message="Line is too long",
        suggestion="Fix it",
        source="llm",
    )
    
    all_findings = {
        "file1.py": [finding1],
        "file2.py": [finding2],
    }
    
    correlator = CrossFileCorrelator(graph=None)
    correlated = correlator.correlate(all_findings)
    
    # Should not group (< 60% overlap)
    assert len(correlated) == 0


def test_create_correlated_finding():
    """Test creating a CorrelatedFinding from a group."""
    finding1 = Finding(
        file_path="file1.py",
        line_number=10,
        severity="HIGH",
        category="security",
        message="SQL injection",
        suggestion="Fix it",
        source="llm",
    )
    
    finding2 = Finding(
        file_path="file2.py",
        line_number=20,
        severity="MEDIUM",
        category="security",
        message="SQL injection",
        suggestion="Fix it",
        source="llm",
    )
    
    correlator = CrossFileCorrelator()
    correlated = correlator._create_correlated_finding([finding1, finding2])
    
    assert correlated.pattern == "SQL injection"
    assert correlated.severity == "HIGH"  # Max severity
    assert len(correlated.affected_files) == 2
    assert "file1.py" in correlated.affected_files
    assert "file2.py" in correlated.affected_files
    assert len(correlated.affected_lines) == 2
    assert ("file1.py", 10) in correlated.affected_lines
    assert ("file2.py", 20) in correlated.affected_lines
    assert len(correlated.individual_findings) == 2


def test_deduplicate_correlated():
    """Test deduplication of correlated findings."""
    finding1 = CorrelatedFinding(
        pattern="SQL injection",
        severity="HIGH",
        affected_files=["file1.py", "file2.py"],
        affected_lines=[("file1.py", 10), ("file2.py", 20)],
        individual_findings=[],
    )
    
    # Duplicate with same pattern and files
    finding2 = CorrelatedFinding(
        pattern="SQL injection",
        severity="HIGH",
        affected_files=["file1.py", "file2.py"],
        affected_lines=[("file1.py", 10), ("file2.py", 20)],
        individual_findings=[],
    )
    
    # Different pattern
    finding3 = CorrelatedFinding(
        pattern="XSS vulnerability",
        severity="MEDIUM",
        affected_files=["file1.py", "file3.py"],
        affected_lines=[("file1.py", 15), ("file3.py", 25)],
        individual_findings=[],
    )
    
    correlator = CrossFileCorrelator()
    deduplicated = correlator._deduplicate_correlated([finding1, finding2, finding3])
    
    # Should remove the duplicate
    assert len(deduplicated) == 2
    patterns = {f.pattern for f in deduplicated}
    assert "SQL injection" in patterns
    assert "XSS vulnerability" in patterns


def test_correlate_empty_findings():
    """Test correlation with empty findings dict."""
    correlator = CrossFileCorrelator(graph=None)
    correlated = correlator.correlate({})
    
    assert len(correlated) == 0


def test_correlate_single_finding():
    """Test correlation with only one finding."""
    finding = Finding(
        file_path="file1.py",
        line_number=10,
        severity="HIGH",
        category="security",
        message="SQL injection",
        suggestion="Fix it",
        source="llm",
    )
    
    correlator = CrossFileCorrelator(graph=None)
    correlated = correlator.correlate({"file1.py": [finding]})
    
    # Single finding should not be correlated
    assert len(correlated) == 0


def test_group_by_pattern(sample_findings):
    """Test _group_by_pattern method."""
    correlator = CrossFileCorrelator()
    groups = correlator._group_by_pattern(sample_findings)
    
    # Should have 2 groups: SQL injection group and style group
    assert len(groups) == 2
    
    # Find the SQL injection group
    sql_group = [g for g in groups if len(g) == 2][0]
    assert len(sql_group) == 2
    assert all("SQL injection" in f.message for f in sql_group)


def test_severity_precedence():
    """Test that highest severity wins in correlated findings."""
    finding1 = Finding(
        file_path="file1.py",
        line_number=10,
        severity="LOW",
        category="security",
        message="Security issue detected",
        suggestion="Fix it",
        source="llm",
    )
    
    finding2 = Finding(
        file_path="file2.py",
        line_number=20,
        severity="HIGH",
        category="security",
        message="Security issue detected",
        suggestion="Fix it",
        source="llm",
    )
    
    finding3 = Finding(
        file_path="file3.py",
        line_number=30,
        severity="MEDIUM",
        category="security",
        message="Security issue detected",
        suggestion="Fix it",
        source="llm",
    )
    
    all_findings = {
        "file1.py": [finding1],
        "file2.py": [finding2],
        "file3.py": [finding3],
    }
    
    correlator = CrossFileCorrelator(graph=None)
    correlated = correlator.correlate(all_findings)
    
    assert len(correlated) == 1
    assert correlated[0].severity == "HIGH"  # Highest severity wins
