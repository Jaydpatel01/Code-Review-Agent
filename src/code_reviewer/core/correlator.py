"""Cross-file finding correlator for root cause analysis.

Analyzes findings across multiple files to identify patterns, call chain
propagation, and shared root causes using the dependency graph.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

from code_reviewer.core.models import Finding
from code_reviewer.indexer.dependency_graph import DependencyGraph

logger = logging.getLogger(__name__)


@dataclass
class CorrelatedFinding:
    """A finding pattern that spans multiple files.

    Attributes:
        pattern: Human-readable description of the pattern
        severity: Highest severity among all findings in this group
        affected_files: List of file paths with this pattern
        affected_lines: List of (file_path, line_number) tuples
        root_cause_file: File path of the suspected root cause (if identified)
        root_cause_line: Line number of the suspected root cause
        root_cause_reason: Explanation of why this is the root cause
        individual_findings: The original Finding objects in this group
    """

    pattern: str
    severity: str
    affected_files: list[str]
    affected_lines: list[tuple[str, int]]
    root_cause_file: Optional[str] = None
    root_cause_line: Optional[int] = None
    root_cause_reason: Optional[str] = None
    individual_findings: list[Finding] = field(default_factory=list)


class CrossFileCorrelator:
    """Correlates findings across multiple files to identify patterns and root causes.

    Uses three correlation passes:
    1. Pattern grouping: Groups findings with similar messages across files
    2. Call chain propagation: Identifies issues that propagate through call chains
    3. Root cause identification: Finds shared dependencies as likely root causes
    """

    def __init__(self, graph: Optional[DependencyGraph] = None):
        """Initialize the correlator.

        Args:
            graph: Optional dependency graph for call chain and root cause analysis
        """
        self.graph = graph

    def correlate(
        self, all_findings: dict[str, list[Finding]]
    ) -> list[CorrelatedFinding]:
        """Correlate findings across multiple files.

        Args:
            all_findings: Dictionary mapping file_path → list of findings

        Returns:
            List of CorrelatedFinding objects representing cross-file patterns.
            Single-file groups are excluded.
        """
        correlated: list[CorrelatedFinding] = []

        # Pass 1: Pattern grouping by message similarity
        pattern_groups = self._group_by_pattern(all_findings)
        for group in pattern_groups:
            # Only report cross-file patterns (2+ files)
            if len(set(f.file_path for f in group)) >= 2:
                correlated.append(self._create_correlated_finding(group))

        # Pass 2: Call chain propagation (requires graph)
        if self.graph is not None:
            call_chain_findings = self._find_call_chain_propagation(all_findings)
            correlated.extend(call_chain_findings)

        # Pass 3: Shared dependency root cause (requires graph)
        if self.graph is not None:
            self._identify_root_causes(correlated)

        # Deduplicate based on affected files + pattern
        deduplicated = self._deduplicate_correlated(correlated)

        logger.info(f"Correlation complete: {len(deduplicated)} cross-file patterns found")
        return deduplicated

    def _group_by_pattern(
        self, all_findings: dict[str, list[Finding]]
    ) -> list[list[Finding]]:
        """Group findings by message pattern similarity.

        Returns:
            List of groups, where each group is a list of findings with
            similar messages (>60% token overlap).
        """
        # Flatten all findings
        flat_findings: list[Finding] = []
        for findings in all_findings.values():
            flat_findings.extend(findings)

        if not flat_findings:
            return []

        # Greedy clustering by similarity
        groups: list[list[Finding]] = []
        used: set[int] = set()

        for i, finding in enumerate(flat_findings):
            if i in used:
                continue

            # Start a new group
            group = [finding]
            used.add(i)

            # Find similar findings
            for j, other in enumerate(flat_findings):
                if j <= i or j in used:
                    continue

                similarity = self._message_token_overlap(
                    finding.message, other.message
                )

                if similarity > 0.6:  # 60% threshold
                    group.append(other)
                    used.add(j)

            groups.append(group)

        return groups

    def _message_token_overlap(self, msg1: str, msg2: str) -> float:
        """Calculate Jaccard similarity between two messages.

        Tokenizes messages and computes the ratio of shared tokens to
        total unique tokens.

        Args:
            msg1: First message
            msg2: Second message

        Returns:
            Similarity score between 0.0 and 1.0
        """
        # Simple tokenization: split on whitespace and punctuation
        import re

        def tokenize(text: str) -> set[str]:
            # Convert to lowercase and split on non-alphanumeric
            tokens = re.findall(r"\b\w+\b", text.lower())
            return set(tokens)

        tokens1 = tokenize(msg1)
        tokens2 = tokenize(msg2)

        if not tokens1 and not tokens2:
            return 1.0  # Both empty

        if not tokens1 or not tokens2:
            return 0.0  # One empty

        intersection = tokens1 & tokens2
        union = tokens1 | tokens2

        return len(intersection) / len(union)

    def _create_correlated_finding(
        self, findings: list[Finding]
    ) -> CorrelatedFinding:
        """Create a CorrelatedFinding from a group of similar findings.

        Args:
            findings: Group of findings with similar patterns

        Returns:
            CorrelatedFinding object summarizing the group
        """
        # Get the pattern from the first finding's message
        pattern = findings[0].message

        # Get highest severity
        severity_order = {"HIGH": 4, "MEDIUM": 3, "LOW": 2, "INFO": 1}
        severity = max(
            findings, key=lambda f: severity_order.get(f.severity, 0)
        ).severity

        # Collect affected files and lines
        affected_files = sorted(set(f.file_path for f in findings))
        affected_lines = [(f.file_path, f.line_number) for f in findings]

        return CorrelatedFinding(
            pattern=pattern,
            severity=severity,
            affected_files=affected_files,
            affected_lines=affected_lines,
            individual_findings=findings,
        )

    def _find_call_chain_propagation(
        self, all_findings: dict[str, list[Finding]]
    ) -> list[CorrelatedFinding]:
        """Find issues that propagate through call chains.

        Identifies cases where function A has a finding, calls function B,
        and B also has a finding in the same category.

        Args:
            all_findings: Dictionary mapping file_path → list of findings

        Returns:
            List of CorrelatedFinding objects representing call chains
        """
        if self.graph is None:
            return []

        call_chain_findings: list[CorrelatedFinding] = []

        # For each HIGH finding
        for file_path, findings in all_findings.items():
            for finding in findings:
                if finding.severity != "HIGH":
                    continue

                # Try to find the node key for this finding
                # This is a best-effort attempt - we need the function name
                # which isn't directly in the Finding object
                # For now, skip if we can't determine the function
                # TODO: Enhance Finding to include function_name
                continue

        return call_chain_findings

    def _identify_root_causes(
        self, correlated_findings: list[CorrelatedFinding]
    ) -> None:
        """Identify root causes for correlated findings using the dependency graph.

        For each correlated finding group, finds common callees shared by
        all affected functions. The most central common callee is marked
        as the likely root cause.

        Modifies correlated_findings in place.

        Args:
            correlated_findings: List of correlated findings to analyze
        """
        if self.graph is None:
            return

        for correlated in correlated_findings:
            # Skip if already has a root cause
            if correlated.root_cause_file is not None:
                continue

            # Try to find common callees for all affected files
            # This requires mapping files to their function nodes
            # For now, this is a stub - full implementation requires
            # tracking which functions have which findings
            # TODO: Enhance to track function-level findings
            pass

    def _deduplicate_correlated(
        self, findings: list[CorrelatedFinding]
    ) -> list[CorrelatedFinding]:
        """Remove duplicate correlated findings.

        Deduplicates based on the combination of affected files and pattern.

        Args:
            findings: List of correlated findings

        Returns:
            Deduplicated list
        """
        seen: set[tuple] = set()
        deduplicated: list[CorrelatedFinding] = []

        for finding in findings:
            # Create a unique key from affected files and pattern
            key = (tuple(sorted(finding.affected_files)), finding.pattern)

            if key not in seen:
                seen.add(key)
                deduplicated.append(finding)

        return deduplicated
